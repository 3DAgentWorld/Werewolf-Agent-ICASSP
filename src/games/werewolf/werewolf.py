#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: werewolf.py 
# @date: 2024/3/18 10:58 
#
# describe:
#
import copy
import json
import os
import random
import re
from typing import Dict, Tuple, Type, List
from collections import Counter

from colorama import Fore

from ..abs_game import Game
from ...agents.abs_agent import Agent
from ...extractor.abs_extractor import Extractor
from ...utils import print_text_animated, write_json, create_dir, COLOR


class Werewolf(Game):
    def __init__(self, player_nums: int, language: str, mode: str, ai_model, output_dir, **kwargs):
        config_file = kwargs.get("config_file")
        if not config_file:
            config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data: dict = json.load(f)
        support_nums = config_data.get("player_nums", [])
        assert player_nums in support_nums, f"game includes {player_nums} players is not supported"

        game_config = config_data.get(language, {})
        if not game_config:
            raise NotImplementedError(f'{language} is not supported.')

        host_instruction: dict = game_config.get('host_instruction')
        response_rule: dict = game_config.get('response_rule')
        role_introduce = game_config.get('role_introduce')
        config_for_num = game_config.get(f'config_{player_nums}', {})
        game_introduce = config_for_num.get('game_introduce', '')
        roles = config_for_num.get('role', [])
        villager_camp = config_for_num.get('villager_camp', [])
        werewolf_camp = config_for_num.get('werewolf_camp', [])
        role_mapping = game_config.get('role_mapping', {})
        create_dir(output_dir)

        # instruction
        self.language = language
        self.host_instruction = host_instruction
        self.role_introduce = role_introduce
        self.game_introduce = game_introduce

        # player
        self.player_nums = player_nums
        self.player_list = []
        self.players: Dict[str, Agent] = {}
        self.alive_players = []
        self.player_mapping = {}

        # role
        self.alive_roles = []
        self.roles = roles
        self.role_mapping = role_mapping

        self.seer_player = None
        self.guard_player = None
        self.witch_player = None
        self.werewolf_players = []
        self.villager_players = []

        self.mode = mode
        self.ai_model = ai_model
        self.process_list = []
        self.output_dir = output_dir

        # skill
        self.antidote_skill = True
        self.poison_skill = True

        # AI extractor
        self.player_extractor = None
        self.antidote_extractor = None
        self.killing_agreement_extractor = None

        self.day_count = 0
        self.winners = []
        self.response_rule = response_rule

    def init_extractor(self, player_extractor: Tuple[Type[Extractor], dict],
                       antidote_extractor: Tuple[Type[Extractor], dict],
                       killing_agreement_extractor: Tuple[Type[Extractor], dict]):
        self.player_extractor = player_extractor[0].init_instance(**player_extractor[1])
        self.antidote_extractor = antidote_extractor[0].init_instance(**antidote_extractor[1])
        self.killing_agreement_extractor = killing_agreement_extractor[0].init_instance(
            **killing_agreement_extractor[1])

    def add_players(self, players: List[Tuple[Type[Agent], dict]]):
        """

        :param players:
        :return:
        """
        # match number of players
        assert self.player_nums == len(players), \
            f"Required {self.player_nums} players, got {len(self.player_list)} players. Please add more players."
        need_random_role = False
        need_random_name = False
        for idx, player_params in enumerate(players):
            name = player_params[1].get("name")
            role = player_params[1].get("role")
            if name is None:
                need_random_name = True
            if role is None:
                need_random_role = True
        self.alive_roles = copy.deepcopy(self.roles)
        random.shuffle(self.roles)
        idx = 0
        for agent_type, player_params in players:
            if need_random_name:
                name = f'player {idx + 1}'
                player_params['name'] = name
            if need_random_role:
                role = self.roles[idx]
                player_params['role'] = role
            player_i = agent_type.init_instance(**player_params)
            self.player_list.append(player_i)
            self.players[player_i.name] = player_i
            self.player_mapping[player_i.name] = player_i.role
            idx += 1
        self.roles = [player_i.role for player_i in self.player_list]
        idx = self.roles.index(self.role_mapping['seer'])
        self.seer_player = self.player_list[idx].name
        idx = self.roles.index(self.role_mapping['guard'])
        self.guard_player = self.player_list[idx].name
        idx = self.roles.index(self.role_mapping['witch'])
        self.witch_player = self.player_list[idx].name
        for idx, role in enumerate(self.roles):
            if role == self.role_mapping["villager"]:
                self.villager_players.append(self.player_list[idx].name)
            if role == self.role_mapping["werewolf"]:
                self.werewolf_players.append(self.player_list[idx].name)

    def init_game(self) -> None:
        self.alive_players = [player_i.name for player_i in self.player_list]
        # reset
        self.antidote_skill = True
        self.poison_skill = True

    def start(self):
        """
        start werewolf game
        :return:
        """
        self.init_game()
        game_continue = True
        day_count = 1
        process_json = {}
        try:
            while game_continue:
                self.day_count = day_count
                instruction = f'day {day_count} starts:'
                print_text_animated(Fore.WHITE + f"System:\n\n{instruction}\n\n")
                game_continue = self.run_one_day(day_count)
                process_json[instruction] = self.process_list
                write_json(process_json, f'{self.output_dir}/process.json')
                day_count += 1
                self.process_list = []
        except Exception as e:
            instruction = f'day {day_count} starts:'
            process_json[instruction] = self.process_list
            write_json(process_json, f'{self.output_dir}/process.json')
            raise e

    def run_one_day(self, day_count):
        game_continue = True

        # night phase
        # killed_players = ['player 2']
        killed_players = self.night_process(day_count)
        for player_i in killed_players:
            if player_i in self.alive_players:
                self.alive_players.remove(player_i)
        if self.game_end():
            game_continue = False

        # daytime phase
        eliminated_player = self.daytime_process(day_count, killed_players)
        if eliminated_player != "pass" and eliminated_player in self.alive_players:
            self.alive_players.remove(eliminated_player)
        if self.game_end():
            game_continue = False

        # end game
        if not game_continue:
            winner = self.check_winner()
            game_end_step = self.host_instruction.get('game_end_step', '')
            instruction = game_end_step.format(winner=winner)
            for player_i, _ in self.players.items():
                self.players[player_i].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)

            self.process_list.append({'Host': instruction})
            print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
        return game_continue

    def night_process(self, day_count):
        # night start
        night_start_prompt = self.host_instruction.get('night_start', '')
        instruction = night_start_prompt
        for player_i in self.alive_players:
            self.players[player_i].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)

        print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
        self.process_list.append({'Host': instruction})

        # guard step
        guard_protect_player = self.guard_step(day_count)

        # werewolf step
        werewolf_kill_player = self.werewolf_step(day_count)

        # guard_protect_player = 'pass'
        # werewolf_kill_player = 'player 2'

        # witch step
        killed_player = None if guard_protect_player == werewolf_kill_player else werewolf_kill_player
        rescue, poison_kill_player = self.witch_step(day_count, killed_player)

        # seer step
        self.seer_step(day_count)

        killed_players = []
        if killed_player and killed_player != 'pass' and not rescue:
            killed_players.append(werewolf_kill_player)
        if poison_kill_player != 'pass' and poison_kill_player not in killed_players:
            killed_players.append(poison_kill_player)

        for player_i in killed_players:
            if player_i == 'pass':
                continue
            killed_step_prompt = self.host_instruction.get('killed_step', '')
            instruction = killed_step_prompt
            self.players[player_i].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)

        return killed_players

    def daytime_process(self, day_count, killed_players):
        #  daytime_start
        daytime_start_prompt = self.host_instruction.get('daytime_start', '')
        instruction = daytime_start_prompt
        for player_i in self.alive_players:
            self.players[player_i].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)

        print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
        self.process_list.append({'Host': instruction})

        if killed_players:
            killing_night_prompt = self.host_instruction.get('killing_night', '')
            instruction = killing_night_prompt.format(player=', '.join(killed_players))
        else:
            peaceful_night_prompt = self.host_instruction.get('peaceful_night', '')
            instruction = peaceful_night_prompt
        for player_i in self.alive_players:
            self.players[player_i].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)

        print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
        self.process_list.append({'Host': instruction})

        # discussion
        self.discussion(day_count)

        # voting
        eliminated_player = self.voting(day_count)

        # last words
        if eliminated_player != "pass":
            self.last_words(day_count, eliminated_player)

        return eliminated_player

    def guard_step(self, day_count) -> str:
        alive_players = copy.deepcopy(self.alive_players)
        # guard step
        guard_protect_player = 'pass'
        player_i = self.guard_player
        if self.guard_player and self.guard_player in self.alive_players:
            guard_step_prompt = self.host_instruction.get('guard_step', '')
            res_rule = copy.deepcopy(self.response_rule.get('guard_step', {}))
            instruction = guard_step_prompt.format(player=self.guard_player, options=alive_players + ["pass"])
            output = self.players[self.guard_player].step(message=f"Day {day_count}, Night Phase |" + instruction)

            if self.player_extractor is not None:
                s = self.player_extractor.step(f"Question：{instruction}\nAnswer：{output}")
                all_match = re.findall('\d+', s)
                s = all_match[-1] if all_match else "pass"
            else:
                all_match = re.findall('\d+', output)
                s = all_match[-1] if all_match else "pass"
            guard_protect_player = f"player {s}" if s != "pass" else s

            self.process_list.append(
                {'Host': instruction,
                 f"{player_i}({self.player_mapping[player_i]})": output,
                 "response_rule": res_rule})
            print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
            print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")
        guard_step_prompt = self.host_instruction.get('guard_step_public', '')
        instruction = guard_step_prompt
        for player_i in self.alive_players:
            if player_i != self.guard_player:
                self.players[player_i].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)

        return guard_protect_player

    def werewolf_step(self, day_count) -> str:
        alive_players = copy.deepcopy(self.alive_players)
        werewolf_open_eyes_prompt = self.host_instruction.get('werewolf_open_eyes', '')
        instruction = werewolf_open_eyes_prompt.format(werewolf_players=self.werewolf_players,
                                                       werewolf_count=len(self.werewolf_players))
        werewolf_order = []
        for player_i in self.alive_players:
            if player_i in self.werewolf_players:
                werewolf_order.append(player_i)
                _ = self.players[player_i].step(message=f"Day {day_count}, Night Phase |" + instruction)

        # kill step
        kill_player = "pass"
        kill_done = False
        retry = 0
        max_retry = 1
        while not kill_done and retry < max_retry:
            # random werewolf order
            random.shuffle(werewolf_order)

            player_i = werewolf_order[0]
            first_werewolf_kill_prompt = self.host_instruction.get('first_werewolf_kill', '')
            res_rule = copy.deepcopy(self.response_rule.get('first_werewolf_kill', {}))
            instruction = first_werewolf_kill_prompt.format(player=player_i, options=alive_players + ['pass'])
            output = self.players[player_i].step(message=f"Day {day_count}, Night Phase |" + instruction)

            self.process_list.append(
                {'Host': instruction,
                 f"{player_i}({self.player_mapping[player_i]})": output,
                 "response_rule": res_rule})
            print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
            print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

            for player_i in werewolf_order[1:]:
                self.players[player_i].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)
                self.players[player_i].receive(name=werewolf_order[0],
                                               message=f"Day {day_count}, Night Phase |" + output)

            if self.player_extractor is not None:
                s = self.player_extractor.step(f"Question：{instruction}\nAnswer：{output}")
                all_match = re.findall('\d+', s)
                s = all_match[-1] if all_match else "pass"
            else:
                all_match = re.findall('\d+', output)
                s = all_match[-1] if all_match else "pass"
            kill_player = f"player {s}" if s != "pass" else s
            agreements = 1

            for player_j in werewolf_order[1:]:
                other_werewolf_kill_prompt = self.host_instruction.get('other_werewolf_kill', '')
                res_rule = copy.deepcopy(self.response_rule.get('other_werewolf_kill', {}))
                instruction = other_werewolf_kill_prompt.format(player=player_j, first_player=player_i,
                                                                target_player=kill_player)
                output = self.players[player_j].step(message=f"Day {day_count}, Night Phase |" + instruction)
                if self.killing_agreement_extractor is not None:
                    s = self.killing_agreement_extractor.step(f"Question：{instruction}\nAnswer：{output}")
                    all_match = re.findall('(true|false)', s.lower())
                    s = all_match[-1] if all_match else "pass"
                else:
                    all_match = re.findall('(true|false)', output.lower())
                    s = all_match[-1] if all_match else "pass"
                if s != 'false':
                    agreements += 1

                self.process_list.append(
                    {'Host': instruction,
                     f"{player_j}({self.player_mapping[player_j]})": output,
                     "response_rule": res_rule})
                print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
                print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")
            if agreements >= len(werewolf_order) / 2:
                kill_done = True
            else:
                retry += 1
        # random eliminate
        if not kill_done:
            kill_player = random.choice(self.alive_players)
        if kill_player == "pass":
            kill_player = random.choice(self.alive_players)
        return kill_player

    def witch_step(self, day_count, killed_player: str) -> Tuple[bool, str]:
        alive_players = copy.deepcopy(self.alive_players)
        rescue = False
        poison_kill_player = "pass"
        if self.witch_player and self.witch_player in self.alive_players:
            player_i = self.witch_player
            if self.antidote_skill and killed_player is not None:
                witch_antidote_step_prompt = self.host_instruction.get('witch_antidote_step', '')
                res_rule = copy.deepcopy(self.response_rule.get('witch_antidote_step', {}))
                instruction = witch_antidote_step_prompt.format(player=self.witch_player, target_player=killed_player)
                output = self.players[player_i].step(message=f"Day {day_count}, Night Phase |" + instruction)

                if self.antidote_extractor is not None:
                    s = self.antidote_extractor.step(f"Question：{instruction}\nAnswer：{output}")
                    all_match = re.findall('(true|false)', s.lower())
                    s = all_match[-1] if all_match else "pass"
                else:
                    all_match = re.findall('(true|false)', output.lower())
                    s = all_match[-1] if all_match else "pass"
                rescue = True if 'true' == s else False

                self.process_list.append(
                    {'Host': instruction,
                     f"{player_i}({self.player_mapping[player_i]})": output,
                     "response_rule": res_rule})
                print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
                print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")
            if self.poison_skill:
                witch_poison_step_prompt = self.host_instruction.get('witch_poison_step', '')
                res_rule = copy.deepcopy(self.response_rule.get('witch_poison_step', {}))
                instruction = witch_poison_step_prompt.format(player=self.witch_player,
                                                              options=alive_players + ['pass'])
                output = self.players[player_i].step(message=f"Day {day_count}, Night Phase |" + instruction)

                if self.player_extractor is not None:
                    s = self.player_extractor.step(f"Question：{instruction}\nAnswer：{output}")
                    all_match = re.findall('\d+', s)
                    s = all_match[-1] if all_match else "pass"
                else:
                    all_match = re.findall('\d+', output)
                    s = all_match[-1] if all_match else "pass"
                poison_kill_player = f"player {s}" if s != "pass" else s

                self.process_list.append(
                    {'Host': instruction,
                     f"{player_i}({self.player_mapping[player_i]})": output,
                     "response_rule": res_rule})
                print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
                print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

        witch_step_prompt = self.host_instruction.get('witch_step_public', '')
        instruction = witch_step_prompt
        for player_i in self.alive_players:
            if player_i != self.witch_player:
                self.players[player_i].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)

        # only one chance for one gamee
        if rescue:
            self.antidote_skill = False
        if poison_kill_player != 'pass':
            self.poison_skill = False
        return rescue, poison_kill_player

    def seer_step(self, day_count):
        alive_players = copy.deepcopy(self.alive_players)
        # seer step seer_step
        if self.seer_player and self.seer_player in self.alive_players:
            player_i = self.seer_player
            seer_step_prompt = self.host_instruction.get('seer_step', '')
            res_rule = copy.deepcopy(self.response_rule.get('seer_step', {}))
            instruction = seer_step_prompt.format(player=self.seer_player, options=alive_players + ['pass'])
            output = self.players[self.seer_player].step(message=f"Day {day_count}, Night Phase |" + instruction)

            if self.player_extractor is not None:
                s = self.player_extractor.step(f"Question：{instruction}\nAnswer：{output}")
                all_match = re.findall('\d+', s)
                s = all_match[-1] if all_match else "pass"
            else:
                all_match = re.findall('\d+', output)
                s = all_match[-1] if all_match else "pass"
            seen_idx = s

            self.process_list.append(
                {'Host': instruction,
                 f"{player_i}({self.player_mapping[player_i]})": output,
                 "response_rule": res_rule})
            print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
            print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

            is_werewolf = True if f"player {seen_idx}" in self.werewolf_players else False
            if is_werewolf:
                seen_prompt = self.host_instruction.get('seen_werewolf', '')
            else:
                seen_prompt = self.host_instruction.get('seen_other', '')
            instruction = seen_prompt.format(player=f"player {seen_idx}")
            _ = self.players[self.seer_player].step(message=f"Day {day_count}, Night Phase |" + instruction)
            self.process_list.append(
                {'Host': instruction})
            print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")

        seer_step_prompt = self.host_instruction.get('seer_step_public', '')
        instruction = seer_step_prompt
        for player_i in self.alive_players:
            if player_i != self.seer_player:
                self.players[player_i].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)
        return

    def discussion(self, day_count):
        # random discussion order
        discussion_order = copy.deepcopy(self.alive_players)
        random.shuffle(discussion_order)

        player_i = discussion_order[0]
        first_discussion_step_prompt = self.host_instruction.get('first_discussion_step', '')
        instruction = first_discussion_step_prompt.format(player=player_i)
        output = self.players[player_i].step(message=f"Day {day_count}, Night Phase |" + instruction)

        self.process_list.append(
            {'Host': instruction,
             f"{player_i}({self.player_mapping[player_i]})": output})
        print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
        print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

        for player_j in self.alive_players:
            if player_i != player_j:
                self.players[player_j].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)
                self.players[player_j].receive(name=player_i, message=f"Day {day_count}, Night Phase |" + output)

        for player_i in discussion_order[1:]:
            other_discussion_step_prompt = self.host_instruction.get('other_discussion_step', '')
            instruction = other_discussion_step_prompt.format(player=player_i)
            output = self.players[player_i].step(message=f"Day {day_count}, Night Phase |" + instruction)

            self.process_list.append(
                {'Host': instruction,
                 f"{player_i}({self.player_mapping[player_i]})": output})
            print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
            print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

            for player_j in self.alive_players:
                if player_i != player_j:
                    self.players[player_j].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)
                    self.players[player_j].receive(name=player_i, message=f"Day {day_count}, Night Phase |" + output)
        return

    def voting(self, day_count):
        voting = []
        candidate_players = copy.deepcopy(self.alive_players)
        voting_done = False
        max_retry = 1
        retry = 0
        eliminate_player = "pass"
        while not voting_done and retry < max_retry:
            # random voting order
            voting_order = copy.deepcopy(self.alive_players)
            random.shuffle(voting_order)

            player_i = voting_order[0]
            first_voting_step_prompt = self.host_instruction.get('first_voting_step', '')
            res_rule = copy.deepcopy(self.response_rule.get('first_voting_step', {}))
            instruction = first_voting_step_prompt.format(player=player_i, options=candidate_players)
            output = self.players[player_i].step(message=f"Day {day_count}, Night Phase |" + instruction)

            self.process_list.append(
                {'Host': instruction,
                 f"{player_i}({self.player_mapping[player_i]})": output,
                 "response_rule": res_rule})
            print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
            print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

            if self.player_extractor is not None:
                s = self.player_extractor.step(f"Question：{instruction}\nAnswer：{output}")
                all_match = re.findall('\d+', s)
                s = all_match[-1] if all_match else "pass"
            else:
                all_match = re.findall('\d+', output)
                s = all_match[-1] if all_match else "pass"
            voting.append(s)

            for player_j in self.alive_players:
                if player_i != player_j:
                    self.players[player_j].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)
                    self.players[player_j].receive(name=player_i, message=f"Day {day_count}, Night Phase |" + output)

            for player_i in voting_order[1:]:
                other_voting_step_prompt = self.host_instruction.get('other_voting_step', '')
                res_rule = copy.deepcopy(self.response_rule.get('other_voting_step', {}))
                instruction = other_voting_step_prompt.format(player=player_i, options=candidate_players)
                output = self.players[player_i].step(message=f"Day {day_count}, Night Phase |" + instruction)

                if self.player_extractor is not None:
                    s = self.player_extractor.step(f"Question：{instruction}\nAnswer：{output}")
                    all_match = re.findall('\d+', s)
                    s = all_match[-1] if all_match else "pass"
                else:
                    all_match = re.findall('\d+', output)
                    s = all_match[-1] if all_match else "pass"
                voting.append(s)

                self.process_list.append(
                    {'Host': instruction,
                     f"{player_i}({self.player_mapping[player_i]})": output,
                     "response_rule": res_rule})
                print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
                print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

                for player_j in self.alive_players:
                    if player_i != player_j:
                        self.players[player_j].receive(name="host",
                                                       message=f"Day {day_count}, Night Phase |" + instruction)
                        self.players[player_j].receive(name=player_i,
                                                       message=f"Day {day_count}, Night Phase |" + output)
            voting = [x if x in self.alive_players else 'pass' for x in voting]
            while 'pass' in voting:
                voting.remove("pass")
            if voting:
                max_equal_voting = self.return_max_equal_voting(voting)
                if len(max_equal_voting) == 1:
                    eliminate_player = f"player {max_equal_voting[0]}"
                    voting_done = True
                else:
                    candidate_players = [f"player {idx}" for idx in max_equal_voting]
                    retry += 1
            else:
                candidate_players = copy.deepcopy(self.alive_players)
                retry += 1

        # random eliminate
        if eliminate_player == "pass":
            eliminate_player = random.choice(candidate_players)

        # eliminated_step
        eliminated_step_prompt = self.host_instruction.get('eliminated_step', '')
        instruction = eliminated_step_prompt.format(player=eliminate_player)
        for player_i in self.alive_players:
            self.players[player_i].receive(name="host", message=f"Day {day_count}, Night Phase |" + instruction)
        print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
        return eliminate_player

    def last_words(self, day_count, eliminated_player):
        if eliminated_player in self.alive_players:
            player_i = eliminated_player
            first_voting_step_prompt = self.host_instruction.get('eliminated_step', '')
            instruction = first_voting_step_prompt.format(player=player_i)
            output = self.players[player_i].step(message=f"Day {day_count}, Night Phase |" + instruction)

            self.process_list.append(
                {'Host': instruction, f"{player_i}({self.player_mapping[player_i]})": output})
            print_text_animated(Fore.WHITE + f"Host:\n\n{instruction}\n\n")
            print_text_animated(COLOR[player_i] + f"{player_i}({self.player_mapping[player_i]}):\n\n{output}\n\n")

            for player_j in self.alive_players:
                if player_i != player_j:
                    self.players[player_j].receive(name="host",
                                                   message=f"Day {day_count}, Night Phase |" + instruction)
                    self.players[player_j].receive(name=player_i,
                                                   message=f"Day {day_count}, Night Phase |" + output)
        return

    @staticmethod
    def return_max_equal_voting(voting):
        counter = Counter(voting)
        max_count = max(counter.values())
        most_common = [element for element, count in counter.items() if count == max_count]
        return most_common

    def game_end(self) -> bool:
        werewolf_all_out = True
        for player_i in self.werewolf_players:
            if player_i in self.alive_players:
                werewolf_all_out = False

        villager_all_out = True
        for player_i in self.villager_players:
            if player_i in self.alive_players:
                villager_all_out = False

        special_all_out = True
        for player_i in [self.witch_player, self.seer_player, self.guard_player]:
            if player_i in self.alive_players:
                special_all_out = False
        return werewolf_all_out or villager_all_out or special_all_out

    def check_winner(self):
        werewolf_all_out = True
        for player_i in self.werewolf_players:
            if player_i in self.alive_players:
                werewolf_all_out = False

        villager_all_out = True
        for player_i in self.villager_players:
            if player_i in self.alive_players:
                villager_all_out = False
        special_all_out = True
        for player_i in [self.witch_player, self.seer_player, self.guard_player]:
            if player_i in self.alive_players:
                special_all_out = False
        if werewolf_all_out and not villager_all_out and not special_all_out:
            winner = "Villager"
            self.winners = ['Villager', 'Seer', 'Guard', 'Witch']
        elif not werewolf_all_out and (villager_all_out or special_all_out):
            winner = "Werewolf"
            self.winners = ['Werewolf']
        else:
            winner = "None"
            self.winners = []
        return winner
