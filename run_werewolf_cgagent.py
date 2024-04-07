#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: run_werewolf_cgagent.py 
# @date: 2024/3/19 16:21 
#
# describe:
#
import random
import argparse

import openai
from sentence_transformers import SentenceTransformer

from src.extractor.llm_extractor.chatgpt_extractor import ChatGPTBasedExtractor
from src.games.werewolf.werewolf import Werewolf
from src.agents import CGAgent
from prompt.werewolf_cg_prompt import rule_role_prompt, select_question_prompt, ask_question_prompt, \
    generate_answer_prompt, reflection_prompt, extract_suggestion_prompt, generate_response_prompt, \
    informativeness_prompt, question_list
from src.games.werewolf.extract_demos import number_extract_prompt, player_extractor_demos, vote_extractor_demos, \
    bool_extract_prompt
from src.utils import create_dir, read_json

api_key = "sk-UAHO5kixcAwbXOoPBd1393C5B51f455bB2D5882f27998d53"
base_url = None
roles = ["Werewolf", "Werewolf", "Villager", "Villager", "Seer", "Guard", "Witch"]

bert_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1", device="cpu")


def run_game(game_output_dir: str, camp, game_idx):
    create_dir(game_output_dir.format(game_idx))

    mode = 'watch'
    language = 'english'
    ai_model = 'gpt-3.5-turbo-16k'
    player_nums = 7
    player_mapping = {}
    random.shuffle(roles)
    # if camp == "villager":
    #     camp_role = ["Villager", "Seer", "Guard", "Witch"]
    # else:
    #     camp_role = ["Werewolf"]
    camp_role = ["Werewolf", "Villager", "Seer", "Guard", "Witch"]

    game = Werewolf(player_nums, language, mode, ai_model, game_output_dir.format(game_idx))

    player_args = []
    for i in range(player_nums):
        log_dir = f"{game_output_dir.format(game_idx)}/player {i + 1}"
        create_dir(log_dir)
        if roles[i] in camp_role:
            if game_idx == 0:
                previous_exp_pool = []
            else:
                load_file = f"{game_output_dir.format(game_idx - 1)}/{roles[i]}_reflection.json"
                previous_exp_pool = read_json(load_file)
            name = f"player {i + 1}"
            role = roles[i]
            player_args.append(
                (
                    CGAgent, {"name": name, "role": role, "rule_role_prompt": rule_role_prompt,
                              "select_question_prompt": select_question_prompt,
                              "ask_question_prompt": ask_question_prompt,
                              "generate_answer_prompt": generate_answer_prompt,
                              "reflection_prompt": reflection_prompt,
                              "extract_suggestion_prompt": extract_suggestion_prompt,
                              "generate_response_prompt": generate_response_prompt,
                              "informativeness_prompt": informativeness_prompt,
                              "question_list": question_list.get(roles[i], []), "retrival_model": bert_model,
                              "model": ai_model, "freshness_k": 15, "informativeness_n": 15, "experience_window": 50,
                              "temperature": 0.3, "api_key": "", "previous_exp_pool": previous_exp_pool,
                              "output_dir": log_dir}
                )
            )

    game.add_players(player_args)

    # extractors
    extractor_args = [(ChatGPTBasedExtractor,
                       {"extractor_name": "player extractor", "model_name": 'gpt-3.5-turbo-16k',
                        "extract_prompt": number_extract_prompt, "system_prompt": "You are a helpful assistant.",
                        "temperature": 0, "few_shot_demos": player_extractor_demos,
                        "output_dir": game_output_dir.format(game_idx)}),
                      (ChatGPTBasedExtractor,
                       {"extractor_name": "antidote extractor", "model_name": 'gpt-3.5-turbo-16k',
                        "extract_prompt": bool_extract_prompt, "system_prompt": "You are a helpful assistant.",
                        "temperature": 0, "few_shot_demos": vote_extractor_demos,
                        "output_dir": game_output_dir.format(game_idx)}),
                      (ChatGPTBasedExtractor,
                       {"extractor_name": "killing agreement extractor", "model_name": 'gpt-3.5-turbo-16k',
                        "extract_prompt": bool_extract_prompt, "system_prompt": "You are a helpful assistant.",
                        "temperature": 0, "few_shot_demos": vote_extractor_demos,
                        "output_dir": game_output_dir.format(game_idx)})
                      ]

    game.init_extractor(player_extractor=extractor_args[0], antidote_extractor=extractor_args[1],
                        killing_agreement_extractor=extractor_args[2])
    game.start()
    for player_i, agent_i in game.players.items():
        if isinstance(agent_i, CGAgent):
            agent_i.reflection(
                player_mapping,
                file_name=f"{game_output_dir.format(game_idx)}/{player_mapping.get(player_i)}_reflection.json",
                winners=game.winners,
                duration=game.day_count
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game_count", type=int, default=10
    )
    parser.add_argument("--camp", type=str, default="good", choices=["villager", "villager"])
    parser.add_argument("--exp_name", type=str, default="cgagent_vs_cgagent")
    parser.add_argument("--use_proxy", type=bool, default=False)
    parser.add_argument("--start_game_idx", type=int, default=0)
    parsed_args = parser.parse_args()
    return parsed_args


def main():
    args = parse_args()
    # if args.use_proxy:
    #     openai.proxy = "http://127.0.0.1:7890"
    #     os.environ["ALL_PROXY"] = "http://127.0.0.1:7890"

    openai.api_key = api_key
    openai.base_url = base_url
    for game_round in range(args.start_game_idx, args.game_count):
        output_dir = f"playing_log/werewolf/battle/{args.exp_name}-{args.camp}" + "-game_{}"
        run_game(output_dir, camp=args.camp, game_idx=game_round)
        print("game finish!!! game index {}".format(game_round))


if __name__ == '__main__':
    main()
    print("done!!!")
    # remote 17522(4)
