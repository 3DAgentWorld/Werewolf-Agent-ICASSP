#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: dvm_agent.py 
# @date: 2024/9/12 09:30 
#

import os
import re
import json

import torch

from .abs_agent import Agent
from ..models.predictor import Predictor
from ..models.decider import Decider, RandomDecider
from ..models.discussor import Discussor
from ..training.decision_chain import DecisionChainReward
from ..utils import write_json


class DVMAgent(Agent):
    PHASE_LIST = [
        "other", "werewolf_kill", "werewolf_agree", "witch_antidote",
        "witch_poison", "seer_verify", "hunter_shoot", "vote", "discussion", "eliminated"
    ]

    def __init__(self, name, role, model, api_key="EMPTY", base_url="http://localhost:8000/v1",
                 output_dir="./logs", win_rate_constraint=0.5, player_nums=7, device="cuda",
                 decider_path=None, dump_steps=False, **kwargs):
        super().__init__(name=name, role=role, **kwargs)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.win_rate_constraint = win_rate_constraint
        self.player_nums = player_nums
        self.action_dim = player_nums + 2

        self.predictor = Predictor(model, api_key, base_url)
        self.discussor = Discussor(model, api_key, base_url)

        state_dim = 32
        pred_dim = (player_nums - 1) * 6
        self.decider = Decider(state_dim, pred_dim, self.action_dim).to(device)
        self.random_decider = RandomDecider()
        self.use_random = kwargs.get("use_random", True)
        self.device = device
        self.decider_path = decider_path
        self.dump_steps = dump_steps
        self.step_data_file = os.path.join(output_dir, "step_data.jsonl") if dump_steps else None

        if decider_path is not None:
            if not os.path.exists(decider_path):
                raise FileNotFoundError(
                    f"Decider checkpoint not found: {decider_path}. "
                    f"Generate training data and train a decider, or provide a valid checkpoint path."
                )
            self.decider.load_state_dict(torch.load(decider_path, map_location=device))
            self.use_random = False
            print(f"[{self.name}] Loaded decider from {decider_path}")

        self.memory = {"name": [], "message": [], "phase": []}
        self.discussions = []
        self.votings = []
        self.decision_chain = []
        self.dc_reward = DecisionChainReward()
        self.dead_players = set()

    def step(self, message: str) -> str:
        phase, instruction = message.split("|", 1)
        phase_num = re.findall(r"\d+", phase)
        phase_num = phase_num[-1] if phase_num else "0"

        self.update_dead_players(instruction)
        discussion_text = "\n".join(self.discussions[-20:])
        voting_text = "\n".join(self.votings[-20:])

        pred_vec, predictions = self.predictor.predict_to_vector(
            self.name, self.role, discussion_text, voting_text,
            [f"player {i + 1}" for i in range(self.player_nums)]
        )

        state_vec = self.build_state_vec(phase, instruction)
        candidate_actions, action_mask = self.build_actions(instruction)

        if self.use_random:
            action_str, action_idx = self.random_decider.decide(
                state_vec, pred_vec, self.win_rate_constraint, action_mask, candidate_actions
            )
        else:
            action_str, action_idx = self.decider.decide(
                state_vec, pred_vec, self.win_rate_constraint, action_mask, candidate_actions
            )

        self.decision_chain.append(action_idx)

        if self.dump_steps:
            step_record = {
                "player": self.name,
                "role": self.role,
                "phase": phase,
                "instruction": instruction,
                "state_vec": state_vec,
                "pred_vec": pred_vec,
                "wr_cons": self.win_rate_constraint,
                "candidate_actions": candidate_actions,
                "action_mask": action_mask,
                "action_idx": action_idx,
                "action_str": action_str,
                "predictions": predictions,
                "use_random": self.use_random,
            }
            with open(self.step_data_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(step_record, ensure_ascii=False) + "\n")

        if self.is_discussion(instruction):
            game_state = f"phase: {phase}\ninstruction: {instruction}\npredictions: {json.dumps(predictions, ensure_ascii=False)}"
            response = self.discussor.discuss(
                self.name, self.role, game_state, discussion_text, voting_text, action_str
            )
        else:
            response = self.format_action_response(instruction, action_str)

        self.update_memory("Host", instruction, phase_num)
        self.update_memory("Self", response, phase_num)
        self.log(f"{self.output_dir}/response.txt", f"phase:{phase}\ninstruction:{instruction}\naction:{action_str}\noutput:{response}\n---\n")
        return response

    def receive(self, name: str, message: str) -> None:
        phase, content = message.split("|", 1)
        phase_num = re.findall(r"\d+", phase)
        phase_num = phase_num[-1] if phase_num else "0"
        self.update_memory(name, content, phase_num)
        lower = content.lower()
        if "vote" in lower:
            self.votings.append(f"{name}: {content}")
        else:
            self.discussions.append(f"{name}: {content}")

    def update_dead_players(self, instruction):
        lower = instruction.lower()
        if "died" in lower or "killed" in lower or "eliminated" in lower:
            for i in range(1, self.player_nums + 1):
                p = f"player {i}"
                if p in instruction and p != self.name:
                    self.dead_players.add(p)

    def is_discussion(self, instruction):
        lower = instruction.lower()
        return "talk" in lower or "discuss" in lower or "statement" in lower

    def format_action_response(self, instruction, action_str):
        lower = instruction.lower()
        if "agree" in lower or "disagree" in lower:
            if "disagree" in action_str.lower():
                return "disagree"
            return "agree"
        if "antidote" in lower or ("save" in lower and "witch" in lower):
            if "no" in action_str.lower():
                return "no"
            return "yes"
        nums = re.findall(r"\d+", action_str)
        if nums:
            return f"player {nums[-1]}"
        return action_str

    def build_state_vec(self, phase, instruction):
        vec = [0.0] * 32
        alive = self.player_nums - len(self.dead_players)
        vec[0] = float(alive) / self.player_nums
        vec[1] = 1.0 if "Werewolf" in self.role else 0.0
        vec[2] = 1.0 if "Seer" in self.role else 0.0
        vec[3] = 1.0 if "Witch" in self.role else 0.0
        vec[4] = 1.0 if "Guard" in self.role else 0.0
        vec[5] = 1.0 if "Hunter" in self.role else 0.0
        vec[6] = float(self.win_rate_constraint)

        lower = instruction.lower()
        if "agree" in lower or "disagree" in lower:
            phase_name = "werewolf_agree"
        elif "antidote" in lower:
            phase_name = "witch_antidote"
        elif "poison" in lower:
            phase_name = "witch_poison"
        elif "verify" in lower or "seer" in lower:
            phase_name = "seer_verify"
        elif "shoot" in lower:
            phase_name = "hunter_shoot"
        elif "vote" in lower:
            phase_name = "vote"
        elif "kill" in lower:
            phase_name = "werewolf_kill"
        elif "talk" in lower or "discuss" in lower or "statement" in lower:
            phase_name = "discussion"
        elif "eliminated" in lower or "last" in lower:
            phase_name = "eliminated"
        else:
            phase_name = "other"
        if phase_name in self.PHASE_LIST:
            vec[7 + self.PHASE_LIST.index(phase_name)] = 1.0
        return vec

    def build_actions(self, instruction):
        lower = instruction.lower()
        if "agree" in lower or "disagree" in lower:
            base = ["agree", "disagree"]
        elif "antidote" in lower or "[yes, no]" in lower:
            base = ["yes", "no"]
        else:
            base = [f"player {i + 1}" for i in range(self.player_nums)]

        mask = []
        for a in base:
            if a.startswith("player") and (a in self.dead_players or a == self.name):
                mask.append(1.0)
            else:
                mask.append(0.0)

        # "pass" is a real option for poison and hunter shoot, but not for
        # werewolf kill / vote / seer verify where the game engine will fall
        # back to random if pass is returned.
        allow_pass = "poison" in lower or "shoot" in lower
        while len(base) < self.action_dim:
            base.append("pass")
            mask.append(0.0 if allow_pass else 1.0)

        return base, mask

    def update_memory(self, name, message, phase):
        self.memory["name"].append(name)
        self.memory["message"].append(message)
        self.memory["phase"].append(phase)

    def log(self, file, data):
        with open(file, mode="a+", encoding="utf-8") as f:
            f.write(data)

    def reflection(self, player_role_mapping, file_name, winners, duration):
        won = self.role in winners
        chain_tuple = tuple(self.decision_chain)
        self.dc_reward.register_chain(chain_tuple, 1.0 if won else 0.0)
        write_json({
            "won": won,
            "duration": duration,
            "chain": self.decision_chain
        }, file_name)
