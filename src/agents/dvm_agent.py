#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: dvm_agent.py 
# @date: 2024/9/12 09:30 
#

import os
import re
import json

from .abs_agent import Agent
from ..models.predictor import Predictor
from ..models.decider import Decider, RandomDecider
from ..models.discussor import Discussor
from ..training.decision_chain import DecisionChainReward
from ..utils import write_json


class DVMAgent(Agent):
    def __init__(self, name, role, model, api_key="EMPTY", base_url="http://localhost:8000/v1",
                 output_dir="./logs", win_rate_constraint=0.5, player_nums=7, device="cuda", **kwargs):
        super().__init__(name=name, role=role, **kwargs)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.win_rate_constraint = win_rate_constraint
        self.player_nums = player_nums

        self.predictor = Predictor(model, api_key, base_url)
        self.discussor = Discussor(model, api_key, base_url)

        state_dim = 32
        pred_dim = (player_nums - 1) * 5
        action_dim = player_nums + 2
        self.decider = Decider(state_dim, pred_dim, action_dim).to(device)
        self.random_decider = RandomDecider()
        self.use_random = kwargs.get("use_random", True)
        self.device = device

        self.memory = {"name": [], "message": [], "phase": []}
        self.discussions = []
        self.votings = []
        self.decision_chain = []
        self.dc_reward = DecisionChainReward()

    def step(self, message: str) -> str:
        phase, instruction = message.split("|", 1)
        phase_num = re.findall(r"\d+", phase)
        phase_num = phase_num[-1] if phase_num else "0"

        discussion_text = "\n".join(self.discussions[-20:])
        voting_text = "\n".join(self.votings[-20:])

        pred_vec, predictions = self.predictor.predict_to_vector(
            self.name, self.role, discussion_text, voting_text,
            [f"player {i + 1}" for i in range(self.player_nums)]
        )

        state_vec = self.build_state_vec(phase, instruction)
        candidate_actions = self.build_candidate_actions(instruction)
        action_mask = self.build_action_mask(candidate_actions)

        if self.use_random:
            action_str, action_idx = self.random_decider.decide(
                state_vec, pred_vec, self.win_rate_constraint, action_mask, candidate_actions
            )
        else:
            action_str, action_idx = self.decider.decide(
                state_vec, pred_vec, self.win_rate_constraint, action_mask, candidate_actions
            )

        self.decision_chain.append(action_idx)

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
        elif "discuss" in lower or "talk" in lower or "statement" in lower:
            self.discussions.append(f"{name}: {content}")
        else:
            self.discussions.append(f"{name}: {content}")

    def is_discussion(self, instruction):
        lower = instruction.lower()
        return "talk" in lower or "discuss" in lower or "statement" in lower

    def format_action_response(self, instruction, action_str):
        if "agree" in instruction.lower() or "disagree" in instruction.lower():
            return "agree" if "player" in action_str.lower() else action_str
        if "yes" in action_str.lower() or "no" in action_str.lower():
            return action_str
        nums = re.findall(r"\d+", action_str)
        if nums:
            return f"player {nums[-1]}"
        return action_str

    def build_state_vec(self, phase, instruction):
        vec = [0.0] * 32
        alive = self.player_nums
        for msg in self.memory.get("message", []):
            if "killed" in msg.lower() or "eliminated" in msg.lower():
                alive -= 1
        vec[0] = float(alive) / self.player_nums
        vec[1] = 1.0 if "Werewolf" in self.role else 0.0
        vec[2] = 1.0 if "Seer" in self.role else 0.0
        vec[3] = 1.0 if "Witch" in self.role else 0.0
        vec[4] = 1.0 if "Guard" in self.role else 0.0
        vec[5] = float(self.win_rate_constraint)
        return vec

    def build_candidate_actions(self, instruction):
        actions = [f"player {i + 1}" for i in range(self.player_nums)]
        actions += ["yes", "no"]
        return actions

    def build_action_mask(self, candidate_actions):
        alive = set()
        for msg in self.memory.get("message", []):
            for i in range(1, self.player_nums + 1):
                p = f"player {i}"
                if p in msg and ("alive" in msg.lower() or "pass" not in msg.lower()):
                    alive.add(p)
        if not alive:
            alive = set([f"player {i + 1}" for i in range(self.player_nums)])
        mask = []
        for a in candidate_actions:
            if a.startswith("player") and a not in alive:
                mask.append(1.0)
            else:
                mask.append(0.0)
        return mask

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
