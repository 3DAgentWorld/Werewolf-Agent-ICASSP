#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: predictor.py 
# @date: 2024/9/10 17:20 
#

import re

from ..apis.vllm_api import vllm_chat


class Predictor:
    def __init__(self, model, api_key="EMPTY", base_url="http://localhost:8000/v1", temperature=0.3):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature

    def predict(self, name, role, discussion, voting):
        sys_prompt = "You are an expert Werewolf player. Predict the roles of other players based on discussion and voting history."
        user_prompt = f"You are {name}, the {role}. Based on the following information, predict the roles of other players.\n\nDiscussion history:\n{discussion}\n\nVoting history:\n{voting}\n\nOutput format:\nPlayer X: Werewolf/Villager/Seer/Guard/Witch (reason)\n\nIf uncertain, output your best guess."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        output = vllm_chat(self.model, messages, self.temperature, self.api_key, self.base_url)
        return self.parse_predictions(output)

    def parse_predictions(self, text):
        predictions = {}
        pattern = re.compile(r"player\s*(\d+)\s*[:：]\s*(\w+)", re.IGNORECASE)
        for match in pattern.finditer(text):
            player = f"player {match.group(1)}"
            pred_role = match.group(2).capitalize()
            predictions[player] = pred_role
        return predictions

    def predict_to_vector(self, name, role, discussion, voting, player_list):
        preds = self.predict(name, role, discussion, voting)
        vec = []
        role_list = ["Werewolf", "Villager", "Seer", "Guard", "Witch"]
        for p in player_list:
            if p == name:
                continue
            pr = preds.get(p, "Unknown")
            one_hot = [1.0 if pr == r else 0.0 for r in role_list]
            vec.extend(one_hot)
        return vec, preds
