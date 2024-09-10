#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: discussor.py 
# @date: 2024/9/11 15:30 
#

from ..apis.vllm_api import vllm_chat


class Discussor:
    def __init__(self, model, api_key="EMPTY", base_url="http://localhost:8000/v1", temperature=0.7):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature

    def discuss(self, name, role, game_state, discussion, voting, action):
        sys_prompt = "You are playing a Werewolf game. Generate natural discussion text based on your role, the game state, and your planned action."
        user_prompt = f"You are {name}, the {role}. The current game state is:\n{game_state}\n\nPrevious discussion:\n{discussion}\n\nPrevious voting:\n{voting}\n\nYour planned action: {action}\n\nPlease provide a brief statement (within 100 words) for the discussion phase. Do not reveal your actual role unless it helps your strategy."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return vllm_chat(self.model, messages, self.temperature, self.api_key, self.base_url)
