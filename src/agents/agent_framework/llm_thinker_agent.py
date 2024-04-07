#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: llm_thinker_agent.py 
# @date: 2024/4/3 20:13 
#
# describe:
#
from typing import List, Any

from ..abs_agent import Agent


class LLMThinkerAgent(Agent):
    """
    Implement of the agent proposed by the paper "Enhance Reasoning for Large Language Models in the Game Werewolf".
    We name the agent as LLM-Thinker-Agent.
    """

    def __init__(self, name: str, role: str, summary_prompt: str, extraction_prompt: str, **kwargs):
        super().__init__(name, role, **kwargs)

        self.summary_prompt = summary_prompt
        self.extraction_prompt = extraction_prompt

        self.memory = []

    def step(self, message: str) -> str:
        phase = message.split("|")[0]
        # self.phase = phase
        message = message.split("|")[1]

        language_feature = self.listening_process()
        speech_instruction = self.thinking_process(language_feature)
        response = self.presenting_process(speech_instruction)
        return response

    def receive(self, name: str, message: str) -> None:
        summarized_message = self.summarize(message)
        format_message = self.extraction_prompt.format(message)
        self.memory.append(
            {'name': name, 'message': message, 'summary': summarized_message}
        )

    def listening_process(self, name, message):
        summarized_message = self.summarize(message)
        format_message = self.extraction_prompt.format(message)
        self.memory.append(
            {'name': name, 'message': message, 'summary': summarized_message}
        )
        language_feature = []
        return language_feature

    def thinking_process(self, language_feature):
        speech_instruction = []
        return speech_instruction

    def presenting_process(self, speech_instruction):
        response = ""
        return response

    def summarize(self, message):
        prompt = self.summary_prompt.format(message)
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        summarized_message = self.send_message(messages)
        return summarized_message

    def extract_information(self, message):
        prompt = self.extraction_prompt.format(message)
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        format_message = self.send_message(messages)
        return format_message

    def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                     temperature: float = None) -> str:
        raise NotImplementedError("Interaction with LLM is not implemented in agent framework class.")
