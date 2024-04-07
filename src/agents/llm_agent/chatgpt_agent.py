#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: chatgpt_agent.py 
# @date: 2024/2/18 14:03 
#
# describe:
#

import re
from typing import List, Any
import openai

from ..abs_agent import Agent, MessageSender
from ...apis.chatgpt_api import chatgpt
from ..agent_framework import CGAgent, SAPARAgentForAvalon, SAPARAgentForWerewolf

try:
    OPENAI_MAX_TOKENS_ERROR = openai.error.InvalidRequestError
except AttributeError as e:
    OPENAI_MAX_TOKENS_ERROR = openai.BadRequestError


class ChatGPTMessageSender(MessageSender):
    def __init__(self, model, tokenizer, temperature, **kwargs):
        super().__init__(model, tokenizer, temperature, **kwargs)
        self.api_key = kwargs.get('api_key')
        self.base_url = kwargs.get('base_url')

    def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                     temperature: float = None) -> str:
        __model = model or self.model
        __temperature = temperature or self.temperature
        output = chatgpt(__model, messages, __temperature, self.api_key, self.base_url)
        return output


class ChatGPT_CGAgent(CGAgent, ChatGPTMessageSender):
    def __init__(self, name: str, role: str, rule_role_prompt: str, select_question_prompt: str,
                 ask_question_prompt: str, generate_answer_prompt: str, reflection_prompt: str,
                 extract_suggestion_prompt: str, generate_response_prompt: str, informativeness_prompt: str,
                 question_list: list, retrival_model, freshness_k: int, informativeness_n: int, experience_window: int,
                 previous_exp_pool: list, output_dir: str, model, tokenizer, temperature, **kwargs):
        CGAgent.__init__(self, name, role, rule_role_prompt, select_question_prompt, ask_question_prompt,
                         generate_answer_prompt, reflection_prompt, extract_suggestion_prompt, generate_response_prompt,
                         informativeness_prompt, question_list, retrival_model, freshness_k, informativeness_n,
                         experience_window, previous_exp_pool, output_dir, **kwargs)
        ChatGPTMessageSender.__init__(self, model, tokenizer, temperature, **kwargs)

    def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                     temperature: float = None) -> str:
        return ChatGPTMessageSender.send_message(self, messages, model, tokenizer, temperature)


class ChatGPT_SAPARAgent_ForAvalon(SAPARAgentForAvalon, ChatGPTMessageSender):
    def __init__(self, name, role, role_intro, game_goal, strategy, system_prompt: str, summary_prompt: str,
                 analysis_prompt: str, plan_prompt: str, action_prompt: str, response_prompt: str,
                 suggestion_prompt: str, strategy_prompt: str, update_prompt: str, suggestion: str, other_strategy: str,
                 candidate_actions: list, output_dir: str, model, tokenizer, temperature, **kwargs):
        SAPARAgentForAvalon.__init__(self, name, role, role_intro, game_goal, strategy, system_prompt, summary_prompt,
                                     analysis_prompt, plan_prompt, action_prompt, response_prompt, suggestion_prompt,
                                     strategy_prompt, update_prompt, suggestion, other_strategy, candidate_actions,
                                     output_dir, **kwargs)
        ChatGPTMessageSender.__init__(self, model, tokenizer, temperature)

    def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                     temperature: float = None) -> str:
        return ChatGPTMessageSender.send_message(self, messages, model, tokenizer, temperature)

    def memory_summary(self, phase):
        prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase))
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]
        output = ""
        discard = 0
        while not output:
            try:
                output = self.send_message(messages)
            except OPENAI_MAX_TOKENS_ERROR as e:
                print("catch error: ", e.user_message)
                discard += 10
                prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase, discard))
                messages = [
                    {"role": 'system', "content": self.system_prompt},
                    {"role": 'user', "content": prompt}
                ]
        match = re.search("(?<=Summary:).*", output, re.S)
        summary = match.group().strip() if match else output
        self.summary[phase] = summary
        if self.summary:
            format_summary = "\n".join(
                [f"Quest Phase Turn {key}:{value}" if key != "0" else f"Reveal Phase: {value}" for key, value
                 in self.summary.items()])
        else:
            format_summary = "None"
        self.log(f"{self.output_dir}/summary.txt",
                 f"phase:{phase}\ninput:{prompt}\noutput:{output}\n--------------------")
        return format_summary


class ChatGPT_SAPARAgent_ForWerewolf(SAPARAgentForWerewolf, ChatGPTMessageSender):
    def __init__(self, name, role, role_intro, game_goal, strategy, system_prompt: str, summary_prompt: str,
                 analysis_prompt: str, plan_prompt: str, action_prompt: str, response_prompt: str,
                 suggestion_prompt: str, strategy_prompt: str, update_prompt: str, suggestion: str, other_strategy: str,
                 candidate_actions: list, output_dir: str, model, tokenizer, temperature, **kwargs):
        SAPARAgentForWerewolf.__init__(self, name, role, role_intro, game_goal, strategy, system_prompt, summary_prompt,
                                       analysis_prompt, plan_prompt, action_prompt, response_prompt, suggestion_prompt,
                                       strategy_prompt, update_prompt, suggestion, other_strategy, candidate_actions,
                                       output_dir, **kwargs)
        ChatGPTMessageSender.__init__(self, model, tokenizer, temperature)

    def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                     temperature: float = None) -> str:
        return ChatGPTMessageSender.send_message(self, messages, model, tokenizer, temperature)

    def memory_summary(self, phase):
        prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase))
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]
        output = ""
        discard = 0
        while not output:
            try:
                output = self.send_message(messages)
            except OPENAI_MAX_TOKENS_ERROR as e:
                print("catch error: ", e.user_message)
                discard += 10
                prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase, discard))
                messages = [
                    {"role": 'system', "content": self.system_prompt},
                    {"role": 'user', "content": prompt}
                ]
        match = re.search("(?<=Summary:).*", output, re.S)
        summary = match.group().strip() if match else output
        self.summary[phase] = summary
        if self.summary:
            format_summary = "\n".join(
                [f"Day {key}:{value}" for key, value in self.summary.items()])
        else:
            format_summary = "None"
        self.log(f"{self.output_dir}/summary.txt",
                 f"phase:{phase}\ninput:{prompt}\noutput:{output}\n--------------------")
        return format_summary

# class ChatGPTBasedSAPARAgent(SAPARAgent, ChatGPTMessageSender):
#     def __init__(self, name, role, role_intro, game_goal, strategy, system_prompt: str, summary_prompt: str,
#                  analysis_prompt: str, plan_prompt: str, action_prompt: str, response_prompt: str, model, tokenizer,
#                  temperature, api_key, output_dir, suggestion_prompt: str, strategy_prompt: str, update_prompt: str,
#                  suggestion: str, other_strategy: str, candidate_actions: list, **kwargs):
#         super().__init__(name, role, role_intro, game_goal, strategy, system_prompt, summary_prompt, analysis_prompt,
#                          plan_prompt, action_prompt, response_prompt, model, tokenizer, temperature, api_key,
#                          output_dir, suggestion_prompt, strategy_prompt, update_prompt, suggestion, other_strategy,
#                          candidate_actions, **kwargs)
#     # def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
#     #                  temperature: float = None) -> str:
#     #     output = chatgpt(self.model, messages, self.temperature)
#     #     return output
#     def memory_summary(self, phase):
#         prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase))
#         messages = [
#             {"role": 'system', "content": self.system_prompt},
#             {"role": 'user', "content": prompt}
#         ]
#         output = ""
#         discard = 0
#         while not output:
#             try:
#                 output = self.send_message(messages)
#             except OPENAI_MAX_TOKENS_ERROR as e:
#                 print("catch error: ", e.user_message)
#                 discard += 10
#                 prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase, discard))
#                 messages = [
#                     {"role": 'system', "content": self.system_prompt},
#                     {"role": 'user', "content": prompt}
#                 ]
#         match = re.search("(?<=Summary:).*", output, re.S)
#         summary = match.group().strip() if match else output
#         self.summary[phase] = summary
#         if self.summary:
#             format_summary = "\n".join(
#                 [f"Quest Phase Turn {key}:{value}" if key != "0" else f"Reveal Phase: {value}" for key, value
#                  in self.summary.items()])
#         else:
#             format_summary = "None"
#         self.log(f"{self.output_dir}/summary.txt",
#                  f"phase:{phase}\ninput:{prompt}\noutput:{output}\n--------------------")
#         return format_summary
#
# class ChatGPTBasedSAPARAgentForWerewolf(SAPARAgentForWerewolf, ChatGPTMessageSender):
#     def __init__(self, name, role, role_intro, game_goal, strategy, system_prompt: str, summary_prompt: str,
#                  analysis_prompt: str, plan_prompt: str, action_prompt: str, response_prompt: str, model, tokenizer,
#                  temperature, api_key, output_dir, suggestion_prompt: str, strategy_prompt: str, update_prompt: str,
#                  suggestion: str, other_strategy: str, candidate_actions: list, **kwargs):
#         super().__init__(name, role, role_intro, game_goal, strategy, system_prompt, summary_prompt, analysis_prompt,
#                          plan_prompt, action_prompt, response_prompt, model, tokenizer, temperature, api_key,
#                          output_dir, suggestion_prompt, strategy_prompt, update_prompt, suggestion, other_strategy,
#                          candidate_actions, **kwargs)
#     @override
#     def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
#                      temperature: float = None) -> str:
#         output = chatgpt(self.model, messages, self.temperature)
#         return output
#     def memory_summary(self, phase):
#         prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase))
#         messages = [
#             {"role": 'system', "content": self.system_prompt},
#             {"role": 'user', "content": prompt}
#         ]
#         output = ""
#         discard = 0
#         while not output:
#             try:
#                 output = self.send_message(messages)
#             except OPENAI_MAX_TOKENS_ERROR as e:
#                 print("catch error: ", e.user_message)
#                 discard += 10
#                 prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase, discard))
#                 messages = [
#                     {"role": 'system', "content": self.system_prompt},
#                     {"role": 'user', "content": prompt}
#                 ]
#         match = re.search("(?<=Summary:).*", output, re.S)
#         summary = match.group().strip() if match else output
#         self.summary[phase] = summary
#         if self.summary:
#             format_summary = "\n".join(
#                 [f"Quest Phase Turn {key}:{value}" if key != "0" else f"Reveal Phase: {value}" for key, value
#                  in self.summary.items()])
#         else:
#             format_summary = "None"
#         self.log(f"{self.output_dir}/summary.txt",
#                  f"phase:{phase}\ninput:{prompt}\noutput:{output}\n--------------------")
#         return format_summary
