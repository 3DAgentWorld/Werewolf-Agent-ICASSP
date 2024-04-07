#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: sapar_agent.py 
# @date: 2024/3/21 16:20 
#
# describe:
#
import json
import re
import time
from typing import List, Any

from ..abs_agent import Agent
from ...utils import write_json


class SAPARAgent(Agent):
    """
    We name the agent used in the paper "*LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay*"
    as SAPAR-Agent (Summary-Analysis-Planning-Action-Response)
    """

    def __init__(self, name, role, role_intro, game_goal, strategy, system_prompt: str, summary_prompt: str,
                 analysis_prompt: str, plan_prompt: str, action_prompt: str, response_prompt: str,
                 suggestion_prompt: str, strategy_prompt: str, update_prompt: str, suggestion: str, other_strategy: str,
                 candidate_actions: list, output_dir: str, use_analysis=True, use_plan=True,
                 use_action=True, reflection_other=True, improve_strategy=True, **kwargs):
        super().__init__(name=name, role=role, **kwargs)
        self.introduction = role_intro
        self.game_goal = game_goal
        self.strategy = strategy

        self.system_prompt = system_prompt
        self.summary_prompt = summary_prompt
        self.analysis_prompt = analysis_prompt
        self.plan_prompt = plan_prompt
        self.action_prompt = action_prompt
        self.response_prompt = response_prompt
        self.suggestion_prompt = suggestion_prompt
        self.strategy_prompt = strategy_prompt
        self.update_prompt = update_prompt
        self.previous_suggestion = suggestion
        self.previous_other_strategy = other_strategy

        self.use_analysis = use_analysis
        self.use_plan = use_plan
        self.use_action = use_action
        self.reflection_other = reflection_other
        self.improve_strategy = improve_strategy

        self.memory_window = 30
        self.T = 3
        self.candidate_actions = candidate_actions
        self.output_dir = output_dir

        self.phase = 0
        self.memory = {"message_type": [], "name": [], "message": [], "phase": []}
        self.phase_memory = {}
        self.summary = {}
        self.plan = {}

    def step(self, message: str) -> str:
        """
        :param message:
        :return:
        """
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        output = temp_phase
        pattern = "\d+"
        matches = re.findall(pattern, output)
        phase = matches[-1] if matches else "0"
        # summary
        format_summary = self.get_summary()
        t_analysis_start = time.time()
        # analysis
        if self.use_analysis and format_summary != "None":
            analysis = self.make_analysis(phase, format_summary)
        else:
            analysis = "None"
        t_analysis = time.time() - t_analysis_start
        # planning
        t_plan_start = time.time()
        if self.use_plan:
            format_plan = self.make_plan(phase, format_summary, analysis)
        else:
            format_plan = "None"
        t_plan = time.time() - t_plan_start

        # action
        t_action_start = time.time()
        if self.use_action:
            action = self.make_action(phase, format_summary, format_plan, analysis, message)
        else:
            action = None
        t_action = time.time() - t_action_start

        # response
        t_response_start = time.time()
        response = self.make_response(phase, format_summary, format_plan, action, message)
        t_response = time.time() - t_response_start
        self.update_private("Host", message, phase)
        self.update_private("Self", response, phase)
        t_summary_start = time.time()
        _ = self.memory_summary(phase)
        t_summary = time.time() - t_summary_start

        self.log(f"{self.output_dir}/time_cost.txt",
                 f"Summary: {t_summary}\nAnalysis: {t_analysis}\nPlan: {t_plan}\nAction: {t_action}\nResponse: {t_response}\n")
        return response

    def memory_summary(self, phase):
        prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase))
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_message(messages)
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

    def make_analysis(self, phase, format_summary):
        prompt = self.analysis_prompt.format(
            name=self.name, phase=self.phase, role=self.role, summary=format_summary
        )
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_message(messages)
        self.log(f"{self.output_dir}/step_reflection.txt",
                 f"phase:{phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        return output

    def make_plan(self, phase, format_summary, analysis):
        if self.plan:
            format_previous_plan = '\n'.join(
                [
                    f"Quest Phase Turn {i}: {self.plan.get(str(i), 'None')}" if i != 0 else f"Reveal Phase: {self.plan.get(str(i), 'None')}"
                    for i in range(int(phase) + 1)]
            )
        else:
            format_previous_plan = "None"

        following_format = '\n'.join(
            [f"Quest Phase Turn {i}: <your_plan_{i}>" if
             i != 0 else f"Reveal Phase: <your_plan_0>" for i in range(int(phase), 6)]
        )
        prompt = self.plan_prompt.format(
            name=self.name, phase=self.phase, role=self.role, introduction=self.introduction, goal=self.game_goal,
            strategy=self.strategy,
            previous_plan=format_previous_plan, summary=format_summary, analysis=analysis, plan=following_format)
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_message(messages)
        # format_plans = output.split('\n')
        match = re.search('<plan>(.*?)</plan>', output, re.S)
        format_plans = match.group().split('\n') if match else output.split('\n')
        plans = [f.split(":", 1) for f in format_plans]
        dict_plans = {}
        for plan in plans:
            if len(plan) == 2:
                match = re.search('\d+', plan[0])
                c_phase = match.group() if match else None
                if c_phase is None and plan[0].lower().startswith("reveal phase"):
                    c_phase = "0"
                c_plan = plan[1].strip()
                if c_phase:
                    dict_plans[c_phase] = c_plan
        self.plan.update(dict_plans)
        self.log(f"{self.output_dir}/plan.txt",
                 f"phase:{phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        format_plan = '\n'.join([
            f"Quest Phase Round {str(c_phase)}:{self.plan.get(str(c_phase))}" if str(
                c_phase) != "0" else f"Reveal Phase: {self.plan.get(str(c_phase))}"
            for c_phase in range(int(phase), 6)])
        return format_plan

    def make_action(self, phase, format_summary, format_plan, analysis, message):
        prompt = self.action_prompt.format(name=self.name, phase=self.phase, role=self.role,
                                           introduction=self.introduction, goal=self.game_goal,
                                           strategy=self.strategy, candidate_actions=self.candidate_actions,
                                           summary=format_summary, analysis=analysis, plan=format_plan,
                                           question=message)

        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_message(messages)
        self.log(f"{self.output_dir}/actions.txt", f"input:{prompt}\noutput:\n{output}\n--------------------")
        actions = re.findall("(?<=<actions>).*?(?=</actions>)", output, re.S)
        if not actions:
            actions = re.findall("(?<=<output>).*?(?=</output>)", output, re.S)
            if not actions:
                return output
        return actions

    def make_response(self, phase, format_summary, format_plan, actions, message):
        if self.use_action:
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
                strategy=self.strategy,
                summary=format_summary, plan=format_plan, question=message, actions=actions)
        else:
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
                strategy=self.strategy,
                summary=format_summary, plan=format_plan, question=message, actions="None")
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]

        output = self.send_message(messages)

        # input_tokens = len(self.tokenizer.encode(self.system_prompt)) + len(self.tokenizer.encode(prompt))
        # output_tokens = len(self.tokenizer.encode(output))
        # self.log(f"{self.output_dir}/gpt_response_tokens.txt", f"input:{input_tokens} output:{output_tokens}\n")

        match = re.search("(?<=<response>).*?(?=</response>)", output, re.S)
        self.log(f"{self.output_dir}/response.txt", f"input:{prompt}\noutput:\n{output}\n--------------------")
        response = match.group().strip() if match else output
        return response

    def reflection(self, player_role_mapping: dict, file_name: str, winners: list, duration: int):
        p_r_mapping = '\n'.join([f"{k}:{v}" for k, v in player_role_mapping.items()])
        format_summary = "\n".join(
            [f"Quest Phase Turn {key}:{value}" if key != "0" else f"Reveal Phase: {value}" for key, value
             in self.summary.items()])
        if self.reflection_other:
            prompt = self.strategy_prompt.format(
                name=self.name, roles=p_r_mapping, summaries=format_summary, strategies=self.previous_other_strategy
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            role_strategy = self.send_message(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{role_strategy}\n--------------------")
        else:
            role_strategy = "None"

        if self.improve_strategy:
            prompt = self.suggestion_prompt.format(
                name=self.name, role=self.role, roles=p_r_mapping, summaries=format_summary, goal=self.game_goal,
                strategy=self.strategy, previous_suggestions=self.previous_suggestion
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            suggestion = self.send_message(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{suggestion}\n--------------------")

            prompt = self.update_prompt.format(
                name=self.name, role=self.role, strategy=self.strategy, suggestions=suggestion
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            output = self.send_message(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{output}\n--------------------")
            match = re.search("(?<=<strategy>).*?(?=</strategy>)", output)
            strategy = match.group() if match else output
        else:
            suggestion = "None"
            strategy = self.strategy

        write_json(
            data={"strategy": strategy, "suggestion": suggestion, "other_strategy": role_strategy},
            path=file_name
        )

    def receive(self, name: str, message: str) -> None:
        temp_phase = message.split("|")[0]
        self.phase = temp_phase
        message = message.split("|")[1]

        output = temp_phase
        pattern = "\d+"
        matches = re.findall(pattern, output)
        phase = matches[0] if matches else '0'
        self.update_public(name, message, phase)
        _ = self.memory_summary(phase)
        return

    def get_summary(self):
        if self.summary:
            format_summary = "\n".join(
                [f"Quest Phase Turn {key}:{value}" if key != "0" else f"Reveal Phase: {value}" for key, value
                 in self.summary.items()])
        else:
            format_summary = "None"
        return format_summary

    def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                     temperature: float = None) -> str:
        raise NotImplementedError("Interaction with LLM is not implemented in agent framework class.")

    def memory_to_json(self, phase: str = None, discard: int = None):
        if phase is None:
            json_data = []
            for t, r, m, p in zip(self.memory.get('message_type', []),
                                  self.memory.get('name', []),
                                  self.memory.get('message', []),
                                  self.memory.get('phase', [])):
                json_data.append(
                    {'message_type': t, 'name': r, 'message': m, 'phase': p}
                )
            json_data = json_data[-self.memory_window:]
            doc = json.dumps(json_data, indent=4, ensure_ascii=False)
            return doc
        else:
            json_data = []
            for t, r, m, p in zip(self.phase_memory.get(phase, {}).get('message_type', []),
                                  self.phase_memory.get(phase, {}).get('name', []),
                                  self.phase_memory.get(phase, {}).get('message', []),
                                  self.phase_memory.get(phase, {}).get('phase', [])):
                json_data.append(
                    {'message_type': t, 'name': r, 'message': m, 'phase': p}
                )
            if discard:
                json_data = json_data[discard:]
            doc = json.dumps(json_data, indent=4, ensure_ascii=False)
            return doc

    def update_private(self, name, message, phase: str = None) -> None:
        self.memory['message_type'].append("private")
        self.memory['name'].append(name)
        self.memory['message'].append(message)
        self.memory['phase'].append(phase)
        if phase not in self.phase_memory:
            self.phase_memory[phase] = {"message_type": [], "name": [], "message": [], "phase": []}
        self.phase_memory[phase]['message_type'].append("private")
        self.phase_memory[phase]['name'].append(name)
        self.phase_memory[phase]['message'].append(message)
        self.phase_memory[phase]['phase'].append(phase)

    def update_public(self, name, message, phase: str = None) -> None:
        self.memory['message_type'].append("public")
        self.memory['name'].append(name)
        self.memory['message'].append(message)
        self.memory['phase'].append(phase)
        if phase is not None:
            if phase not in self.phase_memory:
                self.phase_memory[phase] = {"message_type": [], "name": [], "message": [], "phase": []}
            self.phase_memory[phase]['message_type'].append("public")
            self.phase_memory[phase]['name'].append(name)
            self.phase_memory[phase]['message'].append(message)
            self.phase_memory[phase]['phase'].append(phase)

    @staticmethod
    def log(file, data):
        with open(file, mode='a+', encoding='utf-8') as f:
            f.write(data)
        f.close()


class SAPARAgentForWerewolf(SAPARAgent):
    """
    We name the agent used in the paper "*LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay*"
    as SAPAR-Agent (Summary-Analysis-Planning-Action-Response)
    """

    def memory_summary(self, phase):
        prompt = self.summary_prompt.format(name=self.name, conversation=self.memory_to_json(phase))
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_message(messages)
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

    def make_plan(self, phase, format_summary, analysis):
        if self.plan:
            format_previous_plan = '\n'.join(
                [
                    f"Day {i}: {self.plan.get(str(i), 'None')}" for i in range(int(phase) + 1)]
            )
        else:
            format_previous_plan = "None"

        following_format = '\n'.join(
            [f"Day {i}: <your_plan_{i}>" for i in range(int(phase), 6)]
        )
        prompt = self.plan_prompt.format(
            name=self.name, phase=self.phase, role=self.role, introduction=self.introduction, goal=self.game_goal,
            strategy=self.strategy,
            previous_plan=format_previous_plan, summary=format_summary, analysis=analysis, plan=following_format)
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_message(messages)
        # format_plans = output.split('\n')
        match = re.search('<plan>(.*?)</plan>', output, re.S)
        format_plans = match.group().split('\n') if match else output.split('\n')
        plans = [f.split(":", 1) for f in format_plans]
        dict_plans = {}
        for plan in plans:
            if len(plan) == 2:
                match = re.search('\d+', plan[0])
                c_phase = match.group() if match else None
                if c_phase is None and plan[0].lower().startswith("reveal phase"):
                    c_phase = "0"
                c_plan = plan[1].strip()
                if c_phase:
                    dict_plans[c_phase] = c_plan
        self.plan.update(dict_plans)
        self.log(f"{self.output_dir}/plan.txt",
                 f"phase:{phase}\ninput:{prompt}\noutput:\n{output}\n--------------------")
        format_plan = '\n'.join([
            f"Quest Phase Round {str(c_phase)}:{self.plan.get(str(c_phase))}" if str(
                c_phase) != "0" else f"Reveal Phase: {self.plan.get(str(c_phase))}"
            for c_phase in range(int(phase), 6)])
        return format_plan

    def make_response(self, phase, format_summary, format_plan, actions, message):
        if self.use_action:
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
                strategy=self.strategy,
                summary=format_summary, plan=format_plan, question=message, actions=actions)
        else:
            prompt = self.response_prompt.format(
                name=self.name, phase=self.phase, role=self.role, introduction=self.introduction,
                strategy=self.strategy,
                summary=format_summary, plan=format_plan, question=message, actions="None")
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]

        output = self.send_message(messages)

        # input_tokens = len(self.tokenizer.encode(self.system_prompt)) + len(self.tokenizer.encode(prompt))
        # output_tokens = len(self.tokenizer.encode(output))
        # self.log(f"{self.output_dir}/gpt_response_tokens.txt", f"input:{input_tokens} output:{output_tokens}\n")

        match = re.search("(?<=<response>).*?(?=</response>)", output, re.S)
        self.log(f"{self.output_dir}/response.txt", f"input:{prompt}\noutput:\n{output}\n--------------------")
        response = match.group().strip() if match else output
        return response

    def reflection(self, player_role_mapping: dict, file_name: str, winners: list, duration: int):
        p_r_mapping = '\n'.join([f"{k}:{v}" for k, v in player_role_mapping.items()])
        format_summary = "\n".join(
            [f"Day {key}:{value}" for key, value in self.summary.items()])
        if self.reflection_other:
            prompt = self.strategy_prompt.format(
                name=self.name, roles=p_r_mapping, summaries=format_summary, strategies=self.previous_other_strategy
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            role_strategy = self.send_message(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{role_strategy}\n--------------------")
        else:
            role_strategy = "None"

        if self.improve_strategy:
            prompt = self.suggestion_prompt.format(
                name=self.name, role=self.role, roles=p_r_mapping, summaries=format_summary, goal=self.game_goal,
                strategy=self.strategy, previous_suggestions=self.previous_suggestion
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            suggestion = self.send_message(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{suggestion}\n--------------------")

            prompt = self.update_prompt.format(
                name=self.name, role=self.role, strategy=self.strategy, suggestions=suggestion
            )
            messages = [
                {"role": 'system', "content": ""},
                {"role": 'user', "content": prompt}
            ]
            output = self.send_message(messages)
            self.log(f"{self.output_dir}/round_reflection.txt",
                     f"input:{prompt}\noutput:\n{output}\n--------------------")
            match = re.search("(?<=<strategy>).*?(?=</strategy>)", output)
            strategy = match.group() if match else output
        else:
            suggestion = "None"
            strategy = self.strategy

        write_json(
            data={"strategy": strategy, "suggestion": suggestion, "other_strategy": role_strategy},
            path=file_name
        )

    def get_summary(self):
        if self.summary:
            format_summary = "\n".join(
                [f"Day {key}:{value}" for key, value in self.summary.items()])
        else:
            format_summary = "None"
        return format_summary


class SAPARAgentForAvalon(SAPARAgent):
    """
        We name the agent used in the paper "*LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay*"
        as SAPAR-Agent (Summary-Analysis-Planning-Action-Response)
    """
