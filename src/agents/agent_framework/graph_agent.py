#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: graph_agent.py 
# @date: 2024/3/21 16:55 
#
# describe:
#
import re
from typing import List, Any

from ..abs_agent import Agent


class GraphThoughtAgent(Agent):
    def __init__(self, name, role, system_prompt: str, graph_generate_prompt: str, reasoning_prompt: str,
                 candidate_actions: List[str], output_dir: str, controllable=None, **kwargs):
        super().__init__(name=name, role=role, **kwargs)
        self.system_prompt = system_prompt
        self.graph_generate_prompt = graph_generate_prompt
        self.reasoning_prompt = reasoning_prompt

        self.system_prompt = system_prompt
        self.controllable = controllable

        self.freshness_k = 10

        self.candidate_actions = candidate_actions
        self.graph_example = """graph {
    Q1 --> Q;
    Q2 --> Q;
    Q3 --> Q1;
    Q3 --> Q2;
    Q4 --> Q1;
    Q5 --> Q2;
}"""
        self.phase = "Reveal"
        self.memory = []

        self.output_dir = output_dir

    def step(self, message: str) -> str:
        temp_phase = message.split("|")[0]
        message = message.split("|")[1]

        output = temp_phase
        pattern = "\d+"
        matches = re.findall(pattern, output)
        phase = matches[0] if matches else '0'
        self.phase = "Reveal Phase" if phase == '0' else f"Quest {phase} Phase"

        if self.phase == "Reveal Phase":
            response = self.fast_decision(message)
        else:
            response = self.decision(message)

        self.update_memory('Host', message, False)
        self.update_memory('Self', response, False)

        return response

    def receive(self, name: str, message: str) -> None:
        temp_phase = message.split("|")[0]
        message = message.split("|")[1]

        output = temp_phase
        pattern = "\d+"
        matches = re.findall(pattern, output)
        phase = matches[0] if matches else '0'
        self.phase = "Reveal Phase" if phase == '0' else f"Quest {phase} Phase"
        self.update_memory(name, message, True)

    def statement_retrival(self):
        fresh_memory = self.memory[-self.freshness_k:]
        statements = []
        for data in fresh_memory:
            statements.append(f"{data.get('name')}: {data.get('message')}")
        return statements

    def fast_decision(self, instruction):
        statements = "None"
        pairs = "None"
        prompt = self.reasoning_prompt.format(statements=statements, pairs=pairs,
                                              question=instruction, name=self.name, role=self.role, phase=self.phase)
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]
        response = self.send_messages(messages)
        self.log(f"{self.output_dir}/response.txt", f"input: {prompt}\noutput: {response}----------\n")

        return response

    def decision(self, instruction):
        prompt = self.graph_generate_prompt.format(question=instruction, name=self.name, role=self.role,
                                                   phase=self.phase, graph=self.graph_example)
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_messages(messages)
        self.log(f"{self.output_dir}/graph_generate.txt", f"input: {prompt}\noutput: {output}----------\n")
        match = re.search("(?<=```).*(?=```)", output, re.S)
        graph_str = match.group() if match else ""
        if graph_str:
            graph_dict, node_list = self.parse_graph(output)
        else:
            graph_dict, node_list = {"Q": None}, ["Q"]

        answer_mapping = {q: None for q in node_list}

        question_mapping = {"Q": instruction}
        for q in node_list:
            if q == "Q":
                continue
            pattern = f"(?<={q}:).*"
            match = re.search(pattern, output)
            q_content = match.group() if match else ""
            question_mapping[q] = q_content

        statements = self.statement_retrival()
        statement_str = '\n'.join(statements)
        response = self.graph_reasoning("Q", graph_dict.get("Q"), statement_str, answer_mapping, question_mapping)
        return response

    def graph_reasoning(self, q: str, q_condition: dict, statements: str, answer_mapping: dict, question_mapping: dict):
        q_answer = answer_mapping.get(q)
        if q_answer is not None:
            return q_answer
        if q_condition is not None:
            qa_pairs = []
            for sub_q, sub_condition in q_condition.items():
                sub_ans = answer_mapping.get(sub_q)
                if sub_ans is None:
                    sub_ans = self.graph_reasoning(sub_q, sub_condition, statements, answer_mapping, question_mapping)
                qa_pairs.append(f"{question_mapping.get(sub_q, '')}: {sub_ans}")
            qa_pairs_str = '\n'.join(qa_pairs)
        else:
            qa_pairs_str = "None"

        q_content = question_mapping.get(q, "")
        prompt = self.reasoning_prompt.format(statements=statements, pairs=qa_pairs_str, question=q_content,
                                              name=self.name, role=self.role, phase=self.phase)
        messages = [
            {"role": 'system', "content": self.system_prompt},
            {"role": 'user', "content": prompt}
        ]
        q_answer = self.send_messages(messages)
        answer_mapping[q] = q_answer
        self.log(f"{self.output_dir}/reasoning.txt", f"input: {prompt}\noutput: {q_answer}----------\n")
        return q_answer

    def parse_graph(self, graph_str):
        graph = {}
        nodes = set()
        lines = graph_str.split('\n')
        for line in lines:
            if line.strip().startswith('graph'):
                continue
            if line.strip().startswith('}'):
                break
            match = re.match(r'\s*([\w\d]+)\s*-->\s*([\w\d]+)', line)
            if match:
                node1, node2 = match.groups()
                nodes.add(node1)
                nodes.add(node2)
                if node2 not in graph:
                    graph[node2] = {}
                if node1 not in graph[node2]:
                    graph[node2][node1] = None
        return graph, list(nodes)

    def update_memory(self, name, message, retrival: bool):
        data = {'name': name, 'message': message, 'phase': self.phase}
        self.memory.append(data)
        return

    @staticmethod
    def log(file, data):
        with open(file, mode='a+', encoding='utf-8') as f:
            f.write(data)
        f.close()

    def send_messages(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                      temperature: float = None) -> str:
        raise NotImplementedError("Interaction with LLM is not implemented in agent framework class.")
