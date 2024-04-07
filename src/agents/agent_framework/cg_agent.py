#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: cg_agent.py 
# @date: 2024/3/21 16:13 
#
# describe:
#
import json
import re
from typing import List, Any

import torch
from sentence_transformers import util

from ..abs_agent import Agent
from ...utils import write_json


class CGAgent(Agent):
    """
    Implement of the agent proposed by the paper "*Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf*".
    We name the agent as CG-Agent (Communication Game Agent).
    """

    def __init__(self, name: str, role: str, rule_role_prompt: str,
                 select_question_prompt: str, ask_question_prompt: str, generate_answer_prompt: str,
                 reflection_prompt: str, extract_suggestion_prompt: str, generate_response_prompt: str,
                 informativeness_prompt: str, question_list: list, retrival_model, freshness_k: int,
                 informativeness_n: int, experience_window: int, previous_exp_pool: list, output_dir: str,
                 use_summary: bool = False, **kwargs):
        super().__init__(name=name, role=role, **kwargs)
        self.rule_role_prompt = rule_role_prompt
        self.select_question_prompt = select_question_prompt
        self.ask_question_prompt = ask_question_prompt
        self.generate_answer_prompt = generate_answer_prompt
        self.reflection_prompt = reflection_prompt
        self.extract_suggestion_prompt = extract_suggestion_prompt
        self.generate_response_prompt = generate_response_prompt
        self.informativeness_prompt = informativeness_prompt
        self.question_list = question_list

        self.retrival_model = retrival_model

        self.freshness_k = freshness_k
        self.informativeness_n = informativeness_n
        self.experience_window = experience_window
        self.T = 3
        self.epsilon = 0.85
        self.bad_experience = []
        self.good_experience = []
        self.current_experience = []
        self.previous_exp_pool = previous_exp_pool
        self.use_summary = use_summary

        self.output_dir = output_dir

        self.phase = "{}-th {}"  # {t}-th {day_or_night}
        self.memory = {"name": [], "message": [], "informativeness": []}
        self.phase_memory = {}
        self.summary = {}

    def step(self, message: str) -> str:
        phase = message.split("|")[0]
        self.phase = phase
        message = message.split("|")[1]
        conversations = [
            {"role": 'system', "content": self.rule_role_prompt}
        ]
        # retrieval
        if self.memory.get("message"):
            if self.use_summary:
                r_t = self.summary_memory()
            else:
                r_t, conversations = self.retrival_memory(conversations)
        else:
            r_t = "None"
        # extract
        if self.previous_exp_pool:
            s_t, conversations = self.extract_suggestion(r_t, conversations)
        else:
            s_t = "None"
        # generate response
        prompt = self.generate_response_prompt.format(
            self.phase, self.name, self.role, message, r_t, s_t
        )
        # messages = [
        #     {"role": 'system', "content": self.rule_role_prompt},
        #     {"role": 'user', "content": prompt}
        # ]
        # output = self.__send_message(messages)
        conversations.append({"role": 'user', "content": prompt})
        output = self.send_message(conversations, temperature=0)
        self.log(f"{self.output_dir}/response.txt",
                 f"input:{conversations}\noutput:\n{output}\n--------------------")
        conversations.append({"role": 'assistant', "content": output})
        output = output.replace("\n", "")
        pattern = "(?<=My concise talking content:).*(?=<EOS>)"
        match = re.search(pattern, output)
        if match is None:
            pattern = "(?<=My concise talking content:).*"
            match = re.search(pattern, output)
        response = match.group().strip() if match else output
        self.update_memory("Host", message)
        self.update_memory(self.name, response)
        self.current_experience.append(
            [r_t, response, None]
        )
        return response

    def retrival_memory(self, conversations: List[dict]):
        # freshness
        names = self.memory.get("name", [])[-self.freshness_k:]
        messages = self.memory.get("message", [])[-self.freshness_k:]
        o_t = [f"{n}: {m}" for n, m in zip(names, messages)]

        # informativeness
        x = zip(self.memory.get("name", []),
                self.memory.get("message", []),
                self.memory.get("informativeness", []))
        x = sorted(x)
        v_t = [f"{i[0]}: {i[1]}" for i in x[-self.informativeness_n:]]

        # completeness
        # select question
        prompt = self.select_question_prompt.format(
            self.phase, self.name, self.role, self.question_list
        )
        # messages = [
        #     {"role": 'system', "content": self.rule_role_prompt},
        #     {"role": 'user', "content": prompt}
        # ]
        conversations.append({"role": 'user', "content": prompt})
        output = self.send_message(conversations)
        self.log(f"{self.output_dir}/select_question.txt",
                 f"input:{conversations}\noutput:\n{output}\n--------------------")
        conversations.append({"role": 'assistant', "content": output})
        selected_questions = output.split("#")

        prompt = self.ask_question_prompt.format(
            self.phase, self.name, self.role, selected_questions
        )
        # messages = [
        #     {"role": 'system', "content": self.rule_role_prompt},
        #     {"role": 'user', "content": prompt}
        # ]
        conversations.append({"role": 'user', "content": prompt})
        output = self.send_message(conversations)
        self.log(f"{self.output_dir}/ask_question.txt",
                 f"input:{conversations}\noutput:\n{output}\n--------------------")
        conversations.append({"role": 'assistant', "content": output})
        questions = output.split("#")

        # a_t = []
        candidate_answer = []
        # names = self.memory.get("name", [])
        documents = self.memory.get("message", [])
        documents_embedding = self.retrival_model.encode(documents)
        k = min(len(documents), self.T)
        for q in selected_questions + questions:
            q_embedding = self.retrival_model.encode(q)
            cos_scores = util.cos_sim(q_embedding, documents_embedding)[0]
            top_results = torch.topk(cos_scores, k=k)
            result = [documents[idx] for idx in top_results.indices]
            candidate_answer.append(result)

        # 并行提问加速 parallel questions for faster response
        q = ' '.join([f"{idx + 1}: {q_i}" for idx, q_i in enumerate(selected_questions + questions)])
        c = ' '.join([f"{idx + 1}: {c_i}" for idx, c_i in enumerate(candidate_answer)])
        prompt = self.generate_answer_prompt.format(
            self.phase, self.name, self.role, q, self.T, c
        )
        conversations.append({"role": 'user', "content": prompt})
        output = self.send_message(conversations)
        self.log(f"{self.output_dir}/generate_answer.txt",
                 f"input:{conversations}\noutput:\n{output}\n--------------------")
        self.log(f'{self.output_dir}/qa.txt', f"input:\n{prompt}\noutput:\n{output}")
        a_t = output
        conversations.append({"role": 'assistant', "content": output})

        prompt = "{}".format(o_t + v_t) + self.reflection_prompt.format(
            self.phase, self.name, self.role, a_t, self.role
        )
        conversations.append({"role": 'user', "content": prompt})
        output = self.send_message(conversations)
        self.log(f"{self.output_dir}/reflection.txt",
                 f"input:{conversations}\noutput:\n{output}\n--------------------")
        conversations.append({"role": 'assistant', "content": output})
        r_t = output
        return r_t, conversations

    def summary_memory(self):
        names = self.phase_memory.get(self.phase, {}).get("name", [])
        messages = self.phase_memory.get(self.phase, {}).get("message", [])
        conversations = [f"{n}: {m}" for n, m in zip(names, messages)]
        prompt = """
        Please summarize the conversations of current phase in concise sentences. <fill_in> represents the content of summarization.

        Conversations: {}

        Summary: <fill_in>
        """.format(conversations)
        messages = [
            {"role": 'system', "content": self.rule_role_prompt},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_message(messages)
        prompt = self.summary[self.phase] = output
        """Now its the {}. Assuming you are {}, the {}, what insights can you summarize with few sentences based on the  
        descriptions of previous rounds {} in heart for helping continue the talking and achieving your objective? For example: As the {}, I 
        observed that... I think that... But I am... So...
        """.format(self.phase, self.name, self.role, self.summary, self.role)
        messages = [
            {"role": 'system', "content": self.rule_role_prompt},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_message(messages)
        return output

    def extract_suggestion(self, r_t, conversations):
        r_pool = []
        g_pool = []
        for r, g, s in self.previous_exp_pool:
            r_pool.append(r)
            g_pool.append(g)
        d_embedding = self.retrival_model.encode(r_pool)
        q_embedding = self.retrival_model.encode(r_t)
        cos_scores = util.cos_sim(q_embedding, d_embedding)[0]
        # top_results = torch.topk(cos_scores, k=10)
        sub_e_idx = torch.where(cos_scores > self.epsilon)[0]
        sub_e = [self.previous_exp_pool[idx] for idx in sub_e_idx]
        sub_e = sorted(sub_e, key=lambda x: x[2])[:min(len(sub_e), self.experience_window)]
        if sub_e:
            good_experience = [e[1] for e in sub_e[:-1]]
            bad_experience = [e[1] for e in sub_e[:-1]]
        else:
            good_experience = []
            bad_experience = []
        prompt = self.extract_suggestion_prompt.format(
            bad_experience, good_experience
        )
        conversations.append({"role": 'user', "content": prompt})
        output = self.send_message(conversations)
        self.log(f"{self.output_dir}/suggestion.txt",
                 f"input:{conversations}\noutput:\n{output}\n--------------------")
        conversations.append({"role": 'assistant', "content": output})
        s_t = output
        return s_t, conversations

    def reflection(self, player_role_mapping: dict, file_name: str, winners: list, duration: int):
        score = duration if self.role not in winners else 1000 - duration
        exp = [[e[0], e[1], score] for e in self.current_experience]
        self.previous_exp_pool.extend(exp)
        write_json(
            data=self.previous_exp_pool,
            path=file_name
        )

    def receive(self, name: str, message: str) -> None:
        phase = message.split("|")[0]
        self.phase = phase
        message = message.split("|")[1]
        self.update_memory(name, message)

    def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                      temperature: float = None) -> str:
        raise NotImplementedError("Interaction with LLM is not implemented in agent framework class.")

    def update_memory(self, name: str, message: str):
        prompt = self.informativeness_prompt.format(
            f"{name}: {message}"
        )
        messages = [
            {"role": 'system', "content": ""},
            {"role": 'user', "content": prompt}
        ]
        output = self.send_message(messages)
        scores = re.findall("\d+", output)
        score = scores[-1] if scores else "1"
        score = int(score)
        self.memory['name'].append(name)
        self.memory['message'].append(message)
        self.memory['informativeness'].append(score)

    def memory_to_json(self, phase: str = None):
        if phase is None:
            json_data = []
            for r, m in zip(self.memory.get('name', []), self.memory.get('message', [])):
                json_data.append(
                    {'name': r, 'message': m}
                )
            # json_data = json_data[-self.memory_window:]
            doc = json.dumps(json_data, indent=4, ensure_ascii=False)
            return doc
        else:
            json_data = []
            for r, m in zip(self.phase_memory.get(phase, {}).get('name', []),
                            self.phase_memory.get(phase, {}).get('message', [])):
                json_data.append(
                    {'name': r, 'message': m}
                )
            doc = json.dumps(json_data, indent=4, ensure_ascii=False)
            return doc

    @staticmethod
    def log(file, data):
        with open(file, mode='a+', encoding='utf-8') as f:
            f.write(data)
        f.close()
