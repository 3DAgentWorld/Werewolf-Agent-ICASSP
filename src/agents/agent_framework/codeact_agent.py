import torch
import sys
from sentence_transformers import util
import json
import re
from typing import List, Any
import warnings
from ..abs_agent import Agent


class CodeActAgent(Agent):
    """
    We name the agent used in the paper "*LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay*"
    as CodeAct-Agent ()
    """

    def __init__(self, name: str, role: str, rule_role_prompt: str,
                 private_information: str, current_team_number: int,
                 code_generate_prompt: str, output_dir: str, total_player_number: int,
                 good_number: int, bad_number: int, k: int, informativeness_prompt: str,
                 generate_response_prompt: str, generate_leader_response_prompt: str,
                 informativeness_n: int, select_question_prompt: str, question_list: list,
                 ask_question_prompt: str, retrieval_model, generate_answer_prompt: str,
                 reflection_prompt: str, llama_model, llama_tokenizer,
                 use_summary: bool = False,
                 **kwargs):
        super().__init__(name=name, role=role, **kwargs)
        self.total_player_number = total_player_number
        self.good_number = good_number  # number of good players
        self.bad_number = bad_number
        self.rule_role_prompt = rule_role_prompt
        self.informativeness_prompt = informativeness_prompt
        self.generate_response_prompt = generate_response_prompt
        self.generate_leader_response_prompt = generate_leader_response_prompt
        self.select_question_prompt = select_question_prompt
        self.ask_question_prompt = ask_question_prompt
        self.generate_answer_prompt = generate_answer_prompt
        self.reflection_prompt = reflection_prompt

        self.private_information = private_information
        self.public_information = [[0, []], [0, []], [0, []], [0, []], [0, []]]

        self.phase = "{}-th {}"  # {t}-th {day_or_night}
        self.memory = {"name": [], "message": [], "informativeness": []}
        self.phase_memory = {}
        self.summary = {}
        self.question_list = question_list
        self.retrival_model = retrieval_model

        self.teams = []  # list of possible good people
        self.current_team_number = current_team_number  # how many people are in the quest
        self.code_generate_prompt = code_generate_prompt
        self.output_dir = output_dir

        self.k = k  # memory window size
        self.use_summary = use_summary
        self.current_quest_number = 1  # quest number
        self.informativeness_n = informativeness_n
        self.T = 3

        self.code_runtime_out = 5
        self.llama_model = llama_model
        self.llama_tokenizer = llama_tokenizer

    def step(self, message: str) -> str:
        phase = message.split("|")[0]
        self.phase = phase
        message = message.split("|")[1]
        conversations = [
            {"role": 'system', "content": self.rule_role_prompt}
        ]
        # if "The player who receives the" in message:
        #     pattern = r"The player who receives the (^.{3,20}$) card"
        #     role = re.search(pattern, message)
        #     self.role = role.group(1)
        #     print(self.role)
        if "You can see" in message:
            pattern = r"You can see (.+) opening eyes"
            seen = re.search(pattern, message)
            if seen:
                if self.role == "Assassin":
                    self.private_information += "Your teammates are " + seen.group(1)
                elif self.role == "Morgana":
                    self.private_information += "Your teammates are " + seen.group(1)
            else:
                pattern = "You can see (.+) raising hands"
                seen = re.search(pattern, message)
                if seen:
                    if self.role == "Merlin":
                        self.private_information += "You know that " + seen.group(1) + " are bad."
                    if self.role == "Percival":
                        self.private_information += "You know that between " + seen.group(1) + ", one of them is Merlin and one of them is Morgana."
            print("Private information updated! " + self.private_information)
        if "The quest leader decides that the player" in message:
            pattern = r'player (\d+)'
            players = re.findall(pattern, message)
            self.public_information[int(self.current_quest_number) - 1][1] = players
            print("Self.public_information update: ")
            print(self.public_information)
            print("\n")
        # retrieval
        if self.memory.get("message"):
            if self.use_summary:
                r_t = self.summary_memory()
            else:
                r_t, conversations = self.retrival_memory(conversations)
        else:
            r_t = "None"
        # print("Retrieval memory: ")
        # print(r_t)
        # print("\n")
        if "Please answer the numbers of the players you selected." in message:
            self.code_runtime_out = 5
            current_number = re.search(r"(\d+)", message)
            self.current_team_number = current_number.group(1)
            self.generate_code_program()
            prompt = self.generate_leader_response_prompt.format(
                self.phase, self.name, self.role, self.teams, message
            )
        else:
            prompt = self.generate_response_prompt.format(
                self.phase, self.name, self.role, message, r_t
            )
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
        # response = output
        self.update_memory("Host", message)
        self.update_memory(self.name, response)
        return response

    def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                      temperature: float = None) -> str:
        raise NotImplementedError("Interaction with LLM is not implemented in agent framework class.")

    def send_llama_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                     temperature: float = 0.01) -> str:

        if temperature <= 0:
            # ValueError: `temperature` (=0) has to be a strictly positive float, otherwise your next token scores will be invalid.
            warnings.warn(
                f"temperature has to be a strictly positive float (got temperature = {temperature}, now temperater has set to 0.01 to avoid possible error.)")
            temperature = 0.01
        inputs = self.to_llama_input_format(messages)
        input_ids = self.llama_tokenizer(inputs, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:1")
        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "temperature": temperature,
            "eos_token_id": self.llama_tokenizer.eos_token_id,
            "bos_token_id": self.llama_tokenizer.bos_token_id,
            "pad_token_id": self.llama_tokenizer.pad_token_id
        }
        generate_ids = self.llama_model.generate(**generate_input)
        output = self.llama_tokenizer.decode(generate_ids[0][len(input_ids[0]):]).strip(' ').strip('</s>')
        return output

    def to_llama_input_format(self, messages: List[dict], discard=0):
        # [f"<s>[INST] <<SYS>>\n{self.system_prompt}<</SYS>>\n\n{prompt} [/INST]"]
        inputs = ''
        first_message = messages[0]
        if first_message.get('role') == 'system':
            has_system_prompt = True
            next_role = 'user'
            system_prompt = first_message.get('content', '')
            messages = [messages[0]] + messages[discard * 2 + 1:]
        else:
            has_system_prompt = False
            next_role = 'user'
            system_prompt = ''
            messages = messages[discard * 2:]
        idx = 1 if has_system_prompt else 0
        user_messages = []
        assistant_messages = []
        while idx < len(messages):
            current_message = messages[idx]
            if current_message.get('role') != next_role:
                raise ValueError()
            if current_message.get('role') == 'user':
                user_messages.append(current_message.get('content', ''))
                next_role = 'assistant'
            if current_message.get('role') == 'assistant':
                assistant_messages.append(current_message.get('content', ''))
                next_role = 'user'
            idx += 1
        if len(user_messages) != len(assistant_messages) + 1:
            raise ValueError()
        if len(user_messages) == 1:
            inputs = f"<s>[INST] <<SYS>>\n{system_prompt}<</SYS>>\n\n{user_messages[0]} [/INST]"
        else:
            first_user_message = user_messages[0]
            first_assistant_message = assistant_messages[0]
            inputs = f"<s>[INST] <<SYS>>\n{system_prompt}<</SYS>>\n\n{first_user_message} [/INST] {first_assistant_message}</s>"
            for user_message, assistant_message in zip(user_messages[1:-1], assistant_messages[1:]):
                inputs += f"<s>[INST] {user_message} [/INST] {assistant_message}</s>"
            last_user_message = user_messages[-1]
            inputs += f"<s>[INST] {last_user_message} [/INST]"
        return inputs


    def receive(self, name: str, message: str) -> None:
        phase = message.split("|")[0]
        self.phase = phase
        message = message.split("|")[1]
        self.update_memory(name, message)
        # print("Receive Here!")
        if "Please start discussing the candidates for the turn" in message:
            quest_number = re.search(r"Please start discussing the candidates for the turn (\d+) quest", message)
            self.current_quest_number = quest_number.group(1)
            # print("Current quest number: " + self.current_quest_number + "\n")
        if "The quest leader decides that the player" in message:
            pattern = r'player (\d+)'
            players = re.findall(pattern, message)
            self.public_information[int(self.current_quest_number) - 1][1] = players
            print("Self.public_information update: ")
            print(self.public_information)
            print("\n")
        if "quest successful" in message:
            self.public_information[int(self.current_quest_number) - 1][0] = 1
            print("Quest successful update!")

    def generate_code_program(self, error_message: str = ""):
        public_information = ""
        for i in range(0, 5):
            if self.public_information[i][1]:
                if self.public_information[i][0] == 0:
                    public_information += "In quest {}, players {} are chosen as candidates. The quest result is {}. \n".format(
                        i + 1, self.public_information[i][1], "failed")
                elif self.public_information[i][0] == 1:
                    public_information += "In quest {}, players {} are chosen as candidates. The quest result is {}. \n".format(
                        i + 1, self.public_information[i][1], "succeeded")
        print("Self.public_information: ")
        print(self.public_information)
        print("Public information: " + public_information)
        prompt = self.code_generate_prompt.format(
            self.role, self.total_player_number, self.good_number, self.bad_number, self.private_information,
            public_information, self.current_team_number
        )
        if error_message:
            prompt += ("\n\nThere was an error in last generation, please regenerate. Error message: " + error_message)
        example_prompt = """
        Now there are 7 players in the game, including 4 players in good side and 3 players in bad side.
        You now have private information: 
        Public information:In quest 1, players [2, 4, 7] are chosen as candidates. The quest result is succeeded.
        In quest 2, players [1, 2, 4] are chosen as candidates. The quest result is failed. 
        In quest 3, players [1, 2, 3, 7] are chosen as candidates. The quest result is failed.
        Please generate a Python program that selects 4 players most likely to be good person. Your answer should start with 'Action Output: '
        """
        example = """
        Action Output:
        players = [None] * 7 # Initialize 7 players list with None
        # Apply Round 2 information
        players[1] = True # Player 2 is good
        players[3] = True # Player 4 is good
        players[6] = True # Player 7 is good
        # Apply Round 3 information
        # Since Player 2 and Player 4 are good, Player 1 must be bad
        players[0] = False # Player 1 is bad
        # Apply Round 4 information
        # Since Player 2 and Player 7 are good , and we know Player 1 is bad , Player 3 must be good
        players[2] = True # Player 3 is good
        # This leaves Player 5 as the only possible bad player from this group
        players[4] = False # Player 5 is bad
        # We already have 2 bad players ( Player 1 and Player 5) , so Player 6 must be good
        players[5] = True # Player 6 is good
        # Print the final 4 players most likely to be good
        good_players = [ index + 1 for index,is_good in enumerate(players) if is_good ]
        print(good_players)"""
        print("Prompt: " + prompt + "\n")
        conversations = []
        conversations.append({"role": "user", "content": example_prompt})
        conversations.append({"role": "assistant", "content": example})
        conversations.append({"role": "user", "content": prompt})
        output = self.send_llama_message(conversations, temperature=0.01)
        # print("Output: " + output + "\n")
        # pattern = "Action Output:"
        # match = re.search(pattern, output)
        # if match is None:
        #     pattern = "(?<=Action Output:).*"
        #     match = re.search(pattern, output)
        extracted_code = ""
        if "Action Output:\n" in output:
            answer_index = output.index("Action Output:\n")
            extracted_code = output[answer_index + len("Action Output:\n"):]
        # print("Code: " + match.group().strip() if match else output)
        print("Extracted code: " + extracted_code + "\n")
        # result = self.execute_code_program(match.group().strip() if match else output)
        if int(self.current_quest_number) != 1:
            result = self.execute_code_program(extracted_code)

    def execute_code_program(self, program: str):
        try:
            good_players = []
            exec(program)
            self.teams = good_players
            return self.teams  # a list containing good players
        except Exception as e:
            print(f"Error executing code: {e}")
            # 5 regenerate attempts
            self.code_runtime_out -= 1
            if self.code_runtime_out == 0:
                pass
            else:
                self.generate_code_program(error_message=str(e))

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
        if len(self.memory["message"]) >= self.k:
            self.memory['name'].pop(0)
            self.memory['message'].pop(0)
            self.memory['informativeness'].pop(0)

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
    
    def retrival_memory(self, conversations: List[dict]):
        # freshness
        names = self.memory.get("name", [])[-self.k:]
        messages = self.memory.get("message", [])[-self.k:]
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


class CodeActAgentforWerewolf(CodeActAgent):

    def __init__(self, name: str, role: str, rule_role_prompt: str,
                 private_information: str, current_team_number: int,
                 code_generate_prompt: str, output_dir: str, total_player_number: int,
                 good_number: int, bad_number: int, k: int, informativeness_prompt: str,
                 generate_response_prompt: str, generate_leader_response_prompt: str,
                 informativeness_n: int, select_question_prompt: str, question_list: list,
                 ask_question_prompt: str, retrieval_model, generate_answer_prompt: str,
                 reflection_prompt: str, llama_model, llama_tokenizer,
                 use_summary: bool = False,
                 **kwargs):
        super().__init__(name=name, role=role, rule_role_prompt=rule_role_prompt,private_information=private_information,
                         current_team_number=current_team_number, code_generate_prompt=code_generate_prompt,
                         output_dir=output_dir, total_player_number=total_player_number, good_number=good_number,
                         bad_number=bad_number, k=k, informativeness_prompt=informativeness_prompt,
                         generate_response_prompt=generate_response_prompt, generate_leader_response_prompt=generate_leader_response_prompt,
                         informativeness_n=informativeness_n, select_question_prompt=select_question_prompt,
                         question_list=question_list, ask_question_prompt=ask_question_prompt,
                         retrieval_model=retrieval_model, generate_answer_prompt=generate_answer_prompt,
                         reflection_prompt=reflection_prompt, llama_model=llama_model, llama_tokenizer=llama_tokenizer,
                         use_summary=use_summary, **kwargs)
        self.public_information = ""
        self.current_day_number = 0  # day number
        self.informativeness_n = informativeness_n
        self.T = 3
        self.code_runtime_out = 5
        self.bad_player = None

    def step(self, message: str) -> str:
        phase = message.split("|")[0]
        self.phase = phase
        message = message.split("|")[1]
        conversations = [
            {"role": 'system', "content": self.rule_role_prompt}
        ]
        # if "The player who receives the" in message:
        #     pattern = r"The player who receives the (^.{3,20}$) card"
        #     role = re.search(pattern, message)
        #     self.role = role.group(1)
        #     print(self.role)
        if "is a werewolf" in message:
            pattern = r"(.+) is a werewolf"
            seen = re.search(pattern, message)
            if seen:
                self.private_information += seen.group(1) + "is a werewolf\n"
        elif "is not a werewolf" in message:
            pattern = r"(.+) is not a werewolf"
            seen = re.search(pattern, message)
            if seen:
                self.private_information += seen.group(1) + "is not a werewolf\n"
        elif "Werewolves, please open your eyes! I secretly tell you that" in message:
            pattern = r"Werewolves, please open your eyes! I secretly tell you that (.+) are all of the"
            seen = re.search(pattern, message)
            if seen:
                self.private_information += seen.group(1) + "is your werewolf teammate\n"
        elif "will be killed tonight. You have a bottle of antidote" in message:
            pattern = r"! (.+) will be killed tonight. You have a bottle of antidote"
            seen = re.search(pattern, message)
            if seen:
                self.private_information += "Night " + str(self.current_day_number) + ": " + seen.group(1) + "will be killed"
        # retrieval
        if self.memory.get("message"):
            if self.use_summary:
                r_t = self.summary_memory()
            else:
                r_t, conversations = self.retrival_memory(conversations)
        else:
            r_t = "None"
        if ("are asked to choose which of the players should be voted for eliminating based on the discussion"
                in message or "continue voting the players should be killed based on the discussion" in message):
            # print("Start Coding generating... \n")
            self.code_runtime_out = 5
            self.generate_code_program()
            prompt = self.generate_leader_response_prompt.format(
                self.phase, self.name, self.role, self.bad_player, message
            )
        else:
            prompt = self.generate_response_prompt.format(
                self.phase, self.name, self.role, message, r_t
            )
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
        # response = output
        self.update_memory("Host", message)
        self.update_memory(self.name, response)
        return response

    def receive(self, name: str, message: str) -> None:
        phase = message.split("|")[0]
        self.phase = phase
        message = message.split("|")[1]
        self.update_memory(name, message)
        # print("Receive Here!")
        if "died last night" in message:
            pattern = r"(.+) died last night"
            seen = re.search(pattern, message)
            if seen:
                self.public_information += "Day " + str(self.current_day_number) + ": " + seen.group(1) + " died last night\n"
                self.current_day_number += 1
        elif "It was a peaceful night and no one died" in message:
            self.public_information += "Day " + str(self.current_day_number) + " no one died last night\n"
            self.current_day_number += 1

    def generate_code_program(self, error_message: str = ""):
        print("Self.public_information: ")
        print(self.public_information)
        prompt = self.code_generate_prompt.format(
            self.total_player_number, self.good_number, self.bad_number, self.private_information,
            self.public_information
        )
        if error_message:
            prompt += ("\n\nThere was an error in last generation, please regenerate. Error message: " + error_message)
        example_prompt = """
        Now there are 7 players in the game, including 4 players in good side and 3 players in bad side.
        You now have private information: 
        Public information:In quest 1, players [2, 4, 7] are chosen as candidates. The quest result is succeeded.
        In quest 2, players [1, 2, 4] are chosen as candidates. The quest result is failed. 
        In quest 3, players [1, 2, 3, 7] are chosen as candidates. The quest result is failed.
        Please generate a Python program that selects 4 players most likely to be good person. Your answer should start with 'Action Output: '
        """
        example = """
        Action Output:
        players = [None] * 7 # Initialize 7 players list with None
        # Apply Round 2 information
        players[1] = True # Player 2 is good
        players[3] = True # Player 4 is good
        players[6] = True # Player 7 is good
        # Apply Round 3 information
        # Since Player 2 and Player 4 are good, Player 1 must be bad
        players[0] = False # Player 1 is bad
        # Apply Round 4 information
        # Since Player 2 and Player 7 are good , and we know Player 1 is bad , Player 3 must be good
        players[2] = True # Player 3 is good
        # This leaves Player 5 as the only possible bad player from this group
        players[4] = False # Player 5 is bad
        # We already have 2 bad players ( Player 1 and Player 5) , so Player 6 must be good
        players[5] = True # Player 6 is good
        # Print the final 4 players most likely to be good
        good_players = [ index + 1 for index,is_good in enumerate(players) if is_good ]
        print(good_players)"""
        print("Prompt: " + prompt + "\n")
        conversations = []
        conversations.append({"role": "user", "content": example_prompt})
        conversations.append({"role": "assistant", "content": example})
        conversations.append({"role": "user", "content": prompt})
        output = self.send_llama_message(conversations, temperature=0.01)
        # print("Output: " + output + "\n")
        # pattern = "Action Output:"
        # match = re.search(pattern, output)
        # if match is None:
        #     pattern = "(?<=Action Output:).*"
        #     match = re.search(pattern, output)
        extracted_code = ""
        if "Action Output:\n" in output:
            answer_index = output.index("Action Output:\n")
            extracted_code = output[answer_index + len("Action Output:\n"):]
        # print("Code: " + match.group().strip() if match else output)
        print("Extracted code: " + extracted_code + "\n")
        # result = self.execute_code_program(match.group().strip() if match else output)
        result = self.execute_code_program(extracted_code)

    def execute_code_program(self, program: str):
        try:
            output = []
            # sys.stdout = output
            # exec(program)
            # sys.stdout = sys.__stdout__
            # if output:
            #     self.bad_player = int(output[0])
            #     return self.bad_player
            pattern = r"bad_player = (\d)"
            player_index = re.search(pattern, program)
            player_index = int(player_index.group(1))
            if player_index > 0 and player_index < 8:
                self.bad_player = player_index
                return self.bad_player  # a integer
            else:
                print("No valid player number.")
                return None
        except Exception as e:
            print(f"Error executing code: {e}")
            # 5 regenerate attempts
            self.code_runtime_out -= 1
            if self.code_runtime_out == 0:
                pass
            else:
                self.generate_code_program(error_message=str(e))

    def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                      temperature: float = None) -> str:
        raise NotImplementedError("Interaction with LLM is not implemented in agent framework class.")


class CodeActAgentforAvalon(CodeActAgent):
    """
        just a copy
    """
