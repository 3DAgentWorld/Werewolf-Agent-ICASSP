#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: run_werewolf_llama_cgagent.py 
# @date: 2024/3/19 20:30 
#
# describe:
#
import os
import random
import argparse

import openai
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel

from src.extractor.llm_extractor.chatgpt_extractor import ChatGPTBasedExtractor
from src.games.werewolf.werewolf import Werewolf
from src.agents import ChatGLM_CGAgent, ChatGLM_GraphCGAgent
from prompt.werewolf_cg_prompt import rule_role_prompt, select_question_prompt, ask_question_prompt, \
    generate_answer_prompt, reflection_prompt, extract_suggestion_prompt, generate_response_prompt, \
    informativeness_prompt, question_list
from prompt.werewolf_graph_cg_agent_prompt import graph_generate_prompt, reasoning_prompt
from src.games.werewolf.extract_demos import number_extract_prompt, player_extractor_demos, vote_extractor_demos, \
    bool_extract_prompt
from src.utils import create_dir, read_json

api_key = "sk-UAHO5kixcAwbXOoPBd1393C5B51f455bB2D5882f27998d53"
base_url = None
roles = ["Werewolf", "Werewolf", "Villager", "Villager", "Seer", "Guard", "Witch"]
# os.environ['ALL_PROXY'] = "http://127.0.0.1:7890"
# bert_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1", device="cuda")

bert_model = SentenceTransformer("/home/hdd/lanyihuai/ModelDownload/dataroot/model/sentence-transformers/multi-qa-mpnet-base-cos-v1", device="cuda")
# load chatglm
abs_model_dir = "/home/hdd/lanyihuai/ModelDownload/dataroot/model/THUDM/chatglm3-6b-32k"
tokenizer = AutoTokenizer.from_pretrained(abs_model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(abs_model_dir, trust_remote_code=True).half().cuda()
model = model.eval()


def run_game(game_output_dir: str, camp, game_idx):
    create_dir(game_output_dir.format(game_idx))

    mode = 'watch'
    language = 'english'
    ai_model = 'gpt-3.5-turbo-16k'
    player_nums = 7
    player_mapping = {}
    random.shuffle(roles)
    if camp == 'villager':
        camp_role = ["Villager", "Seer", "Guard", "Witch"]
    else:
        camp_role = ["Werewolf"]
    # camp_role = ["Werewolf", "Villager", "Seer", "Guard", "Witch"]

    game = Werewolf(player_nums, language, mode, ai_model, game_output_dir.format(game_idx))

    player_args = []
    for i in range(player_nums):
        log_dir = f"{game_output_dir.format(game_idx)}/player {i + 1}"
        create_dir(log_dir)
        if roles[i] in camp_role:
            if game_idx == 0:
                previous_exp_pool = []
            else:
                load_file = f"{game_output_dir.format(game_idx - 1)}/{roles[i]}_reflection.json"
                previous_exp_pool = read_json(load_file)
            name = f"player {i + 1}"
            role = roles[i]
            player_mapping[name] = role
            player_args.append(
                (
                    ChatGLM_GraphCGAgent, {"name": name, "role": role, "rule_role_prompt": rule_role_prompt,
                                           "select_question_prompt": select_question_prompt,
                                           "ask_question_prompt": ask_question_prompt,
                                           "generate_answer_prompt": generate_answer_prompt,
                                           "reflection_prompt": reflection_prompt,
                                           "extract_suggestion_prompt": extract_suggestion_prompt,
                                           "generate_response_prompt": generate_response_prompt,
                                           "informativeness_prompt": informativeness_prompt,
                                           "question_list": question_list.get(roles[i], []),
                                           "graph_generate_prompt": graph_generate_prompt,
                                           "reasoning_prompt": reasoning_prompt,
                                           "retrival_model": bert_model,
                                           "model": model, "freshness_k": 10, "informativeness_n": 10,
                                           "experience_window": 30,
                                           "temperature": 0.3, "api_key": "", "previous_exp_pool": previous_exp_pool,
                                           "output_dir": log_dir, "tokenizer": tokenizer}
                )
            )
        else:
            if game_idx == 0:
                previous_exp_pool = []
            else:
                load_file = f"{game_output_dir.format(game_idx - 1)}/{roles[i]}_reflection.json"
                previous_exp_pool = read_json(load_file)
            name = f"player {i + 1}"
            role = roles[i]
            player_mapping[name] = role
            player_args.append(
                (
                    ChatGLM_CGAgent, {"name": name, "role": role, "rule_role_prompt": rule_role_prompt,
                                      "select_question_prompt": select_question_prompt,
                                      "ask_question_prompt": ask_question_prompt,
                                      "generate_answer_prompt": generate_answer_prompt,
                                      "reflection_prompt": reflection_prompt,
                                      "extract_suggestion_prompt": extract_suggestion_prompt,
                                      "generate_response_prompt": generate_response_prompt,
                                      "informativeness_prompt": informativeness_prompt,
                                      "question_list": question_list.get(roles[i], []), "retrival_model": bert_model,
                                      "model": model, "freshness_k": 10, "informativeness_n": 10,
                                      "experience_window": 30,
                                      "temperature": 0.3, "api_key": "", "previous_exp_pool": previous_exp_pool,
                                      "output_dir": log_dir, "tokenizer": tokenizer}
                )
            )

    game.add_players(player_args)

    # extractors
    extractor_args = [(ChatGPTBasedExtractor,
                       {"extractor_name": "player extractor", "model_name": 'gpt-3.5-turbo-16k',
                        "extract_prompt": number_extract_prompt, "system_prompt": "You are a helpful assistant.",
                        "temperature": 0, "few_shot_demos": player_extractor_demos,
                        "output_dir": game_output_dir.format(game_idx)}),
                      (ChatGPTBasedExtractor,
                       {"extractor_name": "antidote extractor", "model_name": 'gpt-3.5-turbo-16k',
                        "extract_prompt": bool_extract_prompt, "system_prompt": "You are a helpful assistant.",
                        "temperature": 0, "few_shot_demos": vote_extractor_demos,
                        "output_dir": game_output_dir.format(game_idx)}),
                      (ChatGPTBasedExtractor,
                       {"extractor_name": "killing agreement extractor", "model_name": 'gpt-3.5-turbo-16k',
                        "extract_prompt": bool_extract_prompt, "system_prompt": "You are a helpful assistant.",
                        "temperature": 0, "few_shot_demos": vote_extractor_demos,
                        "output_dir": game_output_dir.format(game_idx)})
                      ]

    game.init_extractor(player_extractor=extractor_args[0], antidote_extractor=extractor_args[1],
                        killing_agreement_extractor=extractor_args[2])
    game.start()
    for player_i, agent_i in game.players.items():
        if isinstance(agent_i, (ChatGLM_CGAgent,ChatGLM_GraphCGAgent)):
            agent_i.reflection(
                player_mapping,
                file_name=f"{game_output_dir.format(game_idx)}/{player_mapping.get(player_i)}_reflection.json",
                winners=game.winners,
                duration=game.day_count
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game_count", type=int, default=10
    )
    parser.add_argument("--camp", type=str, default="villager", choices=["villager", "werewolf"])
    parser.add_argument("--exp_name", type=str, default="test_glm_graph_cgagent_vs_cgagent")
    parser.add_argument("--use_proxy", type=bool, default=False)
    parser.add_argument("--start_game_idx", type=int, default=0)
    parser.add_argument("--api_idx", type=int, default=0)
    parsed_args = parser.parse_args()
    return parsed_args


def main():
    args = parse_args()
    # if args.use_proxy:
    #     openai.proxy = "http://127.0.0.1:7890"
    #     os.environ["ALL_PROXY"] = "http://127.0.0.1:7890"
    apikeys = [
        'sk-UAHO5kixcAwbXOoPBd1393C5B51f455bB2D5882f27998d53',
        "sk-m1tpWLb9G4SpktRoA6450b5a318b434dBb51453c0e09B576"
    ]
    openai.api_key = api_key
    openai.api_key = apikeys[args.api_idx]
    openai.base_url = base_url
    for game_round in range(args.start_game_idx, args.game_count):
        output_dir = f"playing_log/werewolf/battle/{args.exp_name}-{args.camp}" + "-game_{}"
        #create_dir(output_dir)
        run_game(output_dir, camp=args.camp, game_idx=game_round)
        print("game finish!!! game index {}".format(game_round))


if __name__ == '__main__':
    main()
    print("done!!!")
    # remote 17522(4)
    # CUDA_VISIBLE_DEVICES=0 nohup python -u run_werewolf_chatglm_cgagent.py --camp villager --start_game_idx 1 --api_idx 0 1>v_graph_vs_cg,log&
    # CUDA_VISIBLE_DEVICES=2 nohup python -u run_werewolf_chatglm_cgagent.py --camp werewolf --start_game_idx 1 --api_idx 1 1>w_graph_vs_cg,log&