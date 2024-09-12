#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: run_dvm_werewolf.py 
# @date: 2024/9/12 14:00 
#

import os
import random
import argparse

from src.games.werewolf.werewolf import Werewolf
from src.agents.dvm_agent import DVMAgent
from src.utils import create_dir

api_key = "EMPTY"
base_url = "http://localhost:8000/v1"
roles = ["Werewolf", "Werewolf", "Villager", "Villager", "Seer", "Guard", "Witch"]
model_name = "ChatGLM3-6B"


def run_game(game_output_dir, camp, game_idx, wr_cons):
    create_dir(game_output_dir.format(game_idx))

    mode = "watch"
    language = "english"
    ai_model = model_name
    player_nums = 7
    player_mapping = {}
    random.shuffle(roles)
    if camp == "villager":
        camp_role = ["Villager", "Seer", "Guard", "Witch"]
    else:
        camp_role = ["Werewolf"]

    game = Werewolf(player_nums, language, mode, ai_model, game_output_dir.format(game_idx))

    player_args = []
    for i in range(player_nums):
        log_dir = f"{game_output_dir.format(game_idx)}/player {i + 1}"
        create_dir(log_dir)
        name = f"player {i + 1}"
        role = roles[i]
        player_mapping[name] = role
        player_args.append(
            (
                DVMAgent,
                {
                    "name": name,
                    "role": role,
                    "model": ai_model,
                    "api_key": api_key,
                    "base_url": base_url,
                    "output_dir": log_dir,
                    "win_rate_constraint": wr_cons,
                    "player_nums": player_nums,
                    "use_random": True
                }
            )
        )

    game.add_players(player_args)
    game.start()
    for player_i, agent_i in game.players.items():
        agent_i.reflection(
            player_mapping,
            file_name=f"{game_output_dir.format(game_idx)}/{player_mapping.get(player_i)}_reflection.json",
            winners=game.winners,
            duration=game.day_count
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_count", type=int, default=1)
    parser.add_argument("--camp", type=str, default="villager", choices=["villager", "werewolf"])
    parser.add_argument("--exp_name", type=str, default="dvm_test")
    parser.add_argument("--start_game_idx", type=int, default=0)
    parser.add_argument("--wr_cons", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="ChatGLM3-6B")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    return parser.parse_args()


def main():
    args = parse_args()
    global model_name, base_url, api_key
    model_name = args.model
    base_url = args.base_url
    api_key = args.api_key
    for game_round in range(args.start_game_idx, args.game_count):
        output_dir = f"playing_log/werewolf/dvm/{args.exp_name}-{args.camp}-wr{int(args.wr_cons * 100)}" + "-game_{}"
        run_game(output_dir, camp=args.camp, game_idx=game_round, wr_cons=args.wr_cons)
        print("game finish!!! game index {}".format(game_round))


if __name__ == '__main__':
    main()
    print("done!!!")
