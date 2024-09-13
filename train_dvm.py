#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: train_dvm.py 
# @date: 2024/9/13 10:00 
#

import os
import argparse
import random

import torch
import yaml

from src.models.predictor import Predictor
from src.models.decider import Decider
from src.training.ppo_trainer import PPOTrainer
from src.training.decision_chain import DecisionChainReward
from src.utils import read_json, write_json


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collect_decision_chains(data_dir):
    db = DecisionChainReward()
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                data = read_json(os.path.join(root, file))
                chain = tuple(data.get("chain", []))
                won = 1.0 if data.get("won") else 0.0
                db.register_chain(chain, won)
    return db


def train_decider(config, dc_db):
    decider = Decider(
        config["decider"]["state_dim"],
        config["decider"]["pred_dim"],
        config["decider"]["action_dim"],
        config["decider"]["hidden_dim"]
    )
    trainer = PPOTrainer(
        decider,
        lr=config["training"]["lr"],
        gamma=config["training"]["gamma"],
        lam=config["training"]["lam"],
        clip_eps=config["training"]["clip_eps"],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"]
    )
    return decider, trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dvm_chatglm3-6b.yaml")
    parser.add_argument("--data_dir", type=str, default="playing_log/werewolf/dvm")
    parser.add_argument("--output_dir", type=str, default="checkpoints/dvm")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    dc_db = collect_decision_chains(args.data_dir)
    decider, trainer = train_decider(config, dc_db)
    torch.save(decider.state_dict(), os.path.join(args.output_dir, "decider.pt"))
    write_json({"chain_count": len(dc_db.chain_db)}, os.path.join(args.output_dir, "info.json"))
    print("training done")


if __name__ == '__main__':
    main()
