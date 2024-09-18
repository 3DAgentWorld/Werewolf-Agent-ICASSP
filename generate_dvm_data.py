#!/usr/bin/env python
# encoding: utf-8
# @author: yihuai lan
# @fileName: generate_dvm_data.py
# @date: 2024/9/14 10:00
#
# Generate self-play data for DVM training.
# 1. Runs multiple 9-player Werewolf games with DVM agents.
# 2. Each agent dumps its internal decision vectors (state, prediction, action, mask).
# 3. After each game, an LLM parses the per-player response logs into a
#    supervised-finetuning (SFT) dataset in standard chat format.
#
# Usage:
#   python generate_dvm_data.py --model /path/to/ChatGLM3-6B --game_count 10

import os
import re
import json
import argparse
import random

import openai

from src.games.werewolf.werewolf import Werewolf
from src.agents.dvm_agent import DVMAgent
from src.utils import create_dir

api_key = "EMPTY"
base_url = "http://localhost:8000/v1"
model_name = "ChatGLM3-6B"
roles = ["Werewolf", "Werewolf", "Werewolf",
         "Villager", "Villager", "Villager",
         "Seer", "Witch", "Hunter"]


def parse_response_log(text):
    """Parse response.txt into (phase, instruction, action, response) tuples."""
    examples = []
    for block in text.split("---\n"):
        block = block.strip()
        if not block:
            continue
        phase_match = re.search(r"phase:(.*?)\ninstruction:", block, re.DOTALL)
        instr_match = re.search(r"instruction:(.*?)\naction:", block, re.DOTALL)
        action_match = re.search(r"action:(.*?)\noutput:", block, re.DOTALL)
        output_match = re.search(r"output:(.*?)$", block, re.DOTALL)
        if phase_match and instr_match and action_match and output_match:
            examples.append({
                "phase": phase_match.group(1).strip(),
                "instruction": instr_match.group(1).strip(),
                "action": action_match.group(1).strip(),
                "response": output_match.group(1).strip(),
            })
    return examples


def build_system_prompt(role):
    prompts = {
        "Werewolf": "You are playing a social deduction game called Werewolf. You are a Werewolf. Your goal is to eliminate all villagers and special roles without revealing your identity. Be persuasive but cautious.",
        "Villager": "You are playing a social deduction game called Werewolf. You are a Villager. You have no special abilities. Your goal is to identify and vote out all Werewolves through discussion and observation.",
        "Seer": "You are playing a social deduction game called Werewolf. You are the Seer. Each night you can check one player's identity. Use your information carefully without exposing yourself too early.",
        "Witch": "You are playing a social deduction game called Werewolf. You are the Witch. You have one antidote and one poison. Use them wisely to protect the good camp and eliminate Werewolves.",
        "Hunter": "You are playing a social deduction game called Werewolf. You are the Hunter. When eliminated, you can shoot one player. Try to take a Werewolf with you.",
    }
    return prompts.get(role, prompts["Villager"])


ACTIVE_KEYWORDS = [
    "vote", "kill", "agree", "disagree", "antidote", "poison",
    "verify", "shoot", "talk", "discuss", "statement", "eliminated"
]


def is_active_turn(example):
    lower = example["instruction"].lower()
    return any(kw in lower for kw in ACTIVE_KEYWORDS)


def llm_parse_action(model, role, instruction, action, response):
    """Use an LLM to normalize the chosen action for a single turn."""
    system_prompt = (
        "You are a data parser for a Werewolf game. Given a player role, host instruction, "
        "and player response, output a single JSON object with one field:\n"
        "  - action: the normalized chosen action.\n"
        "Rules:\n"
        "  - For votes/kills/poison/verify/shoot, return a player name like 'player 3'.\n"
        "  - For antidote/save questions, return 'yes' or 'no'.\n"
        "  - For werewolf agreement questions, return 'agree' or 'disagree'.\n"
        "  - If the response is already a valid action, return it unchanged.\n"
        "Output only valid JSON, no markdown, no explanation."
    )
    user_prompt = (
        f"Role: {role}\n"
        f"Host instruction: {instruction}\n"
        f"Player response: {response}\n"
        f"Suggested action label: {action}\n\n"
        f"Return JSON only."
    )

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    try:
        response_obj = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        out = response_obj.choices[0].message.content
    except Exception as e:
        print(f"[LLM parse action failed for {role}] {e}")
        return None
    finally:
        client.close()

    out = out.strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:json)?", "", out)
        out = re.sub(r"```$", "", out)
        out = out.strip()
    try:
        parsed = json.loads(out)
        if isinstance(parsed, dict) and "action" in parsed:
            return parsed["action"]
    except json.JSONDecodeError:
        pass
    return None


def llm_parse_player_log(model, player, role, response_text, outcome, winner):
    """Use an LLM to convert a player's response log into SFT training examples."""
    examples = [ex for ex in parse_response_log(response_text) if is_active_turn(ex)]
    if not examples:
        return []

    parsed_examples = []
    for ex in examples:
        normalized_action = llm_parse_action(
            model=model,
            role=role,
            instruction=ex["instruction"],
            action=ex["action"],
            response=ex["response"],
        )
        if normalized_action is None:
            normalized_action = ex["action"]

        messages = [
            {"role": "system", "content": build_system_prompt(role)},
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["response"]},
        ]
        parsed_examples.append({
            "messages": messages,
            "role": role,
            "phase": ex["phase"],
            "action": normalized_action,
            "outcome": outcome,
            "winner": winner,
        })

    return parsed_examples


def merge_step_data_with_outcome(game_dir):
    """Add final outcome to each dumped step record and write a merged file."""
    merged_path = os.path.join(game_dir, "step_data.jsonl")
    # collect outcomes from reflection files
    outcomes = {}
    for fname in os.listdir(game_dir):
        if fname.endswith("_reflection.json"):
            path = os.path.join(game_dir, fname)
            try:
                data = json.load(open(path, encoding="utf-8"))
                # The reflection file name is <Role>_reflection.json; we do not know
                # which player it belongs to, but step_data records already contain player.
                outcomes[data.get("role", "Unknown")] = data.get("won", False)
            except Exception:
                continue

    written = 0
    with open(merged_path, "w", encoding="utf-8") as out_f:
        for player_dir_name in os.listdir(game_dir):
            player_dir = os.path.join(game_dir, player_dir_name)
            step_file = os.path.join(player_dir, "step_data.jsonl")
            if not os.path.isfile(step_file):
                continue
            for line in open(step_file, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record["won"] = outcomes.get(record.get("role"), False)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    return written


def run_game(game_output_dir, game_idx, wr_cons, dump_steps=True, parse_with_llm=True):
    create_dir(game_output_dir)

    mode = "watch"
    language = "english"
    ai_model = model_name
    player_nums = 9
    player_mapping = {}
    random.shuffle(roles)

    game = Werewolf(player_nums, language, mode, ai_model, game_output_dir)

    player_args = []
    for i in range(player_nums):
        log_dir = f"{game_output_dir}/player {i + 1}"
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
                    "use_random": True,
                    "dump_steps": dump_steps,
                }
            )
        )

    game.add_players(player_args)
    game.start()
    for player_i, agent_i in game.players.items():
        agent_i.reflection(
            player_mapping,
            file_name=f"{game_output_dir}/{player_mapping.get(player_i)}_reflection.json",
            winners=game.winners,
            duration=game.day_count
        )

    # Merge step-level vectors and add outcome labels
    step_count = merge_step_data_with_outcome(game_output_dir)
    print(f"[game {game_idx}] saved {step_count} step records")

    # Use LLM to parse response logs into SFT format
    if parse_with_llm:
        sft_path = os.path.join(game_output_dir, "sft_data.jsonl")
        winner = game.winners[0] if game.winners else "None"
        with open(sft_path, "w", encoding="utf-8") as out_f:
            for player_i, role in player_mapping.items():
                response_file = os.path.join(game_output_dir, player_i, "response.txt")
                if not os.path.exists(response_file):
                    continue
                response_text = open(response_file, encoding="utf-8").read()
                reflection_file = os.path.join(game_output_dir, f"{role}_reflection.json")
                won = False
                if os.path.exists(reflection_file):
                    try:
                        won = json.load(open(reflection_file, encoding="utf-8")).get("won", False)
                    except Exception:
                        pass
                outcome = "win" if won else "loss"
                examples = llm_parse_player_log(
                    model=ai_model,
                    player=player_i,
                    role=role,
                    response_text=response_text,
                    outcome=outcome,
                    winner=winner,
                )
                for ex in examples:
                    out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[game {game_idx}] LLM parsing done -> {sft_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_count", type=int, default=10)
    parser.add_argument("--wr_cons", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="ChatGLM3-6B")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--output_dir", type=str, default="data/dvm_selfplay")
    parser.add_argument("--no_llm_parse", action="store_true",
                        help="Skip the LLM parsing step; only generate raw game logs and step vectors.")
    parser.add_argument("--start_game_idx", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    global model_name, base_url, api_key
    model_name = args.model
    base_url = args.base_url
    api_key = args.api_key

    create_dir(args.output_dir)
    parse_with_llm = not args.no_llm_parse

    for game_idx in range(args.start_game_idx, args.game_count):
        game_dir = os.path.join(args.output_dir, f"game_{game_idx}")
        print(f"\n=== Generating game {game_idx} ===")
        run_game(
            game_dir,
            game_idx=game_idx,
            wr_cons=args.wr_cons,
            dump_steps=True,
            parse_with_llm=parse_with_llm,
        )

    # Aggregate step vectors and SFT data across games
    aggregate_step = os.path.join(args.output_dir, "step_data.jsonl")
    aggregate_sft = os.path.join(args.output_dir, "sft_data.jsonl")
    with open(aggregate_step, "w", encoding="utf-8") as out_step:
        for game_idx in range(args.start_game_idx, args.game_count):
            game_dir = os.path.join(args.output_dir, f"game_{game_idx}")
            src = os.path.join(game_dir, "step_data.jsonl")
            if os.path.exists(src):
                with open(src, encoding="utf-8") as f:
                    out_step.write(f.read())
    if parse_with_llm:
        with open(aggregate_sft, "w", encoding="utf-8") as out_sft:
            for game_idx in range(args.start_game_idx, args.game_count):
                game_dir = os.path.join(args.output_dir, f"game_{game_idx}")
                src = os.path.join(game_dir, "sft_data.jsonl")
                if os.path.exists(src):
                    with open(src, encoding="utf-8") as f:
                        out_sft.write(f.read())

    print(f"\nAll done. Aggregated data saved to:\n  {aggregate_step}")
    if parse_with_llm:
        print(f"  {aggregate_sft}")


if __name__ == "__main__":
    main()
