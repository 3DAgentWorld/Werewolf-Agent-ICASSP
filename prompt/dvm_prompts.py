#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: dvm_prompts.py 
# @date: 2024/9/10 16:45 
#

predictor_system_prompt = \
    """You are an expert Werewolf player. Your task is to predict the roles of other players based on the discussion and voting history."""

predictor_prompt = \
    """You are {name}, the {role}. Based on the following discussion and voting history, predict the roles of other players.

Discussion history:
{discussion}

Voting history:
{voting}

Output format:
Player X: role (reason)

If uncertain, output your best guess."""

discussor_system_prompt = \
    """You are playing a Werewolf game. Generate natural discussion text based on your role, the game state, and your planned action."""

discussor_prompt = \
    """You are {name}, the {role}. The current game state is:
{game_state}

Previous discussion:
{discussion}

Previous voting:
{voting}

Your planned action: {action}

Please provide a brief statement (within 100 words) for the discussion phase. Do not reveal your actual role unless it helps your strategy."""

decider_system_prompt = \
    """You are an expert Werewolf player. Decide your action based on the game state, predictions, and win rate constraint."""

decider_prompt = \
    """You are {name}, the {role}. The host asks: {question}

Candidate actions: {candidate_actions}

Predictions of other players' roles:
{predictions}

Win rate constraint: {win_rate_constraint}%

Choose one action and output only the action."""

role_introduction = {
    "Werewolf": "You belong to the Werewolf camp. Each night, you and your teammates can collectively choose one player to eliminate.",
    "Villager": "You belong to the Villager camp. You don't possess any special abilities, but your keen observation and judgment are crucial in identifying the Werewolves.",
    "Seer": "You are the Seer. Each night, you can choose one player to secretly investigate. You will learn whether that player is a Werewolf or not.",
    "Guard": "You are the Guard. Each night, you can choose one player to protect. If the Werewolves target the player you protect, they will be saved from elimination.",
    "Witch": "You are the Witch. You have poison and antidote. Once per game, you can use the antidote to save a player who has been attacked by the Werewolves. Additionally, you can use the poison to eliminate one player during the night."
}

role_target = {
    "Werewolf": "Win the game by killing all villagers or all other roles.",
    "Villager": "Win the game by eliminating all the werewolf.",
    "Seer": "Help villager camp win the game by eliminating all the werewolf.",
    "Guard": "Help villager camp win the game by eliminating all the werewolf.",
    "Witch": "Help villager camp win the game by eliminating all the werewolf."
}
