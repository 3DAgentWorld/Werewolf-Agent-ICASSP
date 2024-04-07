#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: werewolf_sapar_prompt.py 
# @date: 2024/3/27 15:36 
#
# describe:
#
system_prompt = \
    """You are playing a game called the Werewolf with some other players. This game is based on text conversations. Here are the game rules: 
Roles: The moderator is also host, he organised this game and you need to answer his instructions correctly. Don't talk with the moderator. There are five roles in the game, werewolf, villager, seer, guard and witch. There are two alternate phases in this game, daytime and dark. When it's dark: Your talking content with moderator is confidential. You needn't worry about other players and moderator knowing what you say and do. No need to worry about suspicions from others during the night. If you are werewolf, you can know what your teammates want to kill and you should vote one player to kill based on your analysis. Player who receives the most votes after all werewolves voting will be killed. No one will be killed if there is no consensus! If you are witch, you have a bottle of antidote that can save a player targeted by werewolves after dark, and a bottle of poison that can poison a player after dark. Both poison and antidote can be used only once. If you are seer, you can verify whether a player is a werewolf every night, which is a very important thing. If you are guard, you can protect a player every night to prevent the player from being killed by werewolves, but guard cannot resist the witch's poison and guard cannot protect the same player on two consecutive nights. Villagers can't do anything at night. During the daytime: you discuss with all players including your enemies. At the end of the discussion, players vote to eliminate one player they suspect of being a werewolf. The player with the most votes will be eliminated. The moderator will tell who is killed, otherwise there is no one killed. Note that villager, seer, guard and witch are all in villager side, they have the same objective. 

Objectives: If you are werewolf, your goal is to cooperate with other werewolves to kill all players who are not werewolves at last. If you are not werewolf, you need to kill all werewolves with your partner once you find out that certain players are suspicious to be werewolves. This could greatly improve your chances of winning, although it is somewhat risky.If one player is killed, he can't do anything anymore and will be out of the game. 

Tips: To complete the objective: During night, you should analyze and use your ability correctly. During daytime, you need to reason carefully about the roles of other players and be careful not to reveal your own role casually unless you're cheating other players. Only give the player's name when making a decision/voting, and don't generate other players' conversation.Reasoning based on facts you have observed and you cannot perceive information (such as acoustic info) other than text. You are {name}, the {role}. You're playing with 6 other players. Do not pretend you are other players or the moderator.

You are {name}, the {role}. Your playing style is that {strategy}. There are experience of previous games provided: <experience>
suggestions from previous games: {suggestion}
strategies of other roles from previous games: {other_strategy}
</experience>."""

summary_prompt = \
    """Within the context of the Werewolf game, please assist {name} in summarizing the conversations known to them from the current phase. These conversations are structured in JSON format, with "message" signifying the content of the conversation, "name" identifying the speaker, and "message_type" indicating the type of message relevant to {name}. Specifically, "public" implies that all players have access to the message, while "private" implies that only {name} has access to it.
As this turn progresses, the summary should includes who claimed his role, which players have been killed or eliminated out of the game, what the voting status of the players is towards the werewolf and so on.

Conversations: {conversation}

Use the following format:
Summary: <summary>"""

analysis_prompt = \
    """You currently assume the {name} within an Werewolf game, and the game has progressed to the {phase}. Your task is to analyze roles and strategies of other players according to their behaviors. The behaviors are summarized in paragraphs. The analysis should be no more than 100 words.
the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
</information>

the summary is <summary>{summary}</summary>"""

# analysis_teammate = \
#     """You currently assume the {name} within a Werewolf game, and the game has progressed to the {phase}. Your task is to analyze roles and strategies of the players who might be your teammates according to their behaviors. The behaviors are summarized in paragraphs. The analysis should be no more than 100 words.
# the information of yourself is <information>
# your name is <name>{name}</name>
# your role is <role>{role}</role>
# </information>
#
# the summary is <summary>{summary}</summary>"""
# analysis_enemy = \
#     """You currently assume the {name} within an Werewolf game, and the game has progressed to the {phase}. Your task is to analyze roles and strategies of the players who might be your enemies according to their behaviors. The behaviors are summarized in paragraphs. The analysis should be no more than 100 words.
# the information of yourself is <information>
# your name is <name>{name}</name>
# your role is <role>{role}</role>
# </information>
#
# the summary is <summary>{summary}</summary>"""

plan_prompt = \
    """You currently assume the {name} within an Werewolf game, and the game has progressed to the {phase}. Your task is to devise a playing plan that remains in harmony with your game goal and existing strategy, while also incorporating insights from your previous plan and current environment state.

the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
the role introduction is <introduction>{introduction}</introduction>
your game goal is <goal>{goal}</goal>
your playing strategy <strategy>{strategy}</strategy>
your previous plan <previous plan>
{previous_plan}
</previous plan>
</information>
the environment state is <environment>
the summary of previous turns <summary>{summary}</summary>
the analysis about other players is <analysis>{analysis}</analysis>
</environment>

the output format is <output>
my plan is <plan>
{plan}
</plan>
</output> 

your plans for each turn should be described with no more than one sentence. """

action_prompt = \
    """You currently assume the {name} within an Werewolf game, and the game has progressed to the {phase}. Your objective is to make decisions based on your role, your game goal and the current game state. There are two types of actions you can take: choosing player, selection (yes or no). You should decide your action according to Host's question.

the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
the role introduction is <introduction>{introduction}</introduction>
your game goal is <goal>{goal}</goal>
your playing strategy <strategy>{strategy}</strategy>
your candidate actions <candidate actions>{candidate_actions}</candidate actions>
</information>
the environment state is <environment>
the summary of previous turns <summary>{summary}</summary>
the analysis about other players is <analysis>{analysis}</analysis>
your current playing plan is <plan>{plan}</plan>
the Host's question is <question>{question}</question>
</environment>

the output format is <output>
<actions>...</actions>
</output>

here are examples of the output <example>
Example 1:
the output format is <output>
First, I'll decide on the <action type>choosing players</action type> to choose based on my role and the current game state.
Here are my actions based on the chosen action type:
<actions>choose player x</actions>
</output>

Example 2:
First, I'll decide on the <action type>selection</action type> to choose based on my role and the current game state.
Here are my actions based on the chosen action type:
the output format is <output>
<actions>yes</actions>
</output>

Example 3:
First, I'll decide on the <action type>selection</action type> to choose based on my role and the current game state.
Here is my action based on the chosen action type:
the output format is <output>
<actions>no</actions>
</output>
</example>"""

response_prompt = \
    """You currently assume the {name} within an Werewolf game, and the game has progressed to the {phase}. Your task is to provide detailed response to question of Host, in accordance with the provided actions. Your response should be no more than 100 words.

the information of yourself is <information>
your name is <name>{name}</name>
your role is <role>{role}</role>
the role introduction is <introduction>{introduction}</introduction>
your playing strategy <strategy>{strategy}</strategy>
</information>
the environment state is <environment>
the summary of previous turns <summary> {summary} </summary>
your current playing plan is <plan> {plan} </plan>
the Host's question is <question> {question} </question>
current actions <actions>{actions}</actions>
</environment>

the output format is <output>
my response is <response>...</response>
</output> """
# response_prompt_without_action = \
#     """You currently assume the {name} within an Avalon game, and the game has progressed to the {phase}. Your task is to provide detailed response to question of Host, in accordance with the environment state. Your response should be no more than 100 words.
#
# the information of yourself is <information>
# your name is <name>{name}</name>
# your role is <role>{role}</role>
# the role introduction is <introduction>{introduction}</introduction>
# your playing strategy <strategy>{strategy}</strategy>
# </information>
# the environment state is <environment>
# the summary of previous turns <summary> {summary} </summary>
# your current playing plan is <plan> {plan} </plan>
# the Host's question is <question> {question} </question>
# </environment>
#
# the output format is <output>
# my response is <response>...</response>
# </output> """


suggestion_prompt = \
    """Your task is to provide 3 suggestions for {name}'s playing strategy of the role {role} in Werewolf games, according to the game log. The game log includes the summaries of different turns of a round game.

The roles of the players:
{roles}

The summaries of a round game:
{summaries}

{name}'s game goal:
{goal}

{name}'s playing strategy of role {role}:
{strategy}

Based on previous suggestions:
{previous_suggestions}

Give your suggestions, No more than two sentences per suggestion and the suggestions should be general for future games (This implies that you should avoid referencing player x directly and instead use the respective role names when making your suggestion.) and effectively help him achieve his game goal in future games."""


strategy_prompt = \
    """Your task is to help {name} analyze the strategies of other players in a Werewolf game, according to the game log and game ending. The game log and game ending are summarized in paragraphs.
The roles of the players:
{roles}

The summaries of turns of the game:
{summaries}

previous strategies of other roles:
{strategies}

Your analysis should be no more than 100 words and the analysis should be general for future games (This implies that you should avoid referencing player x directly and instead use the respective role names when giving your analysis). And analyze together with previous strategies.

For example:
The strategy of Villager is that ... The strategy of Werewolf is that... The strategy of ... is ..."""

update_prompt = \
    """Your task is to help {name} improve his playing strategy of the role {role} a Werewolf game with suggestions.

{name}'s strategy:
{strategy}

suggestions:
{suggestions}

Please improve the strategy while retaining the advantages of the original strategy for him and the strategy should be no more than 2 sentences. Describe the strategy you provide using continuous sentences rather than bullet points or numbering."""

candidate_actions = [
    "choose player 1", "choose player 2", "choose player 3", "choose player 4", "choose player 5",
    "choose player 6", "choose player 7"
    "yes", "no"
]

init_strategies = {
    "Werewolf": "None",
    "Villager": "None",
    "Seer": "None",
    "Guard": "None",
    "Witch": "None"
}
role_introduction = {
    "Werewolf": "You belong to the Werewolf camp. Each night, you and your teammates can collectively choose one player to eliminate.",
    "Villager": "You belong to the Villager camp. You don't possess any special abilities, but your keen observation and judgment are crucial in identifying the Werewolves.",
    "Seer": "You are the Seer. Each night, you can choose one player to secretly investigate. You will learn whether that player is a Werewolf or not.",
    "Guard": "You are the Guard. Each night, you can choose one player to protect. If the Werewolves target the player you protect, they will be saved from elimination.",
    "Witch": "You are the Witch. You have poison and antidote. Once per game, you can use the antidote to save a player who has been attacked by the Werewolves. Additionally, you can use the poison to eliminate one player during the night."
}

role_target = {
    "Werewolf": "Win the game by kill all villagers or all other roles.",
    "Villager": "Win the game by eliminating all the werewolf.",
    "Seer": "Help villager camp win the game by eliminating all the werewolf.",
    "Guard": "Help villager camp win the game by eliminating all the werewolf.",
    "Witch": "Help villager camp win the game by eliminating all the werewolf."
}