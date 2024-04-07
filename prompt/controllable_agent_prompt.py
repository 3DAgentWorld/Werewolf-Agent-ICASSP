#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: controllable_agent_prompt.py 
# @date: 2024/3/4 14:53 
#
# describe:
#

role_statement_extraction = """STATEMENT: {statement}

Please extract required information below from {name}'s statement:
Suppositions of the identifications and corresponding reason.
His own identification. 

list the extractions from 1 to n.
If there are none, output 'None'."""


suggestion_statement_extraction = """STATEMENT: {statement}
Please extract required information below from {name}'s statement:
Suggestions he made.

list the extractions from 1 to n.
If there are none, output 'None'."""


fact_statement_extraction = """INSTRUCTION: {instruction}

Please extract required information below from moderator's instruction:
Facts provided by the moderator.

all the information in instruction is true.
list the extractions from 1 to n.
If there are none, output 'None'."""

believing_prompt = """You currently assume the {name} within an Avalon game, your role is {role} and the game has progressed to the {phase}. You are provided with some factual descriptions, as well as statements from players (which may not necessarily be factual). Based on the factual descriptions, you can choose to believe or not believe the players' statements.

FACTS: {facts}
STATEMENT: {statement}

If you believe, please answer 'yes'.
If you do not believe, please answer 'no'. """

confrontation_prompt = """You currently assume the {name} within an Avalon game, your role is {role} and the game has progressed to the {phase}. You are provided with some statements of other players. These statements may not be trustworthy. You need to choose which statements to refute.

STATEMENTS: {statements}

You only need to answer the numbers of the statements you want to refute.
If you choose not to refute anything, then answer 'none'. """

teamwork_prompt = """You currently assume the {name} within an Avalon game, your role is {role} and the game has progressed to the {phase}. You are provided with some statements of other players. These statements may be trustworthy. You can choose to further support these statements.

STATEMENTS: {statements}

You only need to answer the numbers of the statements you want to support.
If you choose not to support anything, then answer 'none'. """

share_prompt = """You currently assume the {name} within an Avalon game, and the game has progressed to the {phase}. You are provided with some information that other players may not know. You can choose to share these information with other players.

INFORMATION: {information}

You only need to answer the numbers of the information you want to share.
If you choose not to share anything, then answer 'none'. 
"""

action_prompt = """You currently assume the {name} within an Avalon game, your role is {role} and the game has progressed to the {phase}. You are provided with some statements of other players, some information you already know and an instruction from the host. Your objective is to make decisions required by the instruction based on these statements and information. You are only allowed choose actions from the candidate actions.

STATEMENTS: {statements}
INFORMATION: {information}
CANDIDATE ACTIONS: {candidate_actions}
INSTRUCTION: {instruction}"""

controllable_action_prompt = """You currently assume the {name} within an Avalon game, your role is {role} and the game has progressed to the {phase}. You are provided with some statements of other players, some information you already know and an instruction from the host. Your objective is to make decisions required by the instruction based on these statements and information. You are only allowed choose actions from the candidate actions.

STATEMENTS: {statements}
INFORMATION: {information}
CANDIDATE ACTIONS: {candidate_actions}
INSTRUCTION: {instruction}

{controllable}"""

response_prompt = """You currently assume the {name} within an Avalon game, your role is {role} and the game has progressed to the {phase}. You are provided with an instruction from the host and a decision about the instruction. You need to provide detailed response to question of host, in accordance with the provided decision.

INSTRUCTION: {instruction}
DECISION: {decision}

Additionally, if the instruction allows you response more information. You need to indicate your stance on the following statements. Your stance and response should be integrated into a naturally expressed sentence. If the instruction does not allow you response other information, ignore the following statements.
REFUTE: {refute}
SUPPORT: {support}
SHARE: {share}
"""