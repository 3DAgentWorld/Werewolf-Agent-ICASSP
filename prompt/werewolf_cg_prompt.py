#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: werewolf_cg_prompt.py 
# @date: 2024/3/19 16:02 
#
# describe:
#
rule_role_prompt = """You are playing a game called the Werewolf with some other players. This game is based on text conversations. Here are the game rules: 
Roles: The moderator is also host, he organised this game and you need to answer his instructions correctly. Don't talk with the moderator. There are five roles in the game, werewolf, villager, seer, guard and witch. There are two alternate phases in this game, daytime and dark. When it's dark: Your talking content with moderator is confidential. You needn't worry about other players and moderator knowing what you say and do. No need to worry about suspicions from others during the night. If you are werewolf, you can know what your teammates want to kill and you should vote one player to kill based on your analysis. Player who receives the most votes after all werewolves voting will be killed. No one will be killed if there is no consensus! If you are witch, you have a bottle of antidote that can save a player targeted by werewolves after dark, and a bottle of poison that can poison a player after dark. Both poison and antidote can be used only once. If you are seer, you can verify whether a player is a werewolf every night, which is a very important thing. If you are guard, you can protect a player every night to prevent the player from being killed by werewolves, but guard cannot resist the witch's poison and guard cannot protect the same player on two consecutive nights. Villagers can't do anything at night. During the daytime: you discuss with all players including your enemies. At the end of the discussion, players vote to eliminate one player they suspect of being a werewolf. The player with the most votes will be eliminated. The moderator will tell who is killed, otherwise there is no one killed. Note that villager, seer, guard and witch are all in villager side, they have the same objective. 

Objectives: If you are werewolf, your goal is to cooperate with other werewolves to kill all players who are not werewolves at last. If you are not werewolf, you need to kill all werewolves with your partner once you find out that certain players are suspicious to be werewolves. This could greatly improve your chances of winning, although it is somewhat risky.If one player is killed, he can't do anything anymore and will be out of the game. 

Tips: To complete the objective: During night, you should analyze and use your ability correctly. During daytime, you need to reason carefully about the roles of other players and be careful not to reveal your own role casually unless you're cheating other players. Only give the player's name when making a decision/voting, and don't generate other players' conversation.Reasoning based on facts you have observed and you cannot perceive information (such as acoustic info) other than text. You are {}, the {}. You're playing with 6 other players. Do not pretend you are other players or the moderator. Always end your response with '<EOS>'."""

select_question_prompt = """Now its the {}. Given the game rules and conversations above, assuming you are {}, the {}, and to complete the instructions of the moderator, you need to think about a few questions clearly first, so that you can make an accurate decision on the next step. Choose only five that you think are the most important in the current situation from the list of questions below: {} Please repeat the five important questions of your choice, separating them with ‘#’."""

ask_question_prompt = """Now its the {}. Given the game rules and conversations above, assuming you are {}, the {}, and to complete the instructions of the moderator, you need to think about a few questions clearly first, so that you can make an accurate decision on the next step. {} Do not answer these questions. In addition to the above questions, please make a bold guess, what else do you want to know about the current situation? Please ask two important questions in first person, separating them with ‘#’."""

generate_answer_prompt = """Now its the {}. Given the game rules and conversations above, assuming you are {}, the {}, for questions: {} There are {} possible answers for each question: {} Generate the correct answer based on the context. If there is not direct answer, you should think and generate the answer based on the context. No need to give options. The answer should in first person using no more than 2 sentences and without any analysis and item numbers."""

reflection_prompt = """Now its the {}. Assuming you are {}, the {}, what insights can you summarize with few sentences based on the above conversations and {} in heart for helping continue the talking and achieving your objective? For example: As the {}, I observed that... I think that... But I am... So..."""

extract_suggestion_prompt = """I retrieve some historical experience similar to current situation that I am facing. There is one bad experience: {} And there are also a set of experience that may consist of good ones: {} Please help me analyze the differences between these experiences and identify the good ones from the set of experiences. The difference is mainly about voting to kill someone or to pass, choosing to protect someone or to pass, using drugs or not. What does the experience set do but the bad experience does not do? Indicate in second person what is the best way for the player to do under such reflection. Clearly indicate whether to vote, protect or use drugs without any prerequisites. For example 1: The experience set involves choosing to protect someone, while the bad experience involves not protecting anyone and choosing to pass in contrast. The best way for you to do under such reflection is to choose someone to protect based on your analysis. For example 2: The bad experience choose to pass the voting, and all the experience in the experience set choose to pass as well. The best way for you to do under such reflection is to observe and analyse the identity of other players. No more than 1 sentence. If there is no obvious difference between them, only generate “No useful experience can be used.”"""

generate_response_prompt = """Now its the {}. Given the game rules and conversations above, assuming you are {}, the {}, think about what to say based on the game rules and context to response the Host's question: {}, especially the just now reflection {}. Besides, there maybe history experience you can refer to: {} Give your step-by-step thought process and your derived consise talking content at last. For example: My step-by-step thought process:... My concise talking content: ..."""

informativeness_prompt = """Score the sentence "{}" according to below rules.
If the sentence describes "the agent get its role", score the sentence 5.
If the sentence describes "someone was eliminated", score the sentence 4.
If the sentence describes "uncover or speculate the role", score the sentence 3.
If the sentence describes "the drugs has been used", score the sentence 2.
Others score the sentence 1.

Output format:
the sentence describes xxx, Therefore, the score is..."""

question_list = {
    "Werewolf": [
        "What is the current phase, daytime or night? what should I do at this phase according to the game rules?",
        "What is my player name and what is my role? What is my final objective in this game?",
        "Based on the current situation, what are the possible consequences if I reveal my role in the talking now?",
        "Is my role disclosed (do not consider moderator and myself)? Do I now disclose my role?",
        "Based on the conversation above, which players have clearly implied their roles?",
        "Based on the chat history, can you guess what some players' role might be?",
        "Which player was voted for killing by my teammate just now?",
        "Is the seer alive? Which player may be the seer that is most threatening to us?",
        "Which player is another werewolf in this game?"
    ],
    "Villager": [
        "What is the current phase, daytime or night? what should I do at this phase according to the game rules?",
        "What is my player name and what is my role? What is my final objective in this game?",
        "Based on the current situation, what are the possible consequences if I reveal my role in the talking now?",
        "Is my role disclosed (do not consider moderator and myself)? Do I now disclose my role?",
        "Based on the conversation above, which players have clearly implied their roles?",
        "Based on the chat history, can you guess what some players' role might be?",
        "Based on the conversation and my inference, who is most likely to be an alive werewolf?",
        "Which player made the statement claiming to be a seer? Can his words be trusted?",
        "Are there any clues or information I can refer to for special characters such as seer, witch and guard?"
    ],
    "Seer": [
        "What is the current phase, daytime or night? what should I do at this phase according to the game rules?",
        "What is my player name and what is my role? What is my final objective in this game?",
        "Based on the current situation, what are the possible consequences if I reveal my role in the talking now?",
        "Is my role disclosed (do not consider moderator and myself)? Do I now disclose my role?",
        "Based on the conversation above, which players have clearly implied their roles?",
        "Based on the chat history, can you guess what some players' role might be?",
        "Which suspicious player should I identify?",
        "Which player is a werewolf among the players I have identified? If so, how should I disclose this information?",
        "Should I disclose my role now?"
    ],
    "Guard": [
        "What is the current phase, daytime or night? what should I do at this phase according to the game rules?",
        "What is my player name and what is my role? What is my final objective in this game?",
        "Based on the current situation, what are the possible consequences if I reveal my role in the talking now?",
        "Is my role disclosed (do not consider moderator and myself)? Do I now disclose my role?",
        "Based on the conversation above, which players have clearly implied their roles?",
        "Based on the chat history, can you guess what some players' role might be?",
        "Based on the conversation and my inference, who is most likely to be an alive werewolf?",
        "Who is the possible werewolf aggressive towards?",
        "Is the seer still alive? If yes, who is the seer?"
    ],
    "Witch": [
        "What is the current phase, daytime or night? what should I do at this phase according to the game rules?",
        "What is my player name and what is my role? What is my final objective in this game?",
        "Based on the current situation, what are the possible consequences if I reveal my role in the talking now?",
        "Is my role disclosed (do not consider moderator and myself)? Do I now disclose my role?",
        "Based on the conversation above, which players have clearly implied their roles?",
        "Based on the chat history, can you guess what some players' role might be?",
        "Based on the conversation and my inference, who is most likely to be an alive werewolf?",
        "Should I poison him? Should I be using my antidote or poison at this point? If I use it now, I won't be able to use it later.",
        "Should I disclose my role now?"
    ]
}
