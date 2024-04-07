#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: extract_demos.py 
# @date: 2024/3/19 16:23 
#
# describe:
#
number_extract_prompt = \
    """Please extract the answer to the question from the answer.\n{}
"""
bool_extract_prompt = \
    """Complete the following instructions based on the questions and answers. 

Response true, if the answer expresses:
1.obvious agreement to something
2.obvious decision to do something
3.obvious yes to something

Response false, if the answer expresses:
1.obvious disagreement to something
2.obvious decision not to do something
3.obvious no to something

Response none, if the answer is not related to the question or didn't express a obvious attitude to something. 

{}
"""

player_extractor_demos = [
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question：You guard, player 5, please open your eyes! Now tell me who you protect tonight? You only choose one from the following living options please: ['player 1', 'player 2', 'player 3', 'player 4', 'player 5', 'player 6', 'player 7', 'pass'].
Answer：As the Guard, I have observed the conversation and considered the implications of each player's implied roles. Based on the chat history, Player 4's suspicious behavior suggests they might be a werewolf. However, I need to be cautious about revealing my role. Considering the objective of protecting the villagers, I choose to protect Player 4 tonight.
    """
    },
    {
        "role": "assistant",
        "content": "4"
    },
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: The next player 3, you, continue voting the players should be killed based on the discussion? Only choose one from the following living options please: ['player 1', 'player 3', 'player 4', 'player 5', 'player 6', 'player 7']
Answer: "Based on the current situation and the lack of concrete evidence, I believe it would be premature to single out a specific player for elimination at this point. We should engage in discussions and gather more information before making a decision."
    """
    },
    {
        "role": "assistant",
        "content": "none"
    },
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: Please finally confirm the two players who went on the quest.
Answer: According to public information, player 1 claims that player 4 is a member of the red camp. Player 6 knows that player 2 and player 3 are both good people and suggested that the candidates for the quest are player 2 and player 3. Player 6 has confirmed that player 2 and player 3 are 3 is a member of the blue camp, and player 3 knows that he is Merlin. Based on this information, I think the players who can finally confirm the quest are player 2 and player 3.
    """
    },
    {
        "role": "assistant",
        "content": "2、3"
    },
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: Please start discussing the candidates for the round 5 quest (please explain the reasons for your choice appropriately). The number of players in this quest is 3, and the order of each player's speech is player 6、player 1、player 2、player 3、player 4、player 5. player 6 has the power to decide which players will ultimately perform the quest. Note: 1. Your comments will be public to other players. 2. Number of players on this questplayer 5 start to speak please.
Answer: Considering the previous rounds and the behavior of the players, I propose that player 1, player 3, and myself should be the candidates for this quest. We have consistently made successful quests in the past rounds, and I believe we can be relied upon to complete this quest successfully as well.
    """
    },
    {
        "role": "assistant",
        "content": "1、3、5"
    },
    {
        "role": "user",
        "content": """Please extract the answer to the question from the answer.
Question: Please start discussing the candidates for the round 1 quest (please explain the reasons for your choice appropriately). The number of players in this quest is 2, and the order of each player's speech is player 3、player 4、player 5、player 6、player 1、player 2. player 3 has the power to decide which players will ultimately perform the quest. Note: 1. Your comments will be public to other players. 2. Number of players on this quest.player 3 start to speak please.
Answer: Based on the information we have so far, I think player 4 and player 1 would be good candidates for the quest. Player 4 had a very convincing argument during the previous discussion phase and seemed committed to the blue camp's success. Player 1 is someone we haven't heard much from yet, so it's difficult to judge their intentions. This combination of players could increase our chances of success. However, I'm open to other suggestions and would like to hear what others think.
    """
    },
    {
        "role": "assistant",
        "content": "4、1"
    }
]
vote_extractor_demos = [
    {
        "role": "user",
        "content": """Complete the following instructions based on the questions and answers. 

Response true, if the answer expresses:
1.obvious agreement to something
2.obvious decision to do something
3.obvious expression of making quest successful
4.obvious yes to something

Response false, if the answer expresses:
1.obvious disagreement to something
2.obvious decision not to do something
3.obvious expression of making quest failed
4.obvious no to something

Response none, if the answer is not related to the question or didn't express a obvious attitude to something. 

Question: The quest leader decides that player 1 and player 4 will go on a quest. You can choose to agree or disagree. If everyone agrees, the quest will succeed. As long as there is one vote against, the quest will fail. Please choose to agree or disagree.
Answer: According to player 6’s public information, he thinks player 1 and player 5 are both good people. According to player 3's public information, he agreed not to select player 4 for the quest. According to player 5's private information, he knows that he is Merlin, and player 4 is considered by player 1 to be a member of the red camp. Considering this information, I agree with player 6's suggestion that the candidates for the quest can be player 1 and player 5. So, I choose to agree.
"""
    },
    {
        "role": "assistant",
        "content": "true"
    },
{
        "role": "user",
        "content": """Complete the following instructions based on the questions and answers. 

Response true, if the answer expresses:
1.obvious agreement to something
2.obvious decision to do something
3.obvious expression of making quest successful
4.obvious yes to something

Response false, if the answer expresses:
1.obvious disagreement to something
2.obvious decision not to do something
3.obvious expression of making quest failed
4.obvious no to something

Response none, if the answer is not related to the question or didn't express a obvious attitude to something. 

Question: The quest leader decides that the player 2、player 3 will perform the quest. You can choose to agree or object. When the number of votes for agreement exceeds 2/3, the selection will be passed and the quest will be executed. Otherwise, the selection will be re-selected until the selection is passed. Please vote to agree or disagree.
Answer: I object to the selection of player 2 and player 3 for the quest. We need to thoroughly discuss the choices before making a decision. It is important to consider all possibilities and ensure the quest's success.
"""
    },
    {
        "role": "assistant",
        "content": "false"
    },
    {
        "role": "user",
        "content": """Complete the following instructions based on the questions and answers. 

Response true, if the answer expresses:
1.obvious agreement to something
2.obvious decision to do something
3.obvious expression of making quest successful
4.obvious yes to something

Response false, if the answer expresses:
1.obvious disagreement to something
2.obvious decision not to do something
3.obvious expression of making quest failed
4.obvious no to something

Response none, if the answer is not related to the question or didn't express a obvious attitude to something. 

Question: The quest leader decides that the player player 2、player 3、player 4 will perform the quest. You can choose to agree or object. Please vote.
Answer: I have observed that players 1, 2, and 5 raised suspicions during the identification phase, but their roles are uncertain. It is important for us to gather more information and evaluate the conversations before deciding on the quest candidates. However, I am cautious about revealing my own role prematurely, as it could benefit the red camp. Therefore, I will continue to observe and analyze the discussions without disclosing my role.
"""
    },
    {
        "role": "assistant",
        "content": "none"
    }
]