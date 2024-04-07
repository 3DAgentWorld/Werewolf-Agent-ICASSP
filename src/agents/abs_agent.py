#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: abs_agent.py 
# @date: 2024/2/18 14:04 
#
# describe:
#

from abc import abstractmethod
from typing import List, Any


class Agent:
    name = None
    role = None

    def __init__(self, name: str, role: str, **kwargs):
        self.name = name
        self.role = role

    @abstractmethod
    def step(self, message: str) -> str:
        """
        interact with the agent.
        :param message: input to the agent
        :return: response of the agent
        """
        pass

    @abstractmethod
    def receive(self, name: str, message: str) -> None:
        """
        receive the message from other agents.
        :param name: name of the agent which send the message
        :param message: content of the agent receives.
        :return:
        """
        pass

    @classmethod
    def init_instance(cls, **kwargs):
        return cls(**kwargs)


class MessageSender:
    model = None
    tokenizer = None
    temperature = None

    def __init__(self, model, tokenizer, temperature, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    @abstractmethod
    def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                     temperature: float = None) -> str:
        """
        Handling conversation with LLM.

        :param messages: conversation history
        :param model: base model
        :param tokenizer: tokenizer
        :param temperature: temperature, a float number greater than or equal to 0 and less than or equal to 1.
        :return: response of current conversation
        """
        raise NotImplementedError("The Agent class should inherit from the MessageSender class and implement the "
                                  "send_message method.")
