#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: vllm_agent.py 
# @date: 2024/9/12 11:20 
#

from typing import List, Any

from ..abs_agent import MessageSender
from ...apis.vllm_api import vllm_chat


class VLLMMessageSender(MessageSender):
    def __init__(self, model, tokenizer, temperature, **kwargs):
        super().__init__(model, tokenizer, temperature, **kwargs)
        self.api_key = kwargs.get('api_key', 'EMPTY')
        self.base_url = kwargs.get('base_url', 'http://localhost:8000/v1')

    def send_message(self, messages: List[dict], model: Any = None, tokenizer: Any = None,
                     temperature: float = None) -> str:
        __model = model or self.model
        __temperature = temperature or self.temperature
        return vllm_chat(__model, messages, __temperature, self.api_key, self.base_url)
