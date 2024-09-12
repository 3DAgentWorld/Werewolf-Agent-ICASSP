#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: vllm_api.py 
# @date: 2024/9/12 10:30 
#

import time
import warnings

import openai
from openai import OpenAI


def vllm_chat(model, messages, temperature, api_key="EMPTY", base_url="http://localhost:8000/v1"):
    client = OpenAI(api_key=api_key, base_url=base_url)
    retry = 0
    flag = False
    out = ''
    while retry < 10 and not flag:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024
            )
            out = response.choices[0].message.content
            flag = True
        except Exception as e:
            if retry < 10:
                retry += 1
                warnings.warn(f"{e} retry:{retry}")
                time.sleep(1)
                continue
            else:
                raise e
    client.close()
    return out
