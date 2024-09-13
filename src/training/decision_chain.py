#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: decision_chain.py 
# @date: 2024/9/11 16:00 
#

import math


class DecisionChainReward:
    def __init__(self, alpha=2.0):
        self.alpha = alpha
        self.chain_db = {}

    def register_chain(self, chain_tuple, win_rate):
        self.chain_db[chain_tuple] = win_rate

    def get_win_rate(self, chain_tuple):
        return self.chain_db.get(chain_tuple, 0.5)

    def chain_reward(self, chain_tuple):
        wr = self.get_win_rate(chain_tuple)
        return self.alpha * (wr - 0.5)

    def controllable_reward(self, chain_tuple, wr_cons, epsilon=0.1, s=2.0, k=0.05):
        wr_dc = self.get_win_rate(chain_tuple)
        d = (wr_cons - wr_dc) ** 2
        r = -math.tanh((d - epsilon ** 2) / k)
        if r >= 0:
            cr = r * (1 - d / epsilon) * s
        else:
            cr = r * (d - epsilon) / (1 - epsilon) * s
        return cr
