#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: decider.py 
# @date: 2024/9/11 14:10 
#

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decider(nn.Module):
    def __init__(self, state_dim, pred_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.pred_dim = pred_dim
        self.action_dim = action_dim
        self.embedding = nn.Linear(state_dim + pred_dim + 1, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_vec, pred_vec, wr_cons):
        x = torch.cat([state_vec, pred_vec, wr_cons], dim=-1)
        x = F.relu(self.embedding(x))
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

    def decide(self, state_vec, pred_vec, wr_cons, action_mask, candidate_actions):
        with torch.no_grad():
            device = next(self.parameters()).device
            s = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)
            p = torch.tensor(pred_vec, dtype=torch.float32).unsqueeze(0).to(device)
            w = torch.tensor([wr_cons], dtype=torch.float32).unsqueeze(0).to(device)
            mask = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0).to(device)
            logits = self.forward(s, p, w)
            masked_logits = logits - mask * 1e9
            probs = F.softmax(masked_logits, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
            return candidate_actions[action_idx], action_idx


class RandomDecider:
    def decide(self, state_vec, pred_vec, wr_cons, action_mask, candidate_actions):
        valid = [i for i, m in enumerate(action_mask) if m < 0.5]
        idx = random.choice(valid) if valid else 0
        return candidate_actions[idx], idx
