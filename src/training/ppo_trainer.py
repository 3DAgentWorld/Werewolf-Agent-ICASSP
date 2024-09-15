#!/usr/bin/env python 
# encoding: utf-8 
# @author: yihuai lan
# @fileName: ppo_trainer.py 
# @date: 2024/9/11 17:00 
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeciderWithValue(nn.Module):
    def __init__(self, decider, state_dim, pred_dim, hidden_dim=128):
        super().__init__()
        self.decider = decider
        self.value_head = nn.Sequential(
            nn.Linear(state_dim + pred_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_vec, pred_vec, wr_cons):
        logits = self.decider(state_vec, pred_vec, wr_cons)
        x = torch.cat([state_vec, pred_vec, wr_cons], dim=-1)
        value = self.value_head(x)
        return logits, value.squeeze(-1)


class PPOTrainer:
    def __init__(self, decider, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2, epochs=4, batch_size=8):
        self.decider = decider
        self.optimizer = torch.optim.Adam(decider.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer = []

    def store(self, state_vec, pred_vec, wr_cons, action_idx, reward, done):
        self.buffer.append({
            "state": state_vec,
            "pred": pred_vec,
            "wr": wr_cons,
            "action": action_idx,
            "reward": reward,
            "done": done
        })

    def compute_advantage(self, rewards, values, dones):
        advantages = []
        gae = 0
        next_value = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    def update(self):
        if not self.buffer:
            return
        states = torch.tensor([x["state"] for x in self.buffer], dtype=torch.float32)
        preds = torch.tensor([x["pred"] for x in self.buffer], dtype=torch.float32)
        wrs = torch.tensor([[x["wr"]] for x in self.buffer], dtype=torch.float32)
        actions = torch.tensor([x["action"] for x in self.buffer], dtype=torch.long)
        rewards = [x["reward"] for x in self.buffer]
        dones = [x["done"] for x in self.buffer]

        with torch.no_grad():
            old_logits, old_values = self.decider(states, preds, wrs)
            old_log_probs = F.log_softmax(old_logits, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            advantages, returns = self.compute_advantage(rewards, old_values.tolist(), dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        for _ in range(self.epochs):
            logits, values = self.decider(states, preds, wrs)
            log_probs = F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            loss = actor_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.buffer.clear()
