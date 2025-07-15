import os
import time
import gc
import random
import asyncio
from io import BytesIO
from collections import deque
from typing import Optional, List, Tuple

import cv2
import psutil
import aiohttp
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
from PIL import Image

from models import create_vision_model
from data import capture_single_screenshot
from utils import MetricLogger, CurriculumManager, MetaTuner, UITracker


def monitor_memory() -> float:
    proc = psutil.Process()
    rss = proc.memory_info().rss / (1024 * 1024)
    print(f"Memory usage: {rss:.2f} MB")
    return rss


class MemoryCleaner:
    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BaseAgent:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)

    def select_action(self, *args, **kwargs):
        raise NotImplementedError

    def optimize(self, *args, **kwargs):
        raise NotImplementedError


class PPOPolicy(nn.Module):
    def __init__(self, input_dim: int, num_subgoals: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_subgoals)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOAgent(BaseAgent):
    def __init__(
        self,
        input_dim: int,
        num_subgoals: int,
        device: str = 'cpu',
        lr: float = 3e-4,
        eps_clip: float = 0.2,
    ):
        super().__init__(device)
        self.policy = PPOPolicy(input_dim, num_subgoals).to(self.device)
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_head.parameters(), lr=lr)
        self.eps_clip = eps_clip
        self.buffer = deque(maxlen=2048)

    def select_action(
        self, state: np.ndarray
    ) -> Tuple[int, float, float, float]:
        state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        logits = self.policy(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.value_head(state_tensor).squeeze()
        return (
            int(action.item()),
            float(probs[0, action].item()),
            float(logprob.item()),
            float(value.item()),
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ):
        self.buffer.append((state, action, logprob, reward, done, value))

    def update(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        epochs: int = 4
    ) -> Tuple[float, float]:
        batch = list(self.buffer)
        states, actions, old_logprobs, rewards, dones, values = zip(*batch)
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        lastgae = 0
        for t in reversed(range(T)):
            next_nonterm = 1.0 - dones[t]
            next_val = values[t+1] if t < T-1 else values[t]
            delta = rewards[t] + gamma * next_val * next_nonterm - values[t]
            lastgae = delta + gamma * lam * next_nonterm * lastgae
            advantages[t] = lastgae
        returns = advantages + np.array(values)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_logprobs_t = torch.tensor(old_logprobs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        for _ in range(epochs):
            logits = self.policy(states_t)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            new_logprobs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logprobs - old_logprobs_t)

            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            values_pred = self.value_head(states_t).squeeze()
            value_loss = F.mse_loss(returns_t, values_pred)

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        self.buffer.clear()
        return policy_loss.item(), value_loss.item()


class CNNBase(nn.Module):
    """
    Deeper CNN for rich pixel-level feature extraction from screenshots.
    Input: RGB image tensor (batch, 3, H, W)
    """

    def __init__(self, input_channels=3, dropout=0.2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # 32x ((H-8)/4+1) x ((W-8)/4+1)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten(),
            nn.Dropout(dropout)
        )

        # Calculate flattened feature size dynamically for input size 84x84 (common for Atari)
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        with torch.no_grad():
            self.feature_dim = self.conv_layers(dummy_input).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class DQNAgent(BaseAgent):
    """
    DQN agent with deeper CNN for pixel-based state input.
    Output: 2 continuous normalized values (x, y coordinates).
    """

    def __init__(self, device='cpu', lr: float = 1e-3, gamma: float = 0.99):
        super().__init__(device)
        self.base = CNNBase().to(self.device)
        self.policy_net = nn.Sequential(
            self.base,
            nn.Linear(self.base.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid(),
        ).to(self.device)

        self.target_net = nn.Sequential(*self.policy_net.children()).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.update_steps = 0

    def select_action(
        self, state: np.ndarray, epsilon: float = 0.1
    ) -> np.ndarray:
        if random.random() < epsilon:
            return np.random.rand(2)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            if s.ndim == 3:
                s = s.unsqueeze(0)
            return self.policy_net(s).cpu().numpy()[0]

    def optimize(
        self, replay_buffer, batch_size: int = 32, beta: float = 0.4
    ) -> Optional[float]:
        if len(replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(batch_size, beta)

        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device)

        q = self.policy_net(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1)[0]
            target_q = r + self.gamma * next_q * (1 - d)

        loss = (w * (q - target_q.detach()).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
