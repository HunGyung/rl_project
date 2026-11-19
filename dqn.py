"""Dueling: V와 A를 분리. Double: 행동 선택과 목표값 평가를 분리."""
import random
from collections import deque

import numpy as np
import torch
from torch import nn


class DuelingQ(nn.Module):
    def __init__(self, n_inputs=22):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(n_inputs, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, 6)

    def forward(self, state):
        features = self.features(state)
        value, advantage = self.value(features), self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


def greedy(model, state):
    with torch.no_grad():
        return int(model(torch.from_numpy(state).unsqueeze(0)).argmax(dim=1).item())


def train(env, cfg, episodes, seed, on_episode=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    online, target = DuelingQ(env.obs_dim), DuelingQ(env.obs_dim)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg["learning_rate"])
    replay = deque(maxlen=cfg["replay_capacity"])
    logs, steps = [], 0
    # 12회당 모든 달을 한 번씩 학습. 월 내부 시간 순서는 유지한다.
    month_order = []
    decay_steps = max(1, episodes * 730 * 0.8)
    for episode in range(episodes):
        if not month_order:
            month_order = random.sample(list(env.months), len(env.months))
        month = month_order.pop()
        state, done, losses = env.reset(month), False, []
        while not done:
            epsilon = max(0.05, 1 - 0.95 * steps / decay_steps)
            action = random.randrange(6) if random.random() < epsilon else greedy(online, state)
            next_state, reward, done, _ = env.step(action)
            replay.append((state, action, reward, next_state, done))
            state = next_state
            steps += 1
            if steps >= cfg["warmup_steps"] and len(replay) >= cfg["batch_size"] and steps % cfg["train_every"] == 0:
                batch = random.sample(replay, cfg["batch_size"])
                s, a, r, ns, done_flags = zip(*batch)
                s = torch.tensor(np.array(s), dtype=torch.float32)
                ns = torch.tensor(np.array(ns), dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32)
                not_done = 1 - torch.tensor(done_flags, dtype=torch.float32)
                prediction = online(s).gather(1, a).squeeze(1)
                with torch.no_grad():
                    # Double DQN: online에서 argmax, target에서 해당 Q값 평가.
                    next_action = online(ns).argmax(dim=1, keepdim=True)
                    next_q = target(ns).gather(1, next_action).squeeze(1)
                    expected = r + cfg["gamma"] * not_done * next_q
                loss = nn.functional.smooth_l1_loss(prediction, expected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                with torch.no_grad():
                    for slow, fast in zip(target.parameters(), online.parameters()):
                        slow.lerp_(fast, cfg["tau"])
                losses.append(float(loss.item()))
        summary = env.summary()
        logs.append(dict(episode=episode + 1, month=month, steps=steps, epsilon=epsilon,
                         mean_loss=float(np.mean(losses)) if losses else None, **summary))
        print(f"episode={episode + 1}/{episodes} month={month:02d} steps={steps} "
              f"reward={summary['reward_sum_won']:,.0f} KRW epsilon={epsilon:.3f}", flush=True)
        if on_episode is not None:
            on_episode(online, episode + 1, logs)
    return online.eval(), logs
