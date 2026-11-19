"""Dueling: V와 A를 분리. Double: 행동 선택과 목표값 평가를 분리."""
import random

import numpy as np
import torch
from torch import nn
from replay import PrioritizedReplay


class DuelingQ(nn.Module):
    def __init__(self, obs_dim=22, n_actions=6):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, n_actions)

    def forward(self, state):
        features = self.features(state)
        value, advantage = self.value(features), self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


def greedy(model, state, mask):
    with torch.no_grad():
        q = model(torch.from_numpy(state).unsqueeze(0))[0]
        return int(q.masked_fill(~torch.from_numpy(mask), -torch.inf).argmax().item())


def train(env, cfg, episodes, seed, on_episode=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    online = DuelingQ(env.obs_dim, env.n_actions)
    target = DuelingQ(env.obs_dim, env.n_actions)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg["learning_rate"])
    replay = PrioritizedReplay(cfg["replay_capacity"], cfg["per_alpha"], cfg["per_epsilon"], seed)
    logs, steps = [], 0
    months_per_pass = len(env.month_ends)
    full_passes, extra_months = divmod(episodes, months_per_pass)
    total_steps = full_passes * len(env.rows)
    if extra_months:
        total_steps += env.month_ends[extra_months - 1]
    decay_steps = max(1, total_steps * 0.8)
    done = True
    for episode in range(episodes):
        if done:
            state = env.reset()
            mask = env.action_mask()
        pass_index, month_index = divmod(episode, months_per_pass)
        month_start = env.i
        month = env.rows[month_start][0].month
        month_end = env.month_ends[month_index]
        losses = []
        upper_masked_hours = lower_masked_hours = 0
        # 월말은 로그 경계이다. 실제 종료 여부는 환경이 연말에만 돌려준다.
        while env.i < month_end:
            epsilon = max(0.05, 1 - 0.95 * steps / decay_steps)
            upper_masked_hours += int(not mask[0])
            lower_masked_hours += int(not mask[4])
            if random.random() < epsilon:
                action = int(random.choice(np.flatnonzero(mask)))
            else:
                action = greedy(online, state, mask)
            next_state, reward, done, _ = env.step(action)
            next_mask = env.action_mask()
            replay.append((state, action, reward, next_state, done, next_mask))
            state, mask = next_state, next_mask
            steps += 1
            if steps >= cfg["warmup_steps"] and len(replay) >= cfg["batch_size"] and steps % cfg["train_every"] == 0:
                beta = cfg["per_beta_start"] + (1 - cfg["per_beta_start"]) * steps / total_steps
                batch, indices, weights = replay.sample(cfg["batch_size"], beta)
                s, a, r, ns, done_flags, masks = zip(*batch)
                s = torch.tensor(np.array(s), dtype=torch.float32)
                ns = torch.tensor(np.array(ns), dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32)
                not_done = 1 - torch.tensor(done_flags, dtype=torch.float32)
                valid_actions = torch.as_tensor(np.asarray(masks), dtype=torch.bool)
                prediction = online(s).gather(1, a).squeeze(1)
                with torch.no_grad():
                    # Double DQN: online에서 argmax, target에서 해당 Q값 평가.
                    next_action = online(ns).masked_fill(~valid_actions, -torch.inf).argmax(dim=1, keepdim=True)
                    next_q = target(ns).gather(1, next_action).squeeze(1)
                    expected = r + cfg["gamma"] * not_done * next_q
                td_error = (expected - prediction).detach().numpy()
                per_item = nn.functional.smooth_l1_loss(prediction, expected, reduction="none")
                loss = (torch.from_numpy(weights) * per_item).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                replay.update(indices, td_error)
                with torch.no_grad():
                    for slow, fast in zip(target.parameters(), online.parameters()):
                        slow.lerp_(fast, cfg["tau"])
                losses.append(float(loss.item()))
        summary = env.summary(start=month_start)
        logs.append(dict(episode=episode + 1, training_pass=pass_index + 1, month=month,
                         year_terminal=int(done), steps=steps, epsilon=epsilon,
                         upper_masked_hours=upper_masked_hours, lower_masked_hours=lower_masked_hours,
                         mean_loss=float(np.mean(losses)) if losses else None, **summary))
        print(f"TRAIN episode={episode + 1}/{episodes} pass={pass_index + 1} month={month:02d} steps={steps} "
              f"net={summary['net_saving_won']:,.0f} KRW epsilon={epsilon:.3f}", flush=True)
        if on_episode is not None:
            on_episode(online, episode + 1, logs)
    return online.eval(), logs
