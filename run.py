"""공장 A의 팀 공통 Dueling Double DQN 실험을 실행한다."""
import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ess_env import ESSEnv, load_year, tariff
from dqn import train, greedy, DuelingQ

ROOT = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(env, model, policies=("rule_tou", "rule_peak", "rl"), collect_traces=True):
    summaries, traces = [], []
    for policy in policies:
        for month in env.months:
            state, done = env.reset(month), False
            while not done:
                date, _ = env.rows[env.i]
                _, band, _ = tariff(date, env.cfg)
                if policy == "rl":
                    action = greedy(model, state)
                elif band == 0:
                    action = 0
                elif policy == "rule_tou":
                    action = 4 if band == 2 else 2
                else:
                    action = 5  # 당월 피크 방어. 발표의 작년 피크 적응형과는 다르다.
                state, _, done, _ = env.step(action)
            summaries.append(dict(policy=policy, month=month, **env.summary()))
            if collect_traces:
                traces.extend(dict(policy=policy, month=month, **r) for r in env.history)
    # 원래 운영은 ESS 자체가 없다. 자가방전/월말 정산도 없다.
    for row in list(summaries[:len(env.months)]):
        base = {k: 0.0 for k in row}
        base.update(policy="original", month=row["month"], base_bill_won=row["base_bill_won"],
                    bill_won=row["base_bill_won"], final_soc="")
        summaries.append(base)
    return summaries, traces


class BestCheckpoint:
    """2013 검증 순절감으로 선택. 검증은 학습 환경/난수 상태를 바꾸지 않는다."""
    def __init__(self, env, out, interval, total):
        self.env, self.out, self.interval, self.total = env, out, interval, total
        self.best_score = -float("inf")
        self.best_episode = None
        self.history = []

    def __call__(self, model, episode, logs):
        if episode % self.interval and episode != self.total:
            return
        was_training = model.training
        model.eval()
        summaries, _ = evaluate(self.env, model, policies=("rl",), collect_traces=False)
        rl = [r for r in summaries if r["policy"] == "rl"]
        score = sum(r["net_saving_won"] for r in rl)
        if not np.isfinite(score):
            raise ValueError("Non-finite validation score")
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_episode = score, episode
            torch.save(model.state_dict(), self.out / "best_model.pt")
            (self.out / "best_checkpoint.json").write_text(json.dumps(dict(
                episode=episode, validation_year=2013, metric="annual_net_saving_won",
                score=score, includes_capex=False), indent=2), encoding="utf-8")
        self.history.append(dict(episode=episode, net_saving_won=score,
                                 bill_saving_won=sum(r["bill_saving_won"] for r in rl),
                                 best_net_saving_won=self.best_score,
                                 best_episode=self.best_episode, improved=int(improved)))
        write_csv(self.out / "validation_history.csv", self.history)
        write_csv(self.out / "training.csv", logs)
        print(f"VALIDATION episode={episode} net={score:,.0f} KRW "
              f"best={self.best_score:,.0f} at episode {self.best_episode}", flush=True)
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=ROOT / "data/factory_A_hourly_kwh_v3.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eval-every", type=int, default=12)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.eval_every < 1:
        parser.error("--eval-every must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    data = args.data
    if not data.is_file():
        parser.error(f"Data file missing: {data}. See README.md.")
    data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    training = load_year(data, cfg["train_year"])
    validation = load_year(data, cfg["validation_year"])
    maximum = max(x for rows in training.values() for _, x in rows)
    out = args.output or ROOT / "outputs" / datetime.now().strftime(f"%Y%m%d_%H%M%S_{cfg['factory']}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(config=cfg, seed=args.seed, episodes=args.episodes, eval_every=args.eval_every,
                    selection_metric="2013 annual net saving, excluding CAPEX", train_max_kw=maximum,
                    power_kw=maximum * cfg["power_fraction"],
                    capacity_kwh=maximum * cfg["power_fraction"] * cfg["duration_hours"],
                    data_sha256=data_hash, replay_mode="uniform", n_step=1,
                    python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    purpose="First practice only; no 2014 evaluation; no CAPEX/NPV conclusion")
    (out / "settings.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Output: {out}\nWARNING: {cfg['tariff_status']}", flush=True)
    checkpoint = BestCheckpoint(ESSEnv(validation, cfg, maximum), out, args.eval_every, args.episodes)
    model, logs = train(ESSEnv(training, cfg, maximum), cfg, args.episodes, args.seed, checkpoint)
    torch.save(model.state_dict(), out / "last_model.pt")
    last_summaries, _ = evaluate(ESSEnv(validation, cfg, maximum), model, policies=("rl",), collect_traces=False)
    write_csv(out / "last_validation_monthly.csv", last_summaries)
    model = DuelingQ()
    model.load_state_dict(torch.load(out / "best_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    summaries, traces = evaluate(ESSEnv(validation, cfg, maximum), model)
    write_csv(out / "training.csv", logs)
    write_csv(out / "validation_monthly.csv", summaries)
    write_csv(out / "validation_hourly.csv", traces)
    annual = []
    for policy in ("original", "rule_tou", "rule_peak", "rl"):
        subset = [r for r in summaries if r["policy"] == policy]
        bill = sum(r["bill_saving_won"] for r in subset)
        net = sum(r["net_saving_won"] for r in subset)
        annual.append(dict(policy=policy, bill_saving_won=bill, net_saving_won=net))
        print(f"{policy:12s} bill savings={bill:>14,.0f} net savings={net:>14,.0f} KRW")
    write_csv(out / "validation_annual.csv", annual)
    print(f"BEST model: episode {checkpoint.best_episode}. Last model saved separately.")
    print("Done. These are validation/demo results, not evidence of RL superiority.")


if __name__ == "__main__":
    main()
