"""공장별 3×3 ESS 설비를 연간 래칫 요금으로 학습·검증한다."""
import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import torch

from annual_env import AnnualESSEnv
from ess_env import load_year
from dqn import train, greedy, DuelingQ
from forecaster import QuantileForecaster
from rules import percentile_thresholds
from tariff_engine import original_monthly_peaks

ROOT = Path(__file__).resolve().parent
POLICIES = ("rule_tou", "rule_percentile", "rl")
POWER_PERCENTAGES = (3, 5, 7)
DURATIONS_HOURS = (1, 2, 4)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(env, model, policies=POLICIES, collect_traces=True, *, percentile_limits=None):
    if "rule_percentile" in policies and (percentile_limits is None or len(percentile_limits) != len(env.rows)):
        raise ValueError("The percentile rule requires one threshold per evaluation hour")
    summaries, traces = [], []
    for policy in policies:
        state, done = env.reset(), False
        while not done:
            _, band, _ = env.tariffs[env.i]
            mask = env.action_mask()
            target = None
            if policy == "rl":
                action = greedy(model, state, mask)
            elif band == 0:
                action = 0
            elif policy == "rule_tou":
                action = 4 if band == 2 else 2
            elif policy == "rule_percentile":
                target = percentile_limits[env.i]
                action = 5 if target is not None and env.forecasts[env.i] > target else 2
            else:
                raise ValueError(f"Unknown policy: {policy}")
            state, _, done, _ = env.step(action if mask[action] else 2, peak_target_kw=target)
        summaries.extend(dict(policy=policy, **row) for row in env.monthly_summaries())
        if collect_traces:
            traces.extend(dict(policy=policy,
                               percentile_threshold_kw=percentile_limits[i] if policy == "rule_percentile" else None,
                               **row) for i, row in enumerate(env.history))
    # 원래 운영은 ESS 자체가 없다. 자가방전/월말 정산도 없다.
    first_policy_rows = [row for row in summaries if row["policy"] == policies[0]]
    for row in first_policy_rows:
        base = {k: 0.0 for k in row}
        base.update(policy="original", month=row["month"], base_bill_won=row["base_bill_won"],
                    bill_won=row["base_bill_won"], final_soc="", base_month_peak_kw=row["base_month_peak_kw"],
                    month_peak_kw=row["base_month_peak_kw"], base_billing_peak_kw=row["base_billing_peak_kw"],
                    billing_peak_kw=row["base_billing_peak_kw"])
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
    parser.add_argument("--factory", choices=("A", "B", "C"), default="A")
    parser.add_argument("--episodes", type=int, default=12, help="Monthly episodes per ESS size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eval-every", type=int, default=12, help="Validate every N monthly episodes and at the end")
    parser.add_argument("--power-percent", type=int, choices=POWER_PERCENTAGES, help="Run only this output percentage")
    parser.add_argument("--duration-hours", type=int, choices=DURATIONS_HOURS, help="Run only this duration")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.eval_every < 1:
        parser.error("--eval-every must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cfg["factory"] = args.factory
    data = args.data or ROOT / "data" / f"factory_{args.factory}_hourly_kwh_v3.csv"
    if not data.is_file():
        parser.error(f"Data file missing: {data}. See README.md.")
    data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    training = load_year(data, cfg["train_year"])
    validation = load_year(data, cfg["validation_year"])
    maximum = max(x for rows in training.values() for _, x in rows)
    forecaster = QuantileForecaster(training, cfg, tau=cfg["forecaster_tau"])
    train_forecast = forecasts_for_year(training, forecaster)
    validation_forecast = forecasts_for_year(validation, forecaster)
    prior = original_monthly_peaks(training, cfg)
    validation_rows = [row for month in sorted(validation) for row in validation[month]]
    validation_start = validation_rows[0][0]
    lookback_days, quantile, min_days = 7, 0.95, 3
    prior_rows = [row for rows in training.values() for row in rows
                  if validation_start - timedelta(days=lookback_days) <= row[0] < validation_start]
    percentile_limits = percentile_thresholds(validation_rows, prior_rows, lookback_days=lookback_days,
                                              quantile=quantile, min_days=min_days)
    grid = list(product((args.power_percent,) if args.power_percent is not None else POWER_PERCENTAGES,
                        (args.duration_hours,) if args.duration_hours is not None else DURATIONS_HOURS))
    out = args.output or ROOT / "outputs" / datetime.now().strftime(f"annual_{args.factory}_%Y%m%d_%H%M%S_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(factory=args.factory, grid=grid, seed=args.seed, episodes=args.episodes, eval_every=args.eval_every,
                    episode_unit="calendar_month", terminal_boundary="end_of_calendar_year",
                    selection_metric="2013 annual ratchet net saving, excluding CAPEX",
                    train_max_kw=maximum, data_sha256=data_hash, replay_mode="per", n_step=1,
                    observation_timing="hour_start_completed_intervals_only",
                    soc_mask="upper_charge_lower_discharge", forecaster=forecaster.report,
                    percentile_rule=dict(lookback_days=lookback_days, quantile=quantile, min_days=min_days,
                                         history="same hour of previous calendar days"),
                    training_opening_peak_history="unavailable before 2012; billing floor used until peaks accrue",
                    validation_opening_peak_history="previous 11 months of 2012 original load for every policy",
                    settlement="SoC carried across months, shortage settled at year end",
                    python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    status="running")
    save_settings(out, metadata)
    print(f"Output: {out}\n{cfg['tariff_status']}", flush=True)
    results = []
    for index, (power, hours) in enumerate(grid, 1):
        size = f"p{power:02d}_h{hours}"
        case_out = out / size
        case_out.mkdir()
        case_cfg = dict(cfg, power_fraction=power / 100, duration_hours=hours)
        train_env = AnnualESSEnv(training, case_cfg, maximum, train_forecast)
        val_env = AnnualESSEnv(validation, case_cfg, maximum, validation_forecast, prior)
        case_settings = dict(metadata, config=case_cfg, power_kw=train_env.power, capacity_kwh=train_env.capacity,
                             observation_size=train_env.obs_dim, action_count=train_env.n_actions)
        save_settings(case_out, case_settings)
        print(f"\n[{index}/{len(grid)}] {args.factory} {size} ({train_env.power:.1f} kW, {train_env.capacity:.1f} kWh)", flush=True)
        annual, best_episode, last_net = run_case(case_out, train_env, val_env, case_cfg, args, percentile_limits)
        nets = {row["policy"]: row["net_saving_won"] for row in annual}
        results.append(dict(factory=args.factory, condition=size, power_percent=power, duration_hours=hours,
                            power_kw=train_env.power, capacity_kwh=train_env.capacity, seed=args.seed,
                            episodes=args.episodes, best_episode=best_episode, rl_best_net_won=nets["rl"],
                            rl_last_net_won=last_net, rule_percentile_net_won=nets["rule_percentile"],
                            rule_tou_net_won=nets["rule_tou"], original_net_won=nets["original"]))
        write_csv(out / "grid_summary.csv", results)
        save_settings(case_out, dict(case_settings, status="complete"))
    save_settings(out, dict(metadata, status="complete"))
    print(f"Completed: {out / 'grid_summary.csv'}")


def forecasts_for_year(months, forecaster):
    rows = [row for month in sorted(months) for row in months[month]]
    # predict()는 시각과 직전 완료 부하만 읽는다. 첫 시간은 학습 프로필을 사용한다.
    return [forecaster.predict(rows, i) for i in range(len(rows))]


def save_settings(folder, metadata):
    (folder / "settings.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def run_case(out, train_env, val_env, cfg, args, percentile_limits):
    checkpoint = BestCheckpoint(val_env, out, args.eval_every, args.episodes)
    model, logs = train(train_env, cfg, args.episodes, args.seed, checkpoint)
    torch.save(model.state_dict(), out / "last_model.pt")
    last_summaries, _ = evaluate(val_env, model, policies=("rl",), collect_traces=False)
    write_csv(out / "last_validation_monthly.csv", last_summaries)
    last_net = sum(r["net_saving_won"] for r in last_summaries if r["policy"] == "rl")
    model = DuelingQ(val_env.obs_dim, val_env.n_actions)
    model.load_state_dict(torch.load(out / "best_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    summaries, traces = evaluate(val_env, model, percentile_limits=percentile_limits)
    write_csv(out / "training.csv", logs)
    write_csv(out / "validation_monthly.csv", summaries)
    write_csv(out / "validation_hourly.csv", traces)
    annual = []
    for policy in ("original", *POLICIES):
        subset = [r for r in summaries if r["policy"] == policy]
        bill = sum(r["bill_saving_won"] for r in subset)
        net = sum(r["net_saving_won"] for r in subset)
        fields = ("base_bill_won", "bill_won", "demand_saving_won", "energy_saving_won", "degradation_won",
                  "settlement_won", "charge_kwh", "discharge_kwh")
        annual.append(dict(policy=policy, bill_saving_won=bill, net_saving_won=net,
                           **{key: sum(row[key] for row in subset) for key in fields}))
        print(f"{policy:12s} bill savings={bill:>14,.0f} net savings={net:>14,.0f} KRW")
    write_csv(out / "validation_annual.csv", annual)
    print(f"BEST model: episode {checkpoint.best_episode}. Last model saved separately.")
    return annual, checkpoint.best_episode, last_net


if __name__ == "__main__":
    main()
