"""공장별 3×3 ESS 설비를 연간 래칫 요금으로 학습·평가한다.

두 단계로 실행한다.
  --stage develop (기본) : 2012 학습 → 2013 검증. 에피소드 수 · 설정 · 격자 칸 비교를 여기서 확정한다.
  --stage final          : 2012+2013 학습 → 2014 평가 1회. develop에서 정한 설정 그대로, 마지막 모델을 평가한다.

평가 연도는 두 상황으로 계산한다.
  year1  : 그해 ESS를 설치. 원래 운영과 ESS 모두 직전 해 원래 운영의 피크 이력으로 시작한다.
  steady : 2년차 이후(정상상태). 설치 후 평가 연도의 부하가 되풀이된다고 보고 3년차를 평가한다.
           1회차는 year1과 같고, 2·3회차는 ESS가 직전 회차에 평가 연도에서 만든 자기 피크로 시작한다.
           원래 운영도 평가 연도의 원래 피크를 이어받는다. 학습 연도에서 ESS가 낮춘 피크는 쓰지 않는다.

금액은 설치비를 뺀 값이 아니라 연간 운영 순절감이다. 보고의 기준은 마지막 모델(시드 평균과 범위)이다.
"""
import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime
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
SCENARIOS = ("year1", "steady")
POWER_PERCENTAGES = (3, 5, 7)
DURATIONS_HOURS = (1, 2, 4)
SIZE_BASE_YEAR = 2012          # 출력 비율 · 계약전력의 기준 연도 (두 단계 모두 같은 설비)
STAGES = {
    # episodes: 월 단위 에피소드 기본값. 두 단계 모두 학습 데이터를 20번 반복한다 (README "에피소드 수 결정").
    "develop": dict(train_years=(2012,), evaluate_year=2013, label="validation", episodes=240),
    "final": dict(train_years=(2012, 2013), evaluate_year=2014, label="test", episodes=480),
}
WARMUP_PASSES = 2
PERCENTILE_RULE = dict(lookback_days=7, quantile=0.95, min_days=3)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flat(months):
    return [row for month in sorted(months) for row in months[month]]


def load_months(path, years):
    """{(연, 월): [(시각, kWh), ...]}. 여러 해를 이어 붙여도 정렬 순서가 시간 순서이다."""
    return {(year, month): rows for year in years for month, rows in load_year(path, year).items()}


def forecasts_for_year(months, forecaster):
    rows = flat(months)
    # predict()는 시각과 직전 완료 부하만 읽는다. 데이터의 첫 시간은 학습 프로필을 사용한다.
    return [forecaster.predict(rows, i) for i in range(len(rows))]


def run_policy(env, policy, model=None, limits=None):
    """env.reset()부터 연말까지 한 정책으로 운전한다. 결정은 모두 시간 시작 시점 정보로 한다."""
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
            target = limits[env.i]
            action = 5 if target is not None and env.forecasts[env.i] > target else 2
        else:
            raise ValueError(f"Unknown policy: {policy}")
        state, _, done, _ = env.step(action if mask[action] else 2, peak_target_kw=target)
    return env


class Evaluator:
    """한 설비 · 한 평가 연도의 year1 / steady 평가.

    직전 해는 설치 시점의 원래 피크로만 쓴다. steady는 설치 첫해부터 평가 연도를 되풀이하므로,
    모델이 학습 연도에서 외운 피크 절감이 평가 성적에 들어가지 않는다."""

    def __init__(self, cfg, maximum, target_months, prior_months, forecast_at, limit_at, passes=WARMUP_PASSES):
        self.prior_peaks = original_monthly_peaks(prior_months, cfg)
        self.passes = passes
        dates = [date for date, _ in flat(target_months)]
        self.target_env = AnnualESSEnv(target_months, cfg, maximum, [forecast_at[d] for d in dates], self.prior_peaks)
        self.limits = [limit_at[d] for d in dates]

    def run(self, scenario, policy, model=None):
        env = self.target_env
        if scenario == "year1":
            env.set_opening(self.prior_peaks)
        elif scenario == "steady":
            env.set_opening(self.prior_peaks)   # 1회차 = 설치 첫해
            for _ in range(self.passes):
                run_policy(env, policy, model, self.limits)
                env.roll_over()         # 이번 회차의 원래 · ESS 피크가 다음 회차의 직전 해 이력이 된다
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        return run_policy(env, policy, model, self.limits)

    def evaluate(self, model, policies=POLICIES, scenarios=SCENARIOS, collect_traces=True):
        summaries, traces = [], []
        for scenario in scenarios:
            for policy in policies:
                env = self.run(scenario, policy, model)
                monthly = env.monthly_summaries()
                summaries.extend(dict(scenario=scenario, policy=policy, **row) for row in monthly)
                if collect_traces:
                    traces.extend(dict(scenario=scenario, policy=policy,
                                       percentile_threshold_kw=self.limits[i] if policy == "rule_percentile" else None,
                                       **row) for i, row in enumerate(env.history))
            # 원래 운영은 ESS 자체가 없다. 자가방전 · 연말 정산도 없다. 두 상황 모두 직전 해 원래 피크로 시작한다.
            for row in [r for r in summaries if r["scenario"] == scenario and r["policy"] == policies[0]]:
                base = {k: 0.0 for k in row}
                base.update(scenario=scenario, policy="original", month=row["month"],
                            base_bill_won=row["base_bill_won"], bill_won=row["base_bill_won"], final_soc="",
                            base_month_peak_kw=row["base_month_peak_kw"], month_peak_kw=row["base_month_peak_kw"],
                            base_billing_peak_kw=row["base_billing_peak_kw"],
                            billing_peak_kw=row["base_billing_peak_kw"])
                summaries.append(base)
        return summaries, traces


def annual_rows(summaries):
    fields = ("base_bill_won", "bill_won", "demand_saving_won", "energy_saving_won", "degradation_won",
              "settlement_won", "charge_kwh", "discharge_kwh", "clipped_steps")
    rows = []
    for scenario in SCENARIOS:
        for policy in ("original", *POLICIES):
            subset = [r for r in summaries if r["scenario"] == scenario and r["policy"] == policy]
            if subset:
                rows.append(dict(scenario=scenario, policy=policy,
                                 bill_saving_won=sum(r["bill_saving_won"] for r in subset),
                                 net_saving_won=sum(r["net_saving_won"] for r in subset),
                                 **{key: sum(r[key] for r in subset) for key in fields}))
    return rows


def rl_net(summaries, scenario):
    return sum(r["net_saving_won"] for r in summaries if r["policy"] == "rl" and r["scenario"] == scenario)


class BestCheckpoint:
    """develop 단계: 2013 steady(2년차 이후) 순절감으로 체크포인트를 고른다. 학습 환경 · 난수는 건드리지 않는다."""
    def __init__(self, evaluator, out, interval, total):
        self.evaluator, self.out, self.interval, self.total = evaluator, out, interval, total
        self.best_score = -float("inf")
        self.best_episode = None
        self.history = []

    def __call__(self, model, episode, logs):
        if episode % self.interval and episode != self.total:
            return
        was_training = model.training
        model.eval()
        summaries, _ = self.evaluator.evaluate(model, policies=("rl",), collect_traces=False)
        score, year1 = rl_net(summaries, "steady"), rl_net(summaries, "year1")
        if not (np.isfinite(score) and np.isfinite(year1)):
            raise ValueError("Non-finite validation score")
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_episode = score, episode
            torch.save(model.state_dict(), self.out / "best_model.pt")
            (self.out / "best_checkpoint.json").write_text(json.dumps(dict(
                episode=episode, validation_year=2013, metric="annual_net_saving_won (steady)",
                score=score, includes_capex=False), indent=2), encoding="utf-8")
        self.history.append(dict(episode=episode, steady_net_saving_won=score, year1_net_saving_won=year1,
                                 best_steady_net_saving_won=self.best_score,
                                 best_episode=self.best_episode, improved=int(improved)))
        write_csv(self.out / "validation_history.csv", self.history)
        write_csv(self.out / "training.csv", logs)
        print(f"VALIDATION episode={episode} steady={score:,.0f} year1={year1:,.0f} KRW "
              f"best steady={self.best_score:,.0f} at episode {self.best_episode}", flush=True)
        model.train(was_training)


class TrainingLog:
    """final 단계: 평가 연도(2014)는 학습이 끝날 때까지 보지 않는다. 학습 기록만 저장한다."""
    def __init__(self, out, interval):
        self.out, self.interval = out, interval

    def __call__(self, model, episode, logs):
        if episode % self.interval == 0:
            write_csv(self.out / "training.csv", logs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=tuple(STAGES), default="develop")
    parser.add_argument("--factory", choices=("A", "B", "C"), default="A")
    parser.add_argument("--episodes", type=int,
                        help="Monthly episodes per ESS size (default: develop 240 = 20 passes over 2012; "
                             "final 480 = 20 passes over 2012-2013)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eval-every", type=int, default=12, help="develop: validate every N monthly episodes and at the end")
    parser.add_argument("--power-percent", type=int, choices=POWER_PERCENTAGES, help="Run only this output percentage")
    parser.add_argument("--duration-hours", type=int, choices=DURATIONS_HOURS, help="Run only this duration")
    parser.add_argument("--cycle-life", type=float,
                        help="Cycle life scenario (default config.json 5250 = PNNL 4,200 full-depth cycles / 0.8; "
                             "conservative replacement at 80%% capacity: 2400)")
    args = parser.parse_args()
    if args.episodes is None:
        args.episodes = STAGES[args.stage]["episodes"]
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.eval_every < 1:
        parser.error("--eval-every must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    stage = STAGES[args.stage]
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cfg["factory"] = args.factory
    if args.cycle_life is not None:
        if args.cycle_life <= 0:
            parser.error("--cycle-life must be positive")
        cfg["cycle_life"] = args.cycle_life
    data = args.data or ROOT / "data" / f"factory_{args.factory}_hourly_kwh_v3.csv"
    if not data.is_file():
        parser.error(f"Data file missing: {data}. See README.md.")
    data_hash = hashlib.sha256(data.read_bytes()).hexdigest()

    train_years, evaluate_year = stage["train_years"], stage["evaluate_year"]
    years = tuple(range(min(train_years), evaluate_year + 1))      # 평가 연도 이후 데이터는 읽지 않는다
    all_months = load_months(data, years)
    by_year = lambda year: {k: v for k, v in all_months.items() if k[0] == year}
    training = {k: v for k, v in all_months.items() if k[0] in train_years}
    maximum = max(x for _, x in flat(by_year(SIZE_BASE_YEAR)))
    forecaster = QuantileForecaster(training, cfg, tau=cfg["forecaster_tau"])
    # 모든 시간의 예측 · 분위수 규칙 기준을 이어진 시간축에서 한 번에 만든다 (각 값은 그 시각 이전 정보만 사용).
    all_rows = flat(all_months)
    forecast_at = {date: value for (date, _), value in zip(all_rows, forecasts_for_year(all_months, forecaster))}
    limit_at = dict(zip((date for date, _ in all_rows), percentile_thresholds(all_rows, **PERCENTILE_RULE)))

    prior_year = evaluate_year - 1
    target_months, prior_months = by_year(evaluate_year), by_year(prior_year)

    grid = list(product((args.power_percent,) if args.power_percent is not None else POWER_PERCENTAGES,
                        (args.duration_hours,) if args.duration_hours is not None else DURATIONS_HOURS))
    out = args.output or ROOT / "outputs" / datetime.now().strftime(
        f"{args.stage}_{args.factory}_%Y%m%d_%H%M%S_seed{args.seed}_cycle{cfg['cycle_life']:g}")
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(stage=args.stage, factory=args.factory, grid=grid, seed=args.seed, episodes=args.episodes,
                    eval_every=args.eval_every if args.stage == "develop" else None,
                    train_years=list(train_years), evaluate_year=evaluate_year, size_base_year=SIZE_BASE_YEAR,
                    episode_unit="calendar_month", terminal_boundary="end_of_training_data",
                    training_opening_peak_history="pass 1: training-data original peaks shifted before the data; "
                                                  "later passes: previous pass's own peaks",
                    scenarios=dict(year1=f"ESS installed in {evaluate_year}; every policy opens with {prior_year} original peaks",
                                   steady=f"third year after installation with {evaluate_year} load repeated: pass 1 equals "
                                          f"year1, later passes open with the ESS peaks of the previous {evaluate_year} "
                                          f"pass; original operation inherits {evaluate_year} original peaks; "
                                          "ESS peaks from training years are not used"),
                    model=("report last model; best checkpoint by 2013 steady net saving is reference only"
                           if args.stage == "develop"
                           else "last model after the fixed number of episodes (no checkpoint selection)"),
                    money_basis="annual operating net saving, excluding CAPEX and fixed O&M",
                    cycle_life=cfg["cycle_life"], calendar_life_years=cfg["calendar_life_years"],
                    aging_model="1/life = cycles/cycle_life + 1/calendar_life; "
                                "operation charges the cycle part per DC kWh",
                    train_max_kw=maximum, data_sha256=data_hash, replay_mode="per", n_step=1,
                    observation_timing="hour_start_completed_intervals_only",
                    soc_mask="upper_charge_lower_discharge", forecaster=forecaster.report,
                    percentile_rule=dict(PERCENTILE_RULE, history="same hour of previous calendar days"),
                    settlement="SoC carried across months, shortage settled at the end of each annual run",
                    python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    status="running")
    save_settings(out, metadata)
    print(f"Output: {out}\nStage {args.stage}: train {list(train_years)} -> evaluate {evaluate_year}\n"
          f"{cfg['tariff_status']}", flush=True)
    results = []
    for index, (power, hours) in enumerate(grid, 1):
        size = f"p{power:02d}_h{hours}"
        case_out = out / size
        case_out.mkdir()
        case_cfg = dict(cfg, power_fraction=power / 100, duration_hours=hours)
        train_env = AnnualESSEnv(training, case_cfg, maximum, [forecast_at[d] for d, _ in flat(training)])
        train_env.set_opening(train_env.as_previous_year(original_monthly_peaks(training, case_cfg)))
        evaluator = Evaluator(case_cfg, maximum, target_months, prior_months, forecast_at, limit_at)
        case_settings = dict(metadata, config=case_cfg, power_kw=train_env.power, capacity_kwh=train_env.capacity,
                             degradation_krw_per_dc_kwh=train_env.degradation_per_dc_kwh,
                             observation_size=train_env.obs_dim, action_count=train_env.n_actions)
        save_settings(case_out, case_settings)
        print(f"\n[{index}/{len(grid)}] {args.factory} {size} ({train_env.power:.1f} kW, {train_env.capacity:.1f} kWh, "
              f"degradation {train_env.degradation_per_dc_kwh:.1f} KRW/DC kWh)",
              flush=True)
        annual, extra = run_case(case_out, train_env, evaluator, case_cfg, args, stage["label"])
        for scenario in SCENARIOS:
            nets = {row["policy"]: row["net_saving_won"] for row in annual if row["scenario"] == scenario}
            # develop의 annual은 최고 체크포인트이므로 참고 열로 두고, 보고 기준은 항상 마지막 모델이다.
            model_columns = (dict(rl_last_net_won=extra[scenario]["rl_last_net_won"], rl_best_net_won=nets["rl"],
                                  best_episode=extra[scenario]["best_episode"])
                             if args.stage == "develop" else dict(rl_last_net_won=nets["rl"]))
            results.append(dict(factory=args.factory, stage=args.stage, evaluate_year=evaluate_year,
                                condition=size, scenario=scenario, power_percent=power, duration_hours=hours,
                                power_kw=train_env.power, capacity_kwh=train_env.capacity, seed=args.seed,
                                episodes=args.episodes, cycle_life=cfg["cycle_life"],
                                rule_percentile_net_won=nets["rule_percentile"], rule_tou_net_won=nets["rule_tou"],
                                original_net_won=nets["original"], **model_columns))
        write_csv(out / "grid_summary.csv", results)
        save_settings(case_out, dict(case_settings, status="complete"))
    save_settings(out, dict(metadata, status="complete"))
    print(f"Completed: {out / 'grid_summary.csv'}")


def save_settings(folder, metadata):
    (folder / "settings.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def run_case(out, train_env, evaluator, cfg, args, label):
    extra = {}
    if args.stage == "develop":
        checkpoint = BestCheckpoint(evaluator, out, args.eval_every, args.episodes)
        model, logs = train(train_env, cfg, args.episodes, args.seed, checkpoint)
        torch.save(model.state_dict(), out / "last_model.pt")
        last_summaries, _ = evaluator.evaluate(model, policies=("rl",), collect_traces=False)
        write_csv(out / "last_validation_monthly.csv", last_summaries)
        write_csv(out / "last_validation_annual.csv", annual_rows(last_summaries))
        for scenario in SCENARIOS:
            extra[scenario] = dict(best_episode=checkpoint.best_episode,
                                   rl_last_net_won=rl_net(last_summaries, scenario))
        model = DuelingQ(train_env.obs_dim, train_env.n_actions)
        model.load_state_dict(torch.load(out / "best_model.pt", map_location="cpu", weights_only=True))
    else:
        model, logs = train(train_env, cfg, args.episodes, args.seed, TrainingLog(out, 12))
        torch.save(model.state_dict(), out / "final_model.pt")
    model.eval()
    summaries, traces = evaluator.evaluate(model)
    write_csv(out / "training.csv", logs)
    write_csv(out / f"{label}_monthly.csv", summaries)
    write_csv(out / f"{label}_hourly.csv", traces)
    annual = annual_rows(summaries)
    write_csv(out / f"{label}_annual.csv", annual)
    for row in annual:
        name = "rl_best(ref)" if args.stage == "develop" and row["policy"] == "rl" else row["policy"]
        print(f"{row['scenario']:7s} {name:16s} bill savings={row['bill_saving_won']:>14,.0f} "
              f"operating net (excl. CAPEX)={row['net_saving_won']:>14,.0f} KRW")
    if args.stage == "develop":
        for scenario in SCENARIOS:
            print(f"{scenario:7s} {'rl_last':16s} operating net (excl. CAPEX)="
                  f"{extra[scenario]['rl_last_net_won']:>14,.0f} KRW")
        print(f"Reference best checkpoint: episode {extra['steady']['best_episode']}. Report the last model.")
    return annual, extra


if __name__ == "__main__":
    main()
