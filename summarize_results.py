"""여러 실행 폴더를 발표용 표 하나로 모은다. 재학습 없음.

  python summarize_results.py outputs/develop_A_... outputs/develop_A_... --output outputs/results_summary.csv

폴더를 여러 개 주거나, 실행 폴더들이 들어 있는 상위 폴더를 주면 된다.
단계 · 사이클 수명 · 공장 · 칸별로 한 줄을 만든다. 보고 기준은 다음과 같다.
  - RL은 마지막 모델의 시드 평균 · 최소 · 최대. 최고 체크포인트는 참고 열이다.
  - 금액은 설치비를 뺀 값이 아니라 연간 운영 순절감이다. 칸 사이 판단은 본전 설치비율로 한다.
  - promising = 공장 안에서 평균 본전 설치비율이 가장 높은 칸과 시드 범위가 겹치는 칸 (유망 구간).
    시드 차이로 구분되지 않는 칸을 '최적 1칸'으로 말하지 않기 위한 표시이다.
    add_economics.py를 먼저 실행해야 계산되고, 그 전에는 비워 둔다.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

SCENARIOS = ("year1", "steady")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run_dirs(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if (path / "grid_summary.csv").exists():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(p.parent for p in path.glob("*/grid_summary.csv")))
    unique = list(dict.fromkeys(found))
    if not unique:
        raise SystemExit("No run folder with grid_summary.csv found")
    return unique


def load_runs(paths: list[Path]) -> dict:
    """{(단계, 평가 연도, 사이클 수명, 에피소드, 공장, 칸): {시드: {상황: 행, 'economics': {정책: 행}}}}"""
    cells = defaultdict(dict)
    for run in run_dirs(paths):
        economics = defaultdict(dict)
        if (run / "grid_economics.csv").exists():
            for row in read_csv(run / "grid_economics.csv"):
                economics[row["condition"]][row["policy"]] = row
        for row in read_csv(run / "grid_summary.csv"):
            if "cycle_life" not in row or "rl_last_net_won" not in row:
                raise SystemExit(f"{run}: 이전 형식의 결과입니다. 현재 코드로 다시 실행해 주세요.")
            key = (row["stage"], row["evaluate_year"], float(row["cycle_life"]), row["episodes"],
                   row["factory"], row["condition"])
            seed = int(row["seed"])
            entry = cells[key].setdefault(seed, {"economics": economics.get(row["condition"], {}), "run": run})
            if entry["run"] != run:
                raise SystemExit(f"Duplicate seed {seed} for {key}: {entry['run']} and {run}")
            entry[row["scenario"]] = row
    return cells


def stats(values: list[float]) -> tuple[float, float, float]:
    return mean(values), min(values), max(values)


def summarize(cells: dict) -> list[dict]:
    rows = []
    for key in sorted(cells):
        stage, year, cycle_life, episodes, factory, condition = key
        seeds = cells[key]
        first = seeds[min(seeds)]["steady"]
        row = dict(stage=stage, evaluate_year=year, cycle_life=f"{cycle_life:g}", factory=factory, condition=condition,
                   power_percent=first["power_percent"], duration_hours=first["duration_hours"], episodes=episodes,
                   n_seeds=len(seeds), seeds=" ".join(str(s) for s in sorted(seeds)))
        for scenario in SCENARIOS:
            data = [seeds[s][scenario] for s in sorted(seeds)]
            last = [float(r["rl_last_net_won"]) for r in data]
            tou, percentile = float(data[0]["rule_tou_net_won"]), float(data[0]["rule_percentile_net_won"])
            row[f"{scenario}_rl_last_mean_won"], row[f"{scenario}_rl_last_min_won"], row[f"{scenario}_rl_last_max_won"] = stats(last)
            row[f"{scenario}_rule_tou_won"], row[f"{scenario}_rule_percentile_won"] = tou, percentile
            row[f"{scenario}_seeds_rl_last_above_rules"] = sum(value > max(tou, percentile) for value in last)
        best = [float(seeds[s]["steady"]["rl_best_net_won"]) for s in sorted(seeds) if seeds[s]["steady"].get("rl_best_net_won")]
        row["reference_steady_rl_best_mean_won"] = mean(best) if best else ""
        ratios = [float(seeds[s]["economics"]["rl"]["breakeven_ratio"]) for s in sorted(seeds) if "rl" in seeds[s]["economics"]]
        complete = len(ratios) == len(seeds)
        row["rl_last_breakeven_mean"], row["rl_last_breakeven_min"], row["rl_last_breakeven_max"] = (
            stats(ratios) if complete else ("", "", ""))
        for policy in ("rule_tou", "rule_percentile"):
            econ = seeds[min(seeds)]["economics"].get(policy)
            row[f"{policy}_breakeven"] = float(econ["breakeven_ratio"]) if econ else ""
        row["promising"] = ""
        rows.append(row)
    mark_promising(rows)
    return rows


def mark_promising(rows: list[dict]) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["evaluate_year"], row["cycle_life"], row["episodes"], row["factory"])].append(row)
    for group in groups.values():
        if any(row["rl_last_breakeven_mean"] == "" for row in group):
            continue
        top = max(group, key=lambda row: row["rl_last_breakeven_mean"])
        for row in group:
            row["promising"] = int(row["rl_last_breakeven_max"] >= top["rl_last_breakeven_min"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="Run folders, or a folder that contains run folders")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "results_summary.csv")
    args = parser.parse_args()
    rows = summarize(load_runs(args.runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {args.output} ({len(rows)} cells)")
    few = sorted({row["factory"] for row in rows if row["n_seeds"] < 3})
    if few:
        print(f"Warning: fewer than 3 seeds for factory {', '.join(few)}")


if __name__ == "__main__":
    main()
