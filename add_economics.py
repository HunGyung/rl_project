"""run.py 출력 폴더에 15년 경제성(설치비 · O&M · 본전 설치비율)을 붙인다. 재학습 없음.

  python add_economics.py outputs/<run 폴더>

칸 폴더(p03_h1 …)마다 settings.json(power_kw, capacity_kwh)과 validation_annual.csv(2013 검증)를 읽어
정책별로 economics.economics()를 계산하고 <run 폴더>/grid_economics.csv로 저장한다.

입력 규칙 (economics.py 맨 위 가정 참고)
  - 크기: capacity_kwh = 명목 용량 (kW × 지속시간). 실사용은 그 × 0.8
  - 절감액: bill_saving_won (래칫 기준 요금 절감). net_saving_won은 열화비를 이미 뺀 값이라 쓰면 이중 계산
  - 방전량: discharge_kwh(계통으로 나간 AC) ÷ √왕복효율 = 배터리에서 나간 DC (사이클 수 계산용)
유망 칸 선택: 칸 사이 비교는 breakeven_ratio(본전 설치비율)로 한다. 칸 안 체크포인트는 순절감 그대로.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import economics as eco

POLICIES = ("rule_tou", "rule_percentile", "rl")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def case_rows(case_dir: Path) -> list[dict]:
    settings = json.loads((case_dir / "settings.json").read_text(encoding="utf-8"))
    size = eco.size_from_run(settings["power_kw"], settings["capacity_kwh"])
    rte = settings["config"]["round_trip_efficiency"]
    rows = []
    for r in read_csv(case_dir / "validation_annual.csv"):
        if r["policy"] not in POLICIES:
            continue
        bill = float(r["bill_saving_won"])
        dc = float(r["discharge_kwh"]) / math.sqrt(rte)
        e = eco.economics(size, bill, dc)
        rows.append(dict(
            condition=case_dir.name, policy=r["policy"],
            power_kw=round(size.power_kw, 1), nominal_kwh=round(size.nominal_kwh, 1),
            usable_kwh=round(size.usable_kwh, 1),
            capex_won=round(e["capex_krw"]), annual_om_won=round(e["annual_fixed_om_krw"]),
            bill_saving_won=round(bill), net_saving_won=round(float(r["net_saving_won"])),
            equivalent_cycles=round(dc / (size.nominal_kwh * eco.DOD), 1),
            breakeven_ratio=round(e["breakeven_ratio"], 4), npv_won=round(e["npv_krw"]),
            simple_payback_yr=round(e["simple_payback_yr"], 1) if math.isfinite(e["simple_payback_yr"]) else "",
            breakeven_capex_usd_per_kwh=round(e["breakeven_capex_usd_per_kwh"], 1),
            pnnl_capex_usd_per_kwh=round(e["pnnl_capex_usd_per_kwh"], 1),
            battery_life_yr=round(e["battery_life_yr"], 1)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    cases = sorted(p for p in args.run_dir.iterdir()
                   if p.is_dir() and (p / "validation_annual.csv").exists() and (p / "settings.json").exists())
    if not cases:
        raise SystemExit(f"No case folders with validation_annual.csv in {args.run_dir}")
    rows = [row for case in cases for row in case_rows(case)]
    out = args.run_dir / "grid_economics.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out}")
    for policy in POLICIES:
        subset = [r for r in rows if r["policy"] == policy]
        if subset:
            top = sorted(subset, key=lambda r: r["breakeven_ratio"], reverse=True)[:3]
            print(f"{policy:16s} top 3 by breakeven_ratio: " +
                  ", ".join(f"{r['condition']} {r['breakeven_ratio']:.3f}" for r in top))


if __name__ == "__main__":
    main()
