"""두 단계 평가: year1 / steady 이력, 결과 집계와 2014 달력."""
import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from run import Evaluator, STAGES
from summarize_results import load_runs, summarize
from tariff_engine import month_index, tariff

ROOT = Path(__file__).resolve().parent


class TwoStageTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.cfg["self_discharge_per_day"] = 0
        # 평가 연도 12월 겨울 최대부하 100 kW가 되풀이되는 다음 해 1월(90 kW)의 요금을 정한다
        self.prior_dec = datetime(2012, 12, 31, 22)
        self.jan, self.charge, self.dec = datetime(2013, 1, 31, 22), datetime(2013, 12, 31, 2), datetime(2013, 12, 31, 22)
        self.target = {(2013, 1): [(self.jan, 90.0)], (2013, 12): [(self.charge, 20.0), (self.dec, 100.0)]}
        self.evaluator = self.build(100.0)

    def build(self, prior_peak):
        prior = {(2012, 12): [(self.prior_dec, prior_peak)]}
        forecast_at = {self.prior_dec: prior_peak, self.jan: 90.0, self.charge: 20.0, self.dec: 100.0}
        limit_at = {date: None for date in forecast_at}
        return Evaluator(self.cfg, 120, self.target, prior, forecast_at, limit_at)

    def test_year1_keeps_original_prior_peak(self):
        january = self.evaluator.run("year1", "rule_tou").monthly_summaries()[0]
        self.assertEqual(january["billing_peak_kw"], 100.0)
        self.assertAlmostEqual(january["demand_saving_won"], 0.0)

    def test_steady_repeats_evaluation_year_with_own_peaks(self):
        env = self.evaluator.run("steady", "rule_tou")
        previous_december = month_index(datetime(2012, 12, 1))
        own = env.ess_opening[previous_december]
        self.assertEqual(env.base_opening[previous_december], 100.0)
        self.assertLess(own, 100.0)
        january = env.monthly_summaries()[0]
        self.assertAlmostEqual(january["demand_saving_won"], (100.0 - own) * self.cfg["demand_rate"])

    def test_steady_bill_does_not_carry_prior_year_peaks(self):
        changed = self.build(500.0)
        original = self.evaluator.run("steady", "rule_tou").summary()["net_saving_won"]
        self.assertAlmostEqual(changed.run("steady", "rule_tou").summary()["net_saving_won"], original)
        self.assertEqual(changed.run("year1", "rule_tou").monthly_summaries()[0]["billing_peak_kw"], 500.0)

    def test_evaluate_adds_original_rows_per_scenario(self):
        summaries, _ = self.evaluator.evaluate(None, policies=("rule_tou",), collect_traces=False)
        self.assertEqual({(r["scenario"], r["policy"]) for r in summaries},
                         {("steady", "original"), ("steady", "rule_tou"), ("year1", "original"), ("year1", "rule_tou")})

    def test_develop_never_reaches_test_year(self):
        develop, final = STAGES["develop"], STAGES["final"]
        self.assertLess(max(develop["train_years"]), develop["evaluate_year"])
        self.assertEqual(develop["evaluate_year"], 2013)
        self.assertEqual((final["train_years"], final["evaluate_year"]), ((2012, 2013), 2014))

    def test_2014_holidays_are_off_peak(self):
        for day in (datetime(2014, 6, 4, 14), datetime(2014, 9, 10, 14), datetime(2014, 5, 6, 14)):
            self.assertEqual(tariff(day, self.cfg)[1], 0, day)
        self.assertEqual(tariff(datetime(2014, 6, 5, 14), self.cfg)[1], 2)


class SummaryTests(unittest.TestCase):
    FIELDS = ["factory", "stage", "evaluate_year", "condition", "scenario", "power_percent", "duration_hours", "seed",
              "episodes", "cycle_life", "rule_percentile_net_won", "rule_tou_net_won", "original_net_won",
              "rl_last_net_won", "rl_best_net_won", "best_episode"]

    def write_run(self, root, seed, values):
        run = root / f"run{seed}"
        run.mkdir()
        with (run / "grid_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            for condition, (last, _) in values.items():
                for scenario in ("year1", "steady"):
                    writer.writerow(dict(factory="A", stage="develop", evaluate_year=2013, condition=condition,
                                         scenario=scenario, power_percent=5, duration_hours=2, seed=seed, episodes=240,
                                         cycle_life=5250, rule_percentile_net_won=1, rule_tou_net_won=20,
                                         original_net_won=0, rl_last_net_won=last, rl_best_net_won=last + 100,
                                         best_episode=120))
        with (run / "grid_economics.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["condition", "policy", "breakeven_ratio"])
            writer.writeheader()
            for condition, (_, ratio) in values.items():
                writer.writerow(dict(condition=condition, policy="rl", breakeven_ratio=ratio))

    def test_reports_last_model_range_and_overlapping_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root, 0, {"p03_h1": (10, 0.50), "p05_h2": (30, 0.70), "p07_h4": (5, 0.10)})
            self.write_run(root, 1, {"p03_h1": (50, 0.65), "p05_h2": (40, 0.60), "p07_h4": (7, 0.20)})
            rows = {row["condition"]: row for row in summarize(load_runs([root]))}
        self.assertEqual(rows["p03_h1"]["steady_rl_last_mean_won"], 30)
        self.assertEqual((rows["p03_h1"]["steady_rl_last_min_won"], rows["p03_h1"]["steady_rl_last_max_won"]), (10, 50))
        self.assertEqual(rows["p03_h1"]["steady_seeds_rl_last_above_rules"], 1)
        self.assertEqual(rows["p03_h1"]["reference_steady_rl_best_mean_won"], 130)
        self.assertEqual([rows[c]["promising"] for c in ("p03_h1", "p05_h2", "p07_h4")], [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
