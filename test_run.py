"""두 단계 평가: year1 / steady 이력과 2014 달력."""
import json
import unittest
from datetime import datetime
from pathlib import Path

from run import Evaluator, STAGES, run_policy
from tariff_engine import month_index, tariff

ROOT = Path(__file__).resolve().parent


class TwoStageTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.cfg["self_discharge_per_day"] = 0
        # 직전 해 12월 겨울 최대부하 100 kW, 평가 연도 1월 최대부하 90 kW
        self.dec, self.jan = datetime(2012, 12, 31, 22), datetime(2013, 1, 31, 22)
        self.prior = {(2012, 12): [(self.dec, 100.0)]}
        self.target = {(2013, 1): [(self.jan, 90.0)]}
        forecast_at = {self.dec: 100.0, self.jan: 90.0}
        limit_at = {self.dec: None, self.jan: None}
        self.evaluator = Evaluator(self.cfg, 120, self.target, self.prior, None, forecast_at, limit_at)

    def test_year1_keeps_original_prior_peak_and_steady_uses_own_prior_peaks(self):
        year1 = self.evaluator.run("year1", "rule_tou").summary()
        self.assertAlmostEqual(year1["demand_saving_won"], 0.0)
        own = self.evaluator.steady_opening("rule_tou")
        prior_grid = own[month_index(self.dec)]
        self.assertLess(prior_grid, 100.0)
        steady = self.evaluator.run("steady", "rule_tou")
        self.assertEqual(steady.base_peak, 100.0)
        self.assertAlmostEqual(steady.summary()["demand_saving_won"], (100.0 - prior_grid) * self.cfg["demand_rate"])

    def test_steady_warmup_starts_from_virtual_previous_year_when_no_earlier_data(self):
        self.assertEqual(self.evaluator.prior_opening, {month_index(datetime(2011, 12, 1)): 100.0})

    def test_evaluate_adds_original_rows_per_scenario(self):
        summaries, _ = self.evaluator.evaluate(None, policies=("rule_tou",), collect_traces=False)
        self.assertEqual(sorted((r["scenario"], r["policy"]) for r in summaries),
                         [("steady", "original"), ("steady", "rule_tou"), ("year1", "original"), ("year1", "rule_tou")])

    def test_develop_never_reaches_test_year(self):
        develop, final = STAGES["develop"], STAGES["final"]
        self.assertLess(max(develop["train_years"]), develop["evaluate_year"])
        self.assertEqual(develop["evaluate_year"], 2013)
        self.assertEqual((final["train_years"], final["evaluate_year"]), ((2012, 2013), 2014))

    def test_2014_holidays_are_off_peak(self):
        for day in (datetime(2014, 6, 4, 14), datetime(2014, 9, 10, 14), datetime(2014, 5, 6, 14)):
            self.assertEqual(tariff(day, self.cfg)[1], 0, day)
        self.assertEqual(tariff(datetime(2014, 6, 5, 14), self.cfg)[1], 2)


if __name__ == "__main__":
    unittest.main()
