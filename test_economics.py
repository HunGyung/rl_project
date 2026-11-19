import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import economics as eco
from add_economics import case_rows


class EconomicsTests(unittest.TestCase):
    def test_reproduces_pnnl_totals_at_1mw(self):
        for (hours, year), total in eco.PNNL_TOTAL_USD_PER_KWH.items():
            size = eco.Size(1000.0, 1000.0 * hours)
            self.assertAlmostEqual(eco.capex_usd(size, year) / size.nominal_kwh, total, delta=0.6)

    def test_short_duration_uses_two_hour_price(self):
        self.assertEqual(eco.unit_costs(1.0), eco.unit_costs(2.0))

    def test_no_replacement_when_combined_life_exceeds_horizon(self):
        size = eco.Size(378.0, 756.0)
        result = eco.economics(size, 2e7, 10 * size.usable_kwh)
        self.assertAlmostEqual(result["battery_life_yr"], 1 / (10 / eco.CYCLE_LIFE + 1 / eco.CALENDAR_LIFE))
        self.assertGreater(result["battery_life_yr"], 15)
        self.assertEqual(result["replacement_years"], [])

    def test_cycle_and_calendar_ageing_add_up(self):
        size = eco.Size(100.0, 200.0)
        idle = eco.battery_life_yr(size, 0.0)
        cycling = eco.battery_life_yr(size, 250 * size.usable_kwh)
        self.assertEqual(idle, eco.CALENDAR_LIFE)
        self.assertAlmostEqual(cycling, 1 / (250 / 3000 + 1 / 16))
        result = eco.economics(size, 5e6, 250 * size.usable_kwh)
        self.assertEqual(result["replacement_years"], [7, 14])

    def test_degradation_rate_matches_cycle_part_of_replacement(self):
        u = eco.unit_costs(4.0)
        expected = (u["storage_block"] + u["storage_bos"]) * eco.USD_TO_KRW / (3000 * 0.8)
        self.assertAlmostEqual(eco.degradation_krw_per_dc_kwh(4.0), expected)
        self.assertAlmostEqual(eco.degradation_krw_per_dc_kwh(4.0), 107.1, delta=0.1)
        self.assertAlmostEqual(eco.degradation_krw_per_dc_kwh(4.0, cycle_life=6000), 53.6, delta=0.1)
        self.assertEqual(eco.degradation_krw_per_dc_kwh(1.0), eco.degradation_krw_per_dc_kwh(2.0))

    def test_ratio_matches_hand_calculation_without_replacement(self):
        size = eco.Size(100.0, 200.0)
        saving = 5e6
        result = eco.economics(size, saving, 0.0)
        u = eco.unit_costs(2.0)
        cap = eco.capex_usd(size) * eco.USD_TO_KRW
        fom = u["fom"] * 100 * eco.USD_TO_KRW
        pv = sum((saving - fom) / 1.05 ** t for t in range(1, 16))
        pv -= u["recycle"] * 200 * eco.USD_TO_KRW / 1.05 ** 15
        self.assertAlmostEqual(result["breakeven_ratio"], pv / cap, places=9)

    def test_first_year_saving_only_changes_year_one(self):
        size = eco.Size(100.0, 200.0)
        steady = eco.economics(size, 5e6, 0.0)
        first = eco.economics(size, 5e6, 0.0, first_year_bill_saving_krw=2e6)
        self.assertAlmostEqual(steady["pv_net_flows_krw"] - first["pv_net_flows_krw"], 3e6 / 1.05)
        self.assertEqual(steady["simple_payback_yr"], first["simple_payback_yr"])

    def test_case_rows_use_year1_then_steady(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "p05_h2"
            case.mkdir()
            settings = dict(power_kw=378.0, capacity_kwh=756.0, config=dict(round_trip_efficiency=0.83))
            (case / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
            with (case / "test_annual.csv").open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["scenario", "policy", "bill_saving_won", "net_saving_won", "discharge_kwh"])
                w.writeheader()
                w.writerow(dict(scenario="year1", policy="rl", bill_saving_won=1e7, net_saving_won=0.9e7, discharge_kwh=5000))
                w.writerow(dict(scenario="steady", policy="rl", bill_saving_won=2e7, net_saving_won=1.9e7, discharge_kwh=6000))
            rows = case_rows(case)
            expected = eco.economics(eco.Size(378.0, 756.0), 2e7, 6000 / math.sqrt(0.83), first_year_bill_saving_krw=1e7)
            self.assertAlmostEqual(rows[0]["breakeven_ratio"], round(expected["breakeven_ratio"], 4))
            self.assertEqual((rows[0]["year1_bill_saving_won"], rows[0]["bill_saving_won"]), (10_000_000, 20_000_000))

    def test_case_rows_use_bill_saving_and_nominal_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "p05_h2"
            case.mkdir()
            settings = dict(power_kw=378.0, capacity_kwh=756.0, config=dict(round_trip_efficiency=0.83))
            (case / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
            with (case / "validation_annual.csv").open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["policy", "bill_saving_won", "net_saving_won", "discharge_kwh"])
                w.writeheader()
                w.writerow(dict(policy="original", bill_saving_won=0, net_saving_won=0, discharge_kwh=0))
                w.writerow(dict(policy="rl", bill_saving_won=2e7, net_saving_won=1.9e7, discharge_kwh=6000))
            rows = case_rows(case)
            self.assertEqual([r["policy"] for r in rows], ["rl"])
            expected = eco.economics(eco.Size(378.0, 756.0), 2e7, 6000 / math.sqrt(0.83))
            self.assertAlmostEqual(rows[0]["breakeven_ratio"], round(expected["breakeven_ratio"], 4))
            self.assertAlmostEqual(rows[0]["usable_kwh"], 604.8)


if __name__ == "__main__":
    unittest.main()
