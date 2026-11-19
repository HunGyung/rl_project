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

    def test_no_replacement_when_calendar_life_exceeds_horizon(self):
        size = eco.Size(378.0, 756.0)
        result = eco.economics(size, 2e7, 10 * size.usable_kwh)
        self.assertEqual(result["battery_life_yr"], eco.CALENDAR_LIFE)
        self.assertEqual(result["replacement_years"], [])

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
