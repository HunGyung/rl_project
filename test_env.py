"""물리 제약, 요금 회계, 시간 누출을 확인하는 핵심 테스트."""
import json
import unittest
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ess_env import ESSEnv, tariff

ROOT = Path(__file__).resolve().parent


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "config.json").read_text())
        self.rows = [(datetime(2012, 1, 1) + timedelta(hours=i), 100 + i % 17) for i in range(744)]

    def test_constraints_and_bill_identity(self):
        env = ESSEnv({1: self.rows}, self.cfg, 120)
        state = env.reset(1)
        self.assertEqual(state.shape, (22,))
        rng = np.random.default_rng(42)
        done = False
        while not done:
            _, _, done, info = env.step(int(rng.integers(6)))
            self.assertGreaterEqual(info["grid_kw"], -1e-9)
            self.assertGreaterEqual(info["soc"], 0.1 - 1e-9)
            self.assertLessEqual(info["soc"], 0.9 + 1e-9)
            self.assertLessEqual(info["charge_kw"], env.power + 1e-9)
            self.assertLessEqual(info["discharge_kw"], env.power + 1e-9)
            self.assertAlmostEqual(info["grid_kw"], info["load_kw"] + info["charge_kw"] - info["discharge_kw"])
        summary = env.summary()
        self.assertAlmostEqual(summary["reward_sum_won"], summary["net_saving_won"] - summary["penalty_won"], places=6)
        # 요금 계산을 시간별 증가분과 독립적으로 다시 계산한다.
        direct = sum(r["grid_kw"] * r["price"] for r in env.history)
        direct += self.cfg["demand_rate"] * max([env.floor] + [r["grid_kw"] for r in env.history if r["measured"]])
        self.assertAlmostEqual(summary["bill_won"], direct, places=6)

    def test_no_future_information_in_state(self):
        changed = deepcopy(self.rows)
        changed[1:] = [(d, x * 100) for d, x in changed[1:]]
        a, b = ESSEnv({1: self.rows}, self.cfg, 120), ESSEnv({1: changed}, self.cfg, 120)
        np.testing.assert_array_equal(a.reset(1), b.reset(1))

    def test_no_battery_action_zero_savings_without_self_discharge(self):
        self.cfg["self_discharge_per_day"] = 0
        env = ESSEnv({1: self.rows}, self.cfg, 120)
        env.reset(1)
        for _ in self.rows:
            _, reward, _, _ = env.step(2)
            self.assertEqual(reward, 0)
        self.assertEqual(env.summary()["net_saving_won"], 0)

    def test_efficiency_and_terminal_settlement(self):
        self.cfg["self_discharge_per_day"] = 0
        env = ESSEnv({1: self.rows[:1]}, self.cfg, 120)
        env.reset(1)
        _, _, done, info = env.step(4)
        self.assertTrue(done)
        self.assertAlmostEqual(info["discharge_kw"], 0.4 * env.capacity * env.eta)
        self.assertAlmostEqual(info["settlement_won"], 0.4 * env.capacity / env.eta * info["price"])

    def test_metering_calendar(self):
        self.assertEqual(tariff(datetime(2012, 1, 1, 11), self.cfg)[1], 0)
        self.assertEqual(tariff(datetime(2012, 1, 2, 11), self.cfg)[1], 2)
        self.assertEqual(tariff(datetime(2012, 1, 7, 11), self.cfg)[1], 1)
        self.cfg["holidays"] = ["2012-01-02"]
        self.assertEqual(tariff(datetime(2012, 1, 2, 11), self.cfg)[1], 0)

    def test_team_tariff(self):
        self.assertAlmostEqual(tariff(datetime(2012, 1, 2, 22), self.cfg)[0], 164.0)
        self.assertEqual(tariff(datetime(2012, 1, 2, 22), self.cfg)[1], 2)
        self.assertEqual(tariff(datetime(2013, 10, 9, 11), self.cfg)[1], 0)
        self.assertAlmostEqual(tariff(datetime(2012, 7, 2, 9), self.cfg)[0], 106.3)


if __name__ == "__main__":
    unittest.main()
