import json
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from annual_env import AnnualESSEnv
from forecaster import QuantileForecaster
from run import forecasts_for_year
from tariff_engine import month_index, ratchet_history


class AnnualRatchetTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).parent / "config.json").read_text())
        self.cfg["self_discharge_per_day"] = 0
        self.jan = datetime(2012, 1, 31, 22)  # 계량 대상 평일 겨울 최대부하
        self.feb = datetime(2012, 2, 1, 10)
        self.months = {1: [(self.jan, 100)], 2: [(self.feb, 50)]}

    def env(self, opening=None):
        return AnnualESSEnv(self.months, self.cfg, train_max=120, forecasts=[100, 50], opening_history=opening)

    def test_energy_and_independent_ratchet_peaks_carry_to_next_month(self):
        env = self.env()
        state = env.reset()
        self.assertEqual(state.shape, (env.obs_dim,))
        _, _, done, january = env.step(4)
        self.assertFalse(done)
        self.assertAlmostEqual(env.energy / env.capacity, january["soc"])
        self.assertAlmostEqual(january["settlement_won"], 0.0)
        _, _, done, february = env.step(2)
        self.assertTrue(done)
        self.assertEqual(february["month_peak_kw"], 50)
        self.assertEqual(february["base_billing_peak_kw"], 100)
        self.assertAlmostEqual(february["billing_peak_kw"], january["grid_kw"])
        self.assertAlmostEqual(february["demand_won"], january["grid_kw"] * 8320)
        self.assertGreater(february["settlement_won"], 0)
        summary = env.summary()
        self.assertAlmostEqual(summary["reward_sum_won"], summary["net_saving_won"] - summary["penalty_won"])

    def test_rolling_window_expiry_and_seasonal_filter(self):
        history = {month_index(datetime(2012, m, 1)): p for m, p in ((7, 150), (8, 120), (10, 900))}
        self.assertEqual(max(ratchet_history(history, month_index(datetime(2013, 6, 1)))), 150)
        self.assertEqual(max(ratchet_history(history, month_index(datetime(2013, 7, 1)))), 120)
        self.assertEqual(max(ratchet_history(history, month_index(datetime(2013, 8, 1)))), 0)

    def test_shared_opening_history_and_soc_masks(self):
        env = self.env({month_index(datetime(2011, 12, 1)): 150})
        env.reset()
        self.assertEqual(env.peak, env.base_peak)
        self.assertEqual(env.peak, 150)
        np.testing.assert_array_equal(env.action_mask(), np.ones(6, dtype=bool))
        env.energy = env.capacity * self.cfg["soc_max"]
        np.testing.assert_array_equal(env.action_mask(), [False, False, True, True, True, True])
        env.energy = env.capacity * self.cfg["soc_min"]
        np.testing.assert_array_equal(env.action_mask(), [True, True, True, False, False, False])

    def test_current_hour_load_is_excluded_from_forecast_and_observation(self):
        forecaster = QuantileForecaster.__new__(QuantileForecaster)
        forecaster.cfg = self.cfg
        forecaster.profile = np.full((3, 24), 80.0)
        forecaster.w = np.array([0.0, 1.0, 0.0, 0.0])
        changed = {1: self.months[1], 2: [(self.feb, 500)]}
        envs = [AnnualESSEnv(months, self.cfg, 120, forecasts_for_year(months, forecaster))
                for months in (self.months, changed)]
        states = []
        for env in envs:
            env.reset()
            states.append(env.step(2)[0])
        np.testing.assert_array_equal(states[0], states[1])
        self.assertAlmostEqual(envs[0].forecasts[1], 100.0)

    def test_optional_defense_target_changes_only_requested_power(self):
        env = self.env()
        env.reset()
        _, _, _, info = env.step(5, peak_target_kw=98)
        self.assertEqual(info["requested_kw"], -2)
        self.assertEqual(info["discharge_kw"], 2)
        self.assertEqual(info["peak_defense_target_kw"], 98)
        env.reset()
        _, _, _, default = env.step(5)
        self.assertEqual(default["requested_kw"], -env.power)
