import unittest
from datetime import datetime, timedelta

from rules import percentile_thresholds


class PercentileRuleTests(unittest.TestCase):
    def test_uses_only_prior_days_same_hour_in_rolling_window(self):
        date = datetime(2013, 1, 8, 10)
        prior = [(date - timedelta(days=d), float(8 - d)) for d in range(1, 8)]
        prior += [(date - timedelta(days=8), 10000), (date - timedelta(days=1, hours=1), 10000)]
        rows = [(date, 2000), (date + timedelta(days=1), 3000)]
        thresholds = percentile_thresholds(rows, prior)
        self.assertAlmostEqual(thresholds[0], 6.7)
        self.assertGreater(thresholds[1], 1000)
        self.assertEqual(percentile_thresholds([(date, 9000)], prior)[0], thresholds[0])

    def test_missing_history_leaves_threshold_unset(self):
        date = datetime(2013, 1, 8, 10)
        self.assertEqual(percentile_thresholds([(date, 10)], [(date - timedelta(days=1), 20)]), [None])
