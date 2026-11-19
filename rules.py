"""완료된 일별 부하 이력으로 분위수 방전 기준을 계산한다."""
from datetime import timedelta

import numpy as np


def percentile_thresholds(rows, prior_rows=(), *, lookback_days=7, quantile=0.95, min_days=3):
    """각 시간대의 기준값은 그날 자정 전에 끝난 동일 시간대 구간만 사용한다."""
    history = dict(prior_rows)
    thresholds = []
    for date, load in rows:
        previous_dates = (date - timedelta(days=days) for days in range(1, lookback_days + 1))
        samples = [history[previous] for previous in previous_dates if previous in history]
        thresholds.append(float(np.quantile(samples, quantile)) if len(samples) >= min_days else None)
        history[date] = load
    return thresholds
