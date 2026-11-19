"""이번 시간 사용량 예측기: 선형 분위수 회귀 (방식 B).

매시간 시작 시점에 t−1까지 완료된 구간만으로 이번 구간 사용량의 τ 분위수(기본 0.95)를 예측한다.
학습은 Train 연도(2012)로만 한다. 근거·실험: 팀 공유 문서 "예측기 선택".

  예측(τ) = w0 + w1·L(t−1) + w2·p(t) + w3·(L(t−1) − p(t−1))
    L(t−1) : 직전 완료 구간 사용량
    p(t)   : Train 연도의 (하루 유형 × 시각) 평균 사용량. 하루 유형 = 평일 / 토요일 / 일요일·공휴일
    L(t−1) − p(t−1) : 오늘이 평소보다 얼마나 높은가
  계수는 핀볼 손실 Σ max(τ·u, (τ−1)·u), u = 실제 − 예측 을 최소화해 정한다 (선형계획, scipy HiGHS).
"""
from datetime import timedelta

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity


def day_type(date, cfg):
    """0 평일, 1 토요일, 2 일요일·공휴일 (config.json의 공휴일 목록)."""
    if date.weekday() == 6 or date.date().isoformat() in cfg["holidays"]:
        return 2
    return 1 if date.weekday() == 5 else 0


class QuantileForecaster:
    START = 168          # 학습 행렬은 첫 주 이후부터 만든다 (sandbox 실험과 같은 구간)

    def __init__(self, train_months, cfg, tau=0.95):
        self.cfg, self.tau = cfg, float(tau)
        rows = [r for m in sorted(train_months) for r in train_months[m]]
        dates = [d for d, _ in rows]
        load = np.array([x for _, x in rows], dtype=float)
        dt = np.array([day_type(d, cfg) for d in dates])
        hr = np.array([d.hour for d in dates])
        self.profile = np.zeros((3, 24))
        for k in range(3):
            for h in range(24):
                sel = (dt == k) & (hr == h)
                self.profile[k, h] = load[sel].mean() if sel.any() else load[hr == h].mean()
        t = np.arange(self.START, len(load))
        p, p1 = self.profile[dt[t], hr[t]], self.profile[dt[t - 1], hr[t - 1]]
        X = np.column_stack([np.ones(len(t)), load[t - 1], p, load[t - 1] - p1])
        self.w = self._fit(X, load[t])
        pred = X @ self.w
        self.report = {"tau": self.tau, "train_coverage": float(np.mean(load[t] <= pred)),
                       "weights": [float(x) for x in self.w]}

    def _fit(self, X, y):
        n, k = X.shape
        sc = X.std(0); sc[sc == 0] = 1.0
        A = hstack([csr_matrix(X / sc), identity(n), -identity(n)]).tocsr()
        c = np.concatenate([np.zeros(k), self.tau * np.ones(n), (1 - self.tau) * np.ones(n)])
        res = linprog(c, A_eq=A, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * n), method="highs")
        if not res.success:
            raise RuntimeError(f"Quantile regression failed: {res.message}")
        return res.x[:k] / sc

    def _p(self, date):
        return self.profile[day_type(date, self.cfg), date.hour]

    def predict(self, rows, i):
        """rows[i] 구간의 τ 분위수 예측. rows[i]의 사용량은 읽지 않는다 (시각만 사용)."""
        date = rows[i][0]
        prev = date - timedelta(hours=1)
        p, p1 = self._p(date), self._p(prev)
        # 월초에는 직전 구간이 에피소드 밖이므로 평균 패턴으로 대신한다 (0으로 채우면 크게 틀린다)
        l1 = rows[i - 1][1] if i > 0 else p1
        return float(max(0.0, self.w @ np.array([1.0, l1, p, l1 - p1])))
