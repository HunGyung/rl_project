"""1시간 간격 ESS 환경. 전력 kW, 저장 에너지 kWh, 비용 원.

외부 RL 라이브러리 없이 reset/step 흐름을 직접 읽을 수 있게 구현한다.
요금 시간대와 단가는 실습 가정이며 실제 한전 청구 엔진이 아니다.
"""
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tariff_engine import tariff, price_parts


def load_year(path, year):
    """확인된 데이터 정의: load_01=00~01시 kWh. 요청한 연도만 검사하고 사용한다."""
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            day = datetime.strptime(row["date"], "%Y-%m-%d")
            if day.year != year:
                continue
            for hour in range(24):
                value = float(row[f"load_{hour + 1:02d}"])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid load: {day}, hour={hour}")
                rows.append((day + timedelta(hours=hour), value))
    start, end = datetime(year, 1, 1), datetime(year + 1, 1, 1)
    if len(rows) != int((end - start).total_seconds() / 3600):
        raise ValueError(f"Incomplete year: {year}")
    for i, (date, _) in enumerate(rows):
        if date != start + timedelta(hours=i):
            raise ValueError(f"Duplicate, unordered or missing timestamp: {date}")
    return {month: [(d, x) for d, x in rows if d.month == month]
            for month in range(1, 13)}


class ESSEnv:
    def __init__(self, months, cfg, train_max):
        self.months, self.cfg, self.scale = months, cfg, float(train_max)
        self.power = self.scale * cfg["power_fraction"]
        self.capacity = self.power * cfg["duration_hours"]
        if self.power <= 0 or self.capacity <= 0:
            raise ValueError("Battery power and capacity must be positive")
        self.eta = math.sqrt(cfg["round_trip_efficiency"])
        self.floor = self.scale * cfg["contract_multiplier"] * cfg["demand_floor_fraction"]
        self.price_scale = max(max(v) for v in cfg["rates"].values()) + cfg["surcharge_per_kwh"]

    def reset(self, month):
        self.rows = self.months[month]
        self.i = 0
        self.energy = self.capacity * self.cfg["initial_soc"]
        self.peak = self.base_peak = self.floor
        self.history = []
        return self.observation()

    def observation(self):
        if self.i == len(self.rows):
            return np.zeros(22, dtype=np.float32)
        date, load = self.rows[self.i]
        price, band, season = tariff(date, self.cfg)
        # 월초의 없는 이력은 첫 관측값으로 채운다. 미래 부하는 쓰지 않는다.
        past = [self.rows[max(0, self.i - k)][1] / self.scale for k in (1, 2, 3)]
        day_start = self.i - date.hour
        day_mean = np.mean([r[1] for r in self.rows[day_start:self.i + 1]]) / self.scale
        state = [math.sin(2 * math.pi * date.hour / 24), math.cos(2 * math.pi * date.hour / 24)]
        state += [float(date.weekday() == k) for k in range(7)]
        state += [float(season == k) for k in range(3)]
        state += [(len(self.rows) - self.i) / 744, float(band > 0), load / self.scale]
        state += past + [day_mean, self.peak / self.scale, self.energy / self.capacity, price / self.price_scale]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        if self.i >= len(self.rows):
            raise RuntimeError("Episode ended; call reset")
        if action not in range(6):
            raise ValueError("Action must be in 0..5")
        c = self.cfg
        date, load = self.rows[self.i]
        price, band, _ = tariff(date, c)
        measured = band > 0
        before = self.energy
        # 최소 SoC는 사용 가능 영역의 하한으로 유지한다.
        self.energy = max(c["soc_min"] * self.capacity,
                          self.energy * (1 - c["self_discharge_per_day"]) ** (1 / 24))
        request = [self.power, self.power / 2, 0, -self.power / 2, -self.power,
                   -min(self.power, max(0, load - self.peak))][action]
        if request >= 0:
            charge = min(request, self.power, max(0, (c["soc_max"] * self.capacity - self.energy) / self.eta))
            discharge = 0.0
        else:
            charge = 0.0
            discharge = min(-request, self.power, load, max(0, (self.energy - c["soc_min"] * self.capacity) * self.eta))
        actual = charge - discharge
        clipped = abs(request - actual)
        self.energy += charge * self.eta - discharge / self.eta
        grid = load + actual
        base_increase = max(0, load - self.base_peak) if measured else 0
        increase = max(0, grid - self.peak) if measured else 0
        base_cost = price * load + c["demand_rate"] * base_increase
        cost = price * grid + c["demand_rate"] * increase
        degradation = discharge * c["degradation_per_kwh"]
        penalty = clipped * c["clip_penalty_per_kwh"]
        if measured:
            self.base_peak = max(self.base_peak, load)
            self.peak = max(self.peak, grid)
        self.i += 1
        done = self.i == len(self.rows)
        # 발표안에 따라 월말 부족 에너지만 가상 매입 정산. 물리적 충전은 아님.
        settlement = 0.0
        if done:
            offpeak_price = sum(price_parts(date, c, 0))
            settlement = max(0, c["initial_soc"] * self.capacity - self.energy) / self.eta * offpeak_price
        reward_won = base_cost - cost - degradation - penalty - settlement
        info = dict(timestamp=date.isoformat(), action=int(action), load_kw=load, grid_kw=grid,
                    requested_kw=request, charge_kw=charge, discharge_kw=discharge,
                    soc_before=before / self.capacity, soc=self.energy / self.capacity,
                    peak_kw=self.peak, base_peak_kw=self.base_peak, price=price,
                    measured=int(measured), clipped_kw=clipped,
                    base_energy_won=price * load, energy_won=price * grid,
                    degradation_won=degradation, penalty_won=penalty,
                    settlement_won=settlement, reward_won=reward_won)
        self.history.append(info)
        return self.observation(), reward_won / c["reward_scale"], done, info

    def summary(self):
        h, c = self.history, self.cfg
        base_energy = sum(r["base_energy_won"] for r in h)
        energy = sum(r["energy_won"] for r in h)
        base_demand, demand = self.base_peak * c["demand_rate"], self.peak * c["demand_rate"]
        saving = base_energy + base_demand - energy - demand
        degradation = sum(r["degradation_won"] for r in h)
        settlement = sum(r["settlement_won"] for r in h)
        penalty = sum(r["penalty_won"] for r in h)
        return dict(base_bill_won=base_energy + base_demand, bill_won=energy + demand,
                    energy_saving_won=base_energy - energy, demand_saving_won=base_demand - demand,
                    bill_saving_won=saving, degradation_won=degradation,
                    settlement_won=settlement, penalty_won=penalty,
                    net_saving_won=saving - degradation - settlement,
                    reward_sum_won=sum(r["reward_won"] for r in h),
                    clipped_steps=sum(r["clipped_kw"] > 1e-8 for r in h),
                    final_soc=self.energy / self.capacity)
