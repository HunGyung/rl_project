"""연간 ESS 에피소드. 시간 시작에 결정하고, 월마다 래칫 기본요금을 정산한다."""
from collections import defaultdict
import math

import numpy as np

from ess_env import dispatch
from tariff_engine import month_index, price_parts, ratchet_history, tariff


class AnnualESSEnv:
    obs_dim = 50
    n_actions = 6

    def __init__(self, months, cfg, train_max, forecasts, opening_history=None):
        self.rows = tuple(row for month in sorted(months) for row in months[month])
        if not self.rows or len(forecasts) != len(self.rows):
            raise ValueError("Each hour requires a load and a forecast")
        self.forecasts = np.asarray(forecasts, dtype=float)
        if not np.all(np.isfinite(self.forecasts)) or np.any(self.forecasts < 0):
            raise ValueError("Forecast values must be finite and nonnegative")
        self.cfg, self.scale = cfg, float(train_max)
        self.power = self.scale * cfg["power_fraction"]
        self.capacity = self.power * cfg["duration_hours"]
        self.eta = math.sqrt(cfg["round_trip_efficiency"])
        self.floor = self.scale * cfg["contract_multiplier"] * cfg["demand_floor_fraction"]
        self.price_scale = max(max(rates) for rates in cfg["rates"].values()) + cfg["surcharge_per_kwh"]
        self.tariffs = tuple(tariff(date, cfg) for date, _ in self.rows)
        first_month = month_index(self.rows[0][0])
        self.opening_history = {m: p for m, p in (opening_history or {}).items()
                                if first_month - 11 <= m < first_month}

    def reset(self):
        self.i = 0
        self.energy = self.capacity * self.cfg["initial_soc"]
        self.base_history = dict(self.opening_history)
        self.ess_history = dict(self.opening_history)
        self.history = []
        self._start_month()
        return self.observation()

    def _start_month(self):
        current = month_index(self.rows[self.i][0])
        self.base_carry = ratchet_history(self.base_history, current)
        self.ess_carry = ratchet_history(self.ess_history, current)
        self.base_peak = max(self.floor, *self.base_carry)
        self.peak = max(self.floor, *self.ess_carry)
        self.base_raw_peak = self.ess_raw_peak = 0.0
        self.month_start = self.i

    def action_mask(self):
        """상한의 충전과 하한의 방전을 제외. 대기는 항상 유효하다."""
        mask = np.ones(self.n_actions, dtype=bool)
        soc = self.energy / self.capacity
        if soc >= self.cfg["soc_max"] - 1e-9:
            mask[:2] = False
        if soc <= self.cfg["soc_min"] + 1e-9:
            mask[3:] = False
        return mask

    def observation(self):
        if self.i == len(self.rows):
            return np.zeros(self.obs_dim, dtype=np.float32)
        date = self.rows[self.i][0]
        price, band, season = self.tariffs[self.i]
        recent = [self.rows[self.i - k][1] / self.scale if self.i >= k else 0.0 for k in range(1, 5)]
        today = self.rows[max(0, self.i - date.hour):self.i]
        day_mean = sum(load for _, load in today) / (len(today) * self.scale) if today else 0.0
        hour_angle = 2 * math.pi * date.hour / 24
        month_angle = 2 * math.pi * (date.month - 1) / 12
        state = [math.sin(hour_angle), math.cos(hour_angle)]
        state += [float(date.weekday() == k) for k in range(7)]
        state += [float(season == k) for k in range(3)]
        state += [(len(self.rows) - self.i) / len(self.rows), float(band > 0), *recent]
        state += [day_mean, self.peak / self.scale, self.energy / self.capacity, price / self.price_scale]
        state += [self.forecasts[self.i] / self.scale, self.ess_raw_peak / self.scale,
                  self.base_peak / self.scale, self.base_raw_peak / self.scale]
        state += [peak / self.scale for peak in (*self.ess_carry, *self.base_carry)]
        state += [math.sin(month_angle), math.cos(month_angle)]
        return np.asarray(state, dtype=np.float32)

    def step(self, action, *, peak_target_kw=None):
        if self.i >= len(self.rows):
            raise RuntimeError("Episode ended; call reset")
        if action not in range(self.n_actions):
            raise ValueError("Action must be in 0..5")
        date = self.rows[self.i][0]
        cfg = self.cfg
        price, band, _ = self.tariffs[self.i]
        forecast = self.forecasts[self.i]
        target = self.peak if peak_target_kw is None else peak_target_kw
        requests = (self.power, self.power / 2, 0, -self.power / 2, -self.power,
                    -min(self.power, max(0, forecast - target)))
        request = requests[action]
        # 실측 부하는 행동 요청을 결정한 다음 물리 한도와 요금에만 사용한다.
        load = self.rows[self.i][1]
        before = self.energy / self.capacity
        charge, discharge, self.energy = dispatch(request, load, self.energy, self.power, self.capacity, self.eta, cfg)
        grid = load + charge - discharge
        base_previous, previous = self.base_peak, self.peak
        if band > 0:
            self.base_raw_peak = max(self.base_raw_peak, load)
            self.ess_raw_peak = max(self.ess_raw_peak, grid)
            self.base_peak = max(self.base_peak, load)
            self.peak = max(self.peak, grid)

        # 첫 시간에는 이월/하한 기본요금을, 이후에는 요금 적용 피크 증가분을 반영한다.
        first_hour = self.i == self.month_start
        rate = cfg["demand_rate"]
        base_demand = rate * (self.base_peak - (0 if first_hour else base_previous))
        demand = rate * (self.peak - (0 if first_hour else previous))
        degradation = cfg["degradation_per_kwh"] * discharge
        clipped = abs(request - (charge - discharge))
        penalty = cfg["clip_penalty_per_kwh"] * clipped
        self.i += 1
        done = self.i == len(self.rows)
        settlement = 0.0
        if done:
            missing = max(0, cfg["initial_soc"] * self.capacity - self.energy)
            settlement = missing / self.eta * sum(price_parts(date, cfg, 0))
        reward = price * (load - grid) + base_demand - demand - degradation - penalty - settlement
        info = dict(timestamp=date.isoformat(), month=date.month, action=int(action), forecast_kw=forecast,
                    peak_defense_target_kw=target if action == 5 else None,
                    load_kw=load, grid_kw=grid, requested_kw=request, charge_kw=charge, discharge_kw=discharge,
                    soc_before=before, soc=self.energy / self.capacity, measured=int(band > 0), price=price,
                    base_month_peak_kw=self.base_raw_peak, month_peak_kw=self.ess_raw_peak,
                    base_billing_peak_kw=self.base_peak, billing_peak_kw=self.peak,
                    base_energy_won=price * load, energy_won=price * grid, base_demand_won=base_demand,
                    demand_won=demand, degradation_won=degradation, settlement_won=settlement,
                    clipped_kw=clipped, penalty_won=penalty, reward_won=reward)
        self.history.append(info)
        if done or month_index(self.rows[self.i][0]) != month_index(date):
            self.ess_history[month_index(date)] = self.ess_raw_peak
            self.base_history[month_index(date)] = self.base_raw_peak
            if not done:
                self._start_month()
        return self.observation(), reward / cfg["reward_scale"], done, info

    def monthly_summaries(self):
        monthly = defaultdict(list)
        for row in self.history:
            monthly[row["month"]].append(row)
        return [dict(month=month, base_month_peak_kw=rows[-1]["base_month_peak_kw"],
                     month_peak_kw=rows[-1]["month_peak_kw"],
                     base_billing_peak_kw=rows[-1]["base_billing_peak_kw"],
                     billing_peak_kw=rows[-1]["billing_peak_kw"], **self._summary(rows))
                for month, rows in monthly.items()]

    def summary(self):
        return self._summary(self.history)

    @staticmethod
    def _summary(rows):
        totals = {key: sum(row[key] for row in rows) for key in (
            "base_energy_won", "energy_won", "base_demand_won", "demand_won", "degradation_won",
            "settlement_won", "penalty_won", "reward_won", "charge_kw", "discharge_kw")}
        base_bill = totals["base_energy_won"] + totals["base_demand_won"]
        bill = totals["energy_won"] + totals["demand_won"]
        saving = base_bill - bill
        return dict(base_bill_won=base_bill, bill_won=bill, bill_saving_won=saving,
                    energy_saving_won=totals["base_energy_won"] - totals["energy_won"],
                    demand_saving_won=totals["base_demand_won"] - totals["demand_won"],
                    degradation_won=totals["degradation_won"], settlement_won=totals["settlement_won"],
                    penalty_won=totals["penalty_won"], reward_sum_won=totals["reward_won"],
                    net_saving_won=saving - totals["degradation_won"] - totals["settlement_won"],
                    charge_kwh=totals["charge_kw"], discharge_kwh=totals["discharge_kw"],
                    clipped_steps=sum(row["clipped_kw"] > 1e-8 for row in rows), final_soc=rows[-1]["soc"])
