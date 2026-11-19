"""LFP ESS 15년 경제성: 설치비 · 고정 O&M · 교체 · 폐기 → NPV · 본전 설치비율 · 회수기간.

운전 · 학습과는 연결하지 않는다. 시뮬레이션이 낸 연간 요금 절감액에 붙이는 사후 계산이다.

출처: PNNL (2022) "2022 Grid Energy Storage Technology Cost and Performance Assessment" (PNNL-33283)
      LFP 1 MW 열, 명목 2 · 4 · 10h, 2021 · 2030 항목별 단가 (USD). 성능은 PNNL 2021 대표값.
      사이클 수명: 표 4.2, 80% DoD에서 6,000회(수명 끝 = 용량 60%). 용량이 선형으로 준다고 보고
      용량 80%까지 = 3,000회를 기본으로 쓴다. PNNL 정의(용량 60%) 그대로인 6,000회는 민감도.
가정:
  - 1 MW 단가를 우리 kW · kWh에 비례 적용한다 (소형 설비 할증 없음 → 설치비를 낮게 본 쪽)
  - 단가 보간 축은 명목 지속시간(명목 kWh ÷ kW). 2h 미만은 2h 단가, 10h 초과는 10h 단가
  - 할인율 5%, 15년, 잔존가치 0, 환율 2021 연평균 1,144.4원/$
  - 배터리 노화는 설치비(0년)와 실제 교체 연도의 교체비로만 반영한다.
    시뮬레이션의 열화비(degradation_won)를 절감액에서 또 빼면 이중 계산이 된다 → 입력은 bill_saving_won
  - 노화 모델은 운전과 같다: 1년 노화 = 연간 사이클 ÷ 사이클 수명 + 1 ÷ 캘린더 수명 (두 몫을 더함).
    운전의 열화비(방전 DC kWh당 degradation_krw_per_dc_kwh)는 이 중 사이클 몫의 값이다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

USD_TO_KRW = 1144.4
YEARS = 15
DISCOUNT = 0.05

KWH_ITEMS = ("storage_block", "storage_bos", "system_integration", "epc", "project_dev")
KW_ITEMS = ("power_equipment", "controls_comm", "grid_integration")


def _row(sb, bos, pe, cc, si, epc, pd, gi, fom, recycle):
    return dict(storage_block=sb, storage_bos=bos, power_equipment=pe, controls_comm=cc,
                system_integration=si, epc=epc, project_dev=pd, grid_integration=gi,
                fom=fom, recycle=recycle)


# PNNL LFP 1 MW. kWh 항목 $/kWh · kW 항목 $/kW · fom $/kW-yr · recycle $/kWh (수명 끝 1회)
PNNL_LFP = {
    2: {2021: _row(184.94, 45.55, 84.65, 40.01, 56.56, 69.88, 83.85, 30.94, 3.17, 2.65),
        2030: _row(121.02, 34.01, 74.88, 29.87, 47.99, 59.29, 71.15, 26.25, 2.69, 0.00)},
    4: {2021: _row(182.27, 42.38, 84.65, 40.01, 50.16, 61.20, 73.43, 30.94, 5.05, 2.65),
        2030: _row(119.28, 31.64, 74.88, 29.87, 42.56, 51.92, 62.31, 26.25, 4.28, 0.00)},
    # 10h 재활용비는 표에서 읽지 못해 4h 값을 쓴다
    10: {2021: _row(178.80, 40.06, 84.65, 40.01, 45.87, 55.44, 66.53, 30.94, 10.59, 2.65),
         2030: _row(117.00, 29.92, 74.88, 29.87, 38.92, 47.04, 56.45, 26.25, 8.99, 0.00)},
}
# 검증용: PNNL 표의 총 설치비 ($/kWh, 1 MW)
PNNL_TOTAL_USD_PER_KWH = {(2, 2021): 519.0, (4, 2021): 448.34, (10, 2021): 402.25,
                          (2, 2030): 399.0, (4, 2030): 340.46, (10, 2030): 302.42}

# PNNL 2021 LFP 대표값 (왕복효율은 변압기까지 포함한 AC-AC). config.json의 효율 · SoC와 같다
CYCLE_LIFE = 3000          # DoD 80%에서 용량 80%까지 (PNNL 6,000회는 용량 60%까지 → 선형으로 절반)
CALENDAR_LIFE = 16         # 년
SOC_MIN, SOC_MAX = 0.1, 0.9
DOD = SOC_MAX - SOC_MIN


@dataclass(frozen=True)
class Size:
    power_kw: float
    nominal_kwh: float

    @property
    def usable_kwh(self) -> float:
        return self.nominal_kwh * DOD

    @property
    def nominal_hours(self) -> float:
        return self.nominal_kwh / self.power_kw


def size_from_run(power_kw: float, capacity_kwh: float) -> Size:
    """팀 환경의 capacity(= kW × 지속시간)는 명목 용량이다. 그 안에서 SoC 10~90%를 쓴다."""
    return Size(float(power_kw), float(capacity_kwh))


def unit_costs(nominal_hours: float, year: int = 2021) -> dict:
    keys = sorted(PNNL_LFP)
    h = min(max(nominal_hours, keys[0]), keys[-1])
    for lo, hi in zip(keys, keys[1:]):
        if lo <= h <= hi:
            w = (h - lo) / (hi - lo)
            a, b = PNNL_LFP[lo][year], PNNL_LFP[hi][year]
            return {k: a[k] + w * (b[k] - a[k]) for k in a}
    raise AssertionError


def capex_usd(size: Size, year: int = 2021) -> float:
    """설치비 = Σ kWh 항목 × 명목 kWh + Σ kW 항목 × kW."""
    u = unit_costs(size.nominal_hours, year)
    return (sum(u[k] for k in KWH_ITEMS) * size.nominal_kwh
            + sum(u[k] for k in KW_ITEMS) * size.power_kw)


def degradation_krw_per_dc_kwh(nominal_hours: float, cycle_life: float = CYCLE_LIFE, year: int = 2021,
                               fx: float = USD_TO_KRW) -> float:
    """배터리에서 나간 DC 1 kWh당 사이클 열화비(원) = 교체 단가(저장 블록 + BOS) ÷ (사이클 수명 × DoD)."""
    u = unit_costs(nominal_hours, year)
    return (u["storage_block"] + u["storage_bos"]) * fx / (cycle_life * DOD)


def battery_life_yr(size: Size, annual_discharge_dc_kwh: float, cycle_life: float = CYCLE_LIFE,
                    calendar_life: float = CALENDAR_LIFE) -> float:
    """교체 부분 수명. 사이클로 닳는 몫과 시간으로 닳는 몫을 더한다:
    1 ÷ 수명 = 연간 등가 사이클 ÷ 사이클 수명 + 1 ÷ 캘린더 수명."""
    cycles = annual_discharge_dc_kwh / (size.nominal_kwh * DOD)
    return 1.0 / (cycles / cycle_life + 1.0 / calendar_life)


def payback(cap: float, flows: list[float]) -> float | None:
    """누적 현금흐름이 설치비를 넘는 시점(년, 선형 보간). 기간 안에 없으면 None."""
    cum = -cap
    for t, cf in enumerate(flows, start=1):
        if cum < 0 <= cum + cf:
            return t - 1 + (-cum) / cf
        cum += cf
    return None


def economics(size: Size, annual_bill_saving_krw: float, annual_discharge_dc_kwh: float,
              year: int = 2021, years: int = YEARS, discount: float = DISCOUNT, fx: float = USD_TO_KRW,
              first_year_bill_saving_krw: float | None = None, cycle_life: float = CYCLE_LIFE,
              calendar_life: float = CALENDAR_LIFE) -> dict:
    """연간 요금 절감액(원)으로 15년 경제성을 계산한다.

    annual_bill_saving_krw     : 2년차 이후(정상상태, steady) 연간 요금 절감 → 2~15년차
    first_year_bill_saving_krw : 설치 첫해(year1) 요금 절감 → 1년차. 생략하면 정상상태 값을 쓴다.
      첫해에는 설치 전 원래 운영의 피크가 래칫으로 남아 기본요금 절감이 줄어든다.
    현금흐름: 0년 −설치비 / 매년 +요금 절감 − 고정 O&M / 교체 연도 −교체비(저장 블록 + BOS) / 마지막 해 −폐기비
    본전 설치비율 = Σ 현금흐름 현재가치 ÷ 설치비 (1 이상이면 NPV 양수)
    손익분기 설치비 = Σ 현금흐름 현재가치 ÷ 명목 kWh ($/kWh) — 설치비가 이 값 이하면 본전"""
    u = unit_costs(size.nominal_hours, year)
    cap = capex_usd(size, year) * fx
    fom = u["fom"] * size.power_kw * fx
    repl = (u["storage_block"] + u["storage_bos"]) * size.nominal_kwh * fx
    life = battery_life_yr(size, annual_discharge_dc_kwh, cycle_life, calendar_life)
    repl_years, n = [], 1
    while n * life < years:
        repl_years.append(math.ceil(n * life))
        n += 1
    flows = []
    first = annual_bill_saving_krw if first_year_bill_saving_krw is None else first_year_bill_saving_krw
    for t in range(1, years + 1):
        cf = (first if t == 1 else annual_bill_saving_krw) - fom - repl * repl_years.count(t)
        if t == years:
            cf -= u["recycle"] * size.nominal_kwh * fx
        flows.append(cf)
    pv = sum(cf / (1 + discount) ** t for t, cf in enumerate(flows, start=1))
    return dict(capex_krw=cap, annual_fixed_om_krw=fom, pv_net_flows_krw=pv, npv_krw=pv - cap,
                breakeven_ratio=pv / cap,
                breakeven_capex_usd_per_kwh=pv / fx / size.nominal_kwh,
                pnnl_capex_usd_per_kwh=cap / fx / size.nominal_kwh,
                # 단순 회수기간은 정상상태 절감 기준
                simple_payback_yr=cap / (annual_bill_saving_krw - fom) if annual_bill_saving_krw > fom else math.inf,
                payback_yr=payback(cap, flows), battery_life_yr=life, replacement_years=repl_years)
