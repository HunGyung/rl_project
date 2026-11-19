"""사용자 요금 명세의 계산. 공식 약관 인증/세금 포함 청구 엔진은 아니다."""

RATCHET_MONTHS = (1, 2, 7, 8, 9, 12)


def month_index(date):
    return date.year * 12 + date.month - 1


def ratchet_history(history, current_month):
    """당월 이전 11개월의 이월 대상 최대수요. 가장 오래된 달부터 반환한다."""
    return tuple(history.get(month, 0.0) if month % 12 + 1 in RATCHET_MONTHS else 0.0
                 for month in range(current_month - 11, current_month))


def original_monthly_peaks(months, cfg):
    return {month_index(rows[0][0]): max((load for date, load in rows if tariff(date, cfg)[1] > 0), default=0.0)
            for rows in months.values()}


def price_parts(date, cfg, band):
    season = "summer" if date.month in (6, 7, 8) else (
        "winter" if date.month in (11, 12, 1, 2) else "shoulder")
    fuel = cfg["fuel_adjustment_by_quarter"][(date.month - 1) // 3]
    return cfg["rates"][season][band], cfg["surcharge_per_kwh"], fuel


def tariff(date, cfg):
    years = cfg.get("holiday_calendar_years")
    if years is not None and date.year not in years:
        raise ValueError(f"Holiday calendar not configured for {date.year}")
    season = "summer" if date.month in (6, 7, 8) else (
        "winter" if date.month in (11, 12, 1, 2) else "shoulder")
    h = date.hour
    if date.weekday() == 6 or date.date().isoformat() in cfg["holidays"] or h < 9 or h >= 23:
        band = 0
    else:
        peak_hours = cfg["peak_hours"]
        band = 2 if h in peak_hours[season] and date.weekday() < 5 else 1
    return sum(price_parts(date, cfg, band)), band, ("summer", "shoulder", "winter").index(season)
