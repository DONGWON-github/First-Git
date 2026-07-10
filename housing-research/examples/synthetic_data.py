"""한국 주택시장 패턴을 모사한 월별 합성 데이터 생성기.

실데이터(국토부 인허가·착공·준공 통계, 통계청 가구추계, 한국부동산원 지수)를
붙이기 전에 전체 분석 파이프라인을 검증하기 위한 데이터다. 다음의 정형화된
사실(stylized facts)을 질적으로 재현한다.

- 인허가의 뚜렷한 사이클: 2008 금융위기 급감 → 2015~2017 대량 인허가 붐 →
  2022~2024 금리 급등기 인허가 붕괴(공급절벽의 씨앗).
- 인허가→착공 평균 약 10개월(실현율 88%), 착공→준공 약 30개월(실현율 97%)의
  분포시차. 준공 물량은 2~3년 전 착공이 결정.
- 가구수 증가는 둔화 추세이나 2020~2021년 세대분화 급증(1~2인 가구) 재현.
- 실질 가격: 수급갭·모멘텀·금리에 반응하고, 2020~2021년에 자기강화적
  (약폭발적) 과열 구간을 포함. 2022~2023 조정 후 2025년 말 재가속.

진짜 시차 분포·실현율을 TRUE_* 상수로 노출해, 추정치가 이를 복원하는지
검증할 수 있게 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import gamma as gamma_dist, norm as norm_dist

START = "2005-01-31"
END = "2026-06-30"


def _true_permit_to_start_weights(max_lag: int = 30) -> np.ndarray:
    """감마형 시차(평균 ≈ 10개월) × 실현율 0.88."""
    k = np.arange(max_lag + 1)
    pdf = gamma_dist.pdf(k, a=4.0, scale=2.6)
    w = pdf / pdf.sum() * 0.88
    return w


def _true_start_to_completion_weights(max_lag: int = 48) -> np.ndarray:
    """정규형 공사기간(평균 30개월, 표준편차 4개월) × 실현율 0.97."""
    k = np.arange(max_lag + 1)
    pdf = norm_dist.pdf(k, loc=30, scale=4)
    w = pdf / pdf.sum() * 0.97
    return w


TRUE_P2S = _true_permit_to_start_weights()
TRUE_S2C = _true_start_to_completion_weights()


def _permit_path(idx: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    """레짐별 수준 + 연말 스파이크 계절성 + 로그정규 잡음."""
    regimes = [
        ("2005-01", "2007-12", 36000),
        ("2008-01", "2010-12", 24000),
        ("2011-01", "2014-12", 34000),
        ("2015-01", "2017-12", 50000),  # 대량 인허가 붐 → 2018~20 입주폭탄
        ("2018-01", "2019-12", 38000),
        ("2020-01", "2021-12", 40000),
        ("2022-01", "2024-12", 19000),  # 금리 급등기 인허가 붕괴
        ("2025-01", "2026-06", 23000),
    ]
    level = pd.Series(index=idx, dtype=float)
    for s, e, v in regimes:
        level.loc[s:e] = v
    level = level.rolling(9, min_periods=1, center=True).mean()
    season = pd.Series(1.0, index=idx)
    season[idx.month == 12] = 1.8  # 연말 인허가 몰림
    season[idx.month == 1] = 0.7
    noise = np.exp(rng.normal(0.0, 0.10, len(idx)))
    return (level * season * noise).rename("permits")


def _household_path(idx: pd.DatetimeIndex) -> pd.Series:
    """연간 가구 증가(천 가구): 둔화 추세 + 2020~21 세대분화 급증."""
    annual_growth = {
        2005: 350, 2006: 350, 2007: 340, 2008: 330, 2009: 320, 2010: 330,
        2011: 320, 2012: 310, 2013: 300, 2014: 300, 2015: 310, 2016: 300,
        2017: 300, 2018: 310, 2019: 330, 2020: 590, 2021: 520, 2022: 320,
        2023: 250, 2024: 220, 2025: 200, 2026: 190,
    }
    monthly = []
    hh = 15_900_000.0
    for ts in idx:
        hh += annual_growth[ts.year] * 1000 / 12
        monthly.append(hh)
    return pd.Series(monthly, index=idx, name="households")


def make_dataset(seed: int = 7) -> dict:
    """합성 데이터셋 생성.

    Returns
    -------
    dict
        permits, starts, completions, households, stock, price_real (모두
        pd.Series, 월말 인덱스), 그리고 진짜 시차 가중치 true_p2s/true_s2c.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, END, freq="ME")
    n = len(idx)

    permits = _permit_path(idx, rng)

    # 파이프라인 전파: 표본 시작 이전에도 공급이 있었으므로 워밍업 구간을
    # 앞에 붙여 경계 왜곡을 없앤다.
    warmup = 60
    pre_idx = pd.date_range(end=idx[0] - pd.offsets.MonthEnd(1), periods=warmup, freq="ME")
    pre_permits = pd.Series(
        34000 * np.exp(rng.normal(0, 0.08, warmup)), index=pre_idx
    )
    permits_full = pd.concat([pre_permits, permits])

    x = permits_full.to_numpy()
    starts_clean = np.convolve(x, TRUE_P2S)[: len(x)]
    starts_full = pd.Series(
        starts_clean * np.exp(rng.normal(0, 0.06, len(x))), index=permits_full.index
    )
    s = starts_full.to_numpy()
    comp_clean = np.convolve(s, TRUE_S2C)[: len(s)]
    completions_full = pd.Series(
        comp_clean * np.exp(rng.normal(0, 0.05, len(s))), index=permits_full.index
    )

    starts = starts_full.loc[idx[0]:].rename("starts")
    completions = completions_full.loc[idx[0]:].rename("completions")

    households = _household_path(idx)

    stock = pd.Series(index=idx, dtype=float, name="stock")
    level = 13_000_000.0
    for i, ts in enumerate(idx):
        level += completions.iloc[i] - level * 0.004 / 12
        stock.iloc[i] = level

    # ---- 실질 가격: 트렌드 + 모멘텀 + 수급 압력 + 금리 + 과열(약폭발) ----
    rate = pd.Series(3.0, index=idx)
    rate.loc["2008-01":"2008-12"] = 5.0
    rate.loc["2009-01":"2013-12"] = 2.75
    rate.loc["2014-01":"2019-12"] = 1.75
    rate.loc["2020-01":"2021-07"] = 0.5
    rate.loc["2021-08":"2022-12"] = 2.0
    rate.loc["2023-01":"2024-09"] = 3.5
    rate.loc["2024-10":] = 2.75
    d_rate = rate.diff().fillna(0.0)

    req_proxy = households.diff() + stock * 0.004 / 12
    gap12 = (
        completions.rolling(12).sum() / req_proxy.rolling(12).sum()
    ).shift(3)  # 수급 압력은 3개월 시차로 가격에 반영

    # 과열(버블) 성분: 구간별 (AR계수, 유입). AR>1 = 약폭발적 자기강화.
    bubble_regimes = [
        ("2006-01", "2007-03", 1.05, 0.0008),   # 2006 급등기
        ("2008-09", "2009-06", 0.85, -0.0012),  # 금융위기 조정
        ("2020-06", "2021-10", 1.05, 0.0008),   # 2020~21 과열
        ("2022-01", "2023-03", 0.85, -0.0025),  # 금리 급등기 조정
        ("2025-07", "2026-06", 1.05, 0.0015),   # 공급절벽발 재가속
    ]

    g = np.zeros(n)
    log_p = np.log(100.0)
    prices = np.zeros(n)
    bubble = 0.0
    for i in range(n):
        ym = idx[i].strftime("%Y-%m")
        mom = np.mean(g[max(0, i - 6) : i]) if i > 0 else 0.0
        pressure = 0.0
        if i >= 15 and not np.isnan(gap12.iloc[i]):
            # 부족(<1)이면 상승, 과잉(>1)이면 하락 압력
            pressure = float(np.clip(1.0 - gap12.iloc[i], -0.25, 0.40))
        drift = 0.0002 + 0.30 * mom + 0.005 * pressure - 0.005 * d_rate.iloc[i]
        rho, inflow = 0.85, 0.0
        for s, e, r, f in bubble_regimes:
            if s <= ym <= e:
                rho, inflow = r, f
                break
        bubble = rho * bubble + inflow
        g[i] = drift + bubble + rng.normal(0, 0.008)
        log_p += g[i]
        prices[i] = np.exp(log_p)

    price_real = pd.Series(prices, index=idx, name="price_real")

    return {
        "permits": permits,
        "starts": starts,
        "completions": completions,
        "households": households,
        "stock": stock,
        "price_real": price_real,
        "policy_rate": rate,
        "true_p2s": TRUE_P2S,
        "true_s2c": TRUE_S2C,
    }
