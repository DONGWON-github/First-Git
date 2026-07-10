"""핵심 로직 스모크 테스트.

실행:  python tests/test_housing_ews.py   (pytest 없이도 동작)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from housing_ews import demand, exuberance, gap, pipeline, thresholds


def test_lag_recovery():
    """알려진 시차 커널을 NNLS가 복원하는지."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-31", periods=300, freq="ME")
    true_w = np.zeros(13)
    true_w[[6, 7, 8]] = [0.3, 0.4, 0.2]  # 실현율 0.9, 평균시차 ~7
    x = pd.Series(1000 + 200 * rng.standard_normal(300).cumsum() * 0.05 + 100 * np.sin(np.arange(300) / 12), index=idx).clip(lower=100)
    y_clean = np.convolve(x.to_numpy(), true_w)[:300]
    y = pd.Series(y_clean * np.exp(rng.normal(0, 0.02, 300)), index=idx)
    dist = pipeline.estimate_lag_distribution(x, y, max_lag=12)
    assert abs(dist.realization_rate - 0.9) < 0.05, dist.realization_rate
    assert abs(dist.mean_lag - 6.9) < 0.7, dist.mean_lag
    assert dist.r2 > 0.9


def test_gsadf_detects_explosive_not_randomwalk():
    """폭발적 구간이 있는 시계열은 검출하고, 순수 랜덤워크는 기각하지 않는지."""
    rng = np.random.default_rng(1)
    T = 160
    idx = pd.date_range("2000-01-31", periods=T, freq="ME")
    # 양(+)의 수준에서 출발하는 랜덤워크 + 중간의 폭발 구간 (60~90)
    y = np.zeros(T)
    y[0] = 50.0
    for t in range(1, T):
        rho = 1.03 if 60 <= t < 90 else 1.0
        y[t] = rho * y[t - 1] + rng.standard_normal()
    res = exuberance.gsadf_test(pd.Series(y, index=idx), nreps=199, seed=3)
    assert res.gsadf_stat > res.gsadf_cv[0.95], "폭발 구간 미검출"
    assert len(res.episodes) >= 1
    # 검출 구간이 진짜 폭발 구간(60~90)과 겹치는지
    ep_idx = [(idx.get_loc(s), idx.get_loc(e)) for s, e in res.episodes]
    assert any(s < 90 and e >= 60 for s, e in ep_idx), ep_idx

    rw = pd.Series(np.cumsum(rng.standard_normal(T)), index=idx)
    res_rw = exuberance.gsadf_test(rw, nreps=199, seed=4)
    assert res_rw.gsadf_stat < res_rw.gsadf_cv[0.99] + 1.0  # 큰 초과는 없어야

def test_threshold_separable():
    """지표가 사건을 완전히 분리하면 최적 임계값이 그 경계를 찾는지."""
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    x = pd.Series(np.linspace(0, 1, 200), index=idx)
    event = pd.Series((x > 0.7).astype(float), index=idx)
    cal = thresholds.optimal_threshold(x, event, direction="above", min_hit_rate=0.5)
    assert 0.6 <= cal.best.threshold <= 0.72, cal.best.threshold
    assert cal.best.hit_rate > 0.95
    assert cal.best.false_alarm_rate < 0.05
    assert cal.distance_to_trigger(0.5) > 0  # 아직 안쪽
    assert cal.distance_to_trigger(0.9) < 0  # 경계 초과


def test_required_supply_accounting():
    """필요공급 = 가구증가 + 멸실 + 버퍼 항등식."""
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    hh = pd.Series(1_000_000 + 1000 * np.arange(24), index=idx)
    stock = pd.Series(900_000.0, index=idx)
    out = demand.required_supply(hh, stock, demand.DemandAssumptions(0.006, 0.05))
    row = out.iloc[5]
    assert abs(row["household_growth"] - 1000) < 1e-9
    assert abs(row["demolition"] - 900_000 * 0.006 / 12) < 1e-9
    assert abs(row["required"] - (1000 + 450 + 50)) < 1e-9


def test_gap_identity():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    sup = pd.Series(100.0, index=idx)
    req = pd.Series(80.0, index=idx)
    g = gap.supply_demand_gap(sup, req, initial_stock_gap=-50)
    assert (g["flow_gap"] == 20).all()
    assert g["stock_gap"].iloc[-1] == -50 + 20 * 12
    assert np.allclose(g["gap_ratio"], 1.25)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}개 테스트 통과")
