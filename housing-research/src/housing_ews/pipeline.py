"""인허가→착공→준공 공급 파이프라인 분포시차 모형.

주택 공급은 인허가(permits) → 착공(starts) → 준공(completions)의 단계를
거치며, 각 단계 사이에 수개월~수년의 시차가 존재한다. 아파트의 경우
착공→준공이 통상 24~36개월 걸리므로, 향후 2~3년의 준공(입주) 물량은
이미 관측된 착공 실적에 의해 대부분 "결정되어" 있다.

이 모듈은 두 단계의 시차 구조를 비음(非陰) 분포시차 회귀로 추정한다::

    starts(t)      = Σ_k  w^s_k · permits(t-k) + ε_t      (w^s_k ≥ 0)
    completions(t) = Σ_k  w^c_k · starts(t-k)  + ε_t      (w^c_k ≥ 0)

가중치 합 Σw 는 실현율(인허가 물량 중 실제 착공/준공으로 이어지는 비율)로
해석되고, 가중치의 분포 자체가 시차 분포(몇 개월 뒤에 몇 %가 실현되는가)가
된다. 추정된 시차 분포로 미래 준공 물량을 전망하며, 전망치 중 이미 관측된
인허가·착공만으로 결정된 부분(locked-in)과 미래 인허가 시나리오에 의존하는
부분을 분리해 준다.

참고 문헌: 분포시차 모형은 Almon(1965) 이후 표준 기법이며, 주택 파이프라인
적용은 미국 센서스국 Survey of Construction의 permit→start→completion 시차
통계, Coulson(1999) 등 착공 시계열 연구를 따른다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def _lag_matrix(x: np.ndarray, max_lag: int) -> np.ndarray:
    """x의 0~max_lag 시차 행렬을 만든다. 행 t, 열 k = x[t-k] (없으면 NaN)."""
    n = len(x)
    mat = np.full((n, max_lag + 1), np.nan)
    for k in range(max_lag + 1):
        if k == 0:
            mat[:, 0] = x
        else:
            mat[k:, k] = x[:-k]
    return mat


@dataclass
class LagDistribution:
    """단계 간 시차 분포 추정 결과.

    Attributes
    ----------
    weights : np.ndarray
        시차 k별 실현 가중치 w_k (k = 0..max_lag, 비음).
    realization_rate : float
        Σw_k. 상류 물량 중 하류로 실현되는 총비율 (1보다 작으면 소멸 존재).
    mean_lag, median_lag : float
        가중치 기준 평균·중위 시차 (기간 단위 = 입력 데이터 주기).
    r2 : float
        적합도.
    """

    weights: np.ndarray
    realization_rate: float
    mean_lag: float
    median_lag: float
    r2: float

    def cumulative_by(self, k: int) -> float:
        """시차 k 이내에 실현되는 누적 비율(실현율 대비)."""
        if self.realization_rate == 0:
            return 0.0
        return float(self.weights[: k + 1].sum() / self.realization_rate)


def estimate_lag_distribution(
    upstream: pd.Series, downstream: pd.Series, max_lag: int
) -> LagDistribution:
    """비음최소제곱(NNLS)으로 상류→하류 시차 분포를 추정한다.

    Parameters
    ----------
    upstream, downstream : pd.Series
        동일 주기의 물량 시계열 (예: 월별 인허가 호수, 월별 착공 호수).
    max_lag : int
        고려할 최대 시차 (기간 수). 아파트 인허가→착공은 24~36개월,
        착공→준공은 36~48개월 정도가 안전하다.
    """
    df = pd.concat({"up": upstream, "down": downstream}, axis=1).dropna()
    x = df["up"].to_numpy(float)
    y = df["down"].to_numpy(float)
    lagged = _lag_matrix(x, max_lag)
    valid = ~np.isnan(lagged).any(axis=1)
    if valid.sum() < max_lag + 10:
        raise ValueError(
            f"유효 표본 {valid.sum()}개로는 max_lag={max_lag} 추정이 불가능합니다. "
            "표본을 늘리거나 max_lag를 줄이세요."
        )
    a, b = lagged[valid], y[valid]
    weights, _ = nnls(a, b)
    fitted = a @ weights
    ss_res = float(((b - fitted) ** 2).sum())
    ss_tot = float(((b - b.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    total = weights.sum()
    lags = np.arange(max_lag + 1)
    if total > 0:
        mean_lag = float((lags * weights).sum() / total)
        cum = np.cumsum(weights) / total
        median_lag = float(lags[np.searchsorted(cum, 0.5)])
    else:
        mean_lag = median_lag = np.nan
    return LagDistribution(
        weights=weights,
        realization_rate=float(total),
        mean_lag=mean_lag,
        median_lag=median_lag,
        r2=r2,
    )


@dataclass
class CompletionForecast:
    """준공 물량 전망 결과.

    total : 전망 준공 물량 (기간별).
    locked_in : 이미 관측된 인허가·착공만으로 결정된 부분.
    scenario_dependent : 미래 인허가 시나리오에 의존하는 부분.
    locked_in_share : locked_in / total. 1에 가까울수록 "이미 정해진 미래".
    """

    total: pd.Series
    locked_in: pd.Series
    scenario_dependent: pd.Series

    @property
    def locked_in_share(self) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            share = self.locked_in / self.total
        return share.fillna(1.0)


@dataclass
class PipelineModel:
    """인허가→착공→준공 2단계 파이프라인 모형.

    사용 순서: ``fit(permits, starts, completions)`` →
    ``forecast_completions(horizon, permit_scenario)``.
    """

    permit_to_start_maxlag: int = 30
    start_to_completion_maxlag: int = 48
    permit_to_start: LagDistribution | None = field(default=None, init=False)
    start_to_completion: LagDistribution | None = field(default=None, init=False)
    _permits: pd.Series | None = field(default=None, init=False)
    _starts: pd.Series | None = field(default=None, init=False)

    def fit(
        self, permits: pd.Series, starts: pd.Series, completions: pd.Series
    ) -> "PipelineModel":
        self.permit_to_start = estimate_lag_distribution(
            permits, starts, self.permit_to_start_maxlag
        )
        self.start_to_completion = estimate_lag_distribution(
            starts, completions, self.start_to_completion_maxlag
        )
        self._permits = permits.copy()
        self._starts = starts.copy()
        return self

    def _require_fit(self) -> None:
        if self.permit_to_start is None or self.start_to_completion is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")

    def forecast_completions(
        self, horizon: int, permit_scenario: pd.Series | float | None = None
    ) -> CompletionForecast:
        """향후 horizon 기간의 준공 물량을 전망한다.

        Parameters
        ----------
        horizon : int
            전망 기간 수 (입력 데이터 주기 기준).
        permit_scenario : pd.Series | float | None
            미래 인허가 가정. None이면 최근 12기간 평균, float면 그 값으로
            일정, Series면 기간별 값. 시나리오는 scenario_dependent 부분에만
            영향을 준다.
        """
        self._require_fit()
        assert self._permits is not None and self._starts is not None
        freq = self._permits.index.freq or pd.infer_freq(self._permits.index)
        future_idx = pd.date_range(
            self._permits.index[-1], periods=horizon + 1, freq=freq
        )[1:]

        if permit_scenario is None:
            scen_val = float(self._permits.tail(12).mean())
            scen = pd.Series(scen_val, index=future_idx)
        elif isinstance(permit_scenario, (int, float)):
            scen = pd.Series(float(permit_scenario), index=future_idx)
        else:
            scen = permit_scenario.reindex(future_idx).ffill()

        # locked-in: 미래 인허가를 0으로 두고 관측치만으로 전파
        permits_locked = pd.concat(
            [self._permits, pd.Series(0.0, index=future_idx)]
        )
        permits_scen = pd.concat([self._permits, scen])

        def propagate(series: pd.Series, dist: LagDistribution) -> pd.Series:
            x = series.to_numpy(float)
            w = dist.weights
            out = np.convolve(x, w)[: len(x)]
            return pd.Series(out, index=series.index)

        assert self.permit_to_start is not None
        assert self.start_to_completion is not None

        def completions_path(permits_path: pd.Series) -> pd.Series:
            starts_pred = propagate(permits_path, self.permit_to_start)
            # 관측된 착공 실적이 있는 구간은 실적으로 대체 (예측오차 제거)
            starts_full = starts_pred.copy()
            starts_full.loc[self._starts.index] = self._starts
            return propagate(starts_full, self.start_to_completion)

        comp_locked = completions_path(permits_locked).reindex(future_idx)
        comp_scen = completions_path(permits_scen).reindex(future_idx)
        scenario_part = (comp_scen - comp_locked).clip(lower=0.0)
        return CompletionForecast(
            total=comp_scen, locked_in=comp_locked, scenario_dependent=scenario_part
        )
