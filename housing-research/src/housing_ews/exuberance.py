"""Phillips-Shi-Yu 폭발적 근(explosive root) 검정 — SADF/GSADF/BSADF.

주택가격(또는 가격/소득, 가격/임대료 비율)이 랜덤워크(단위근)를 넘어
'폭발적'(mildly explosive, 자기회귀계수 > 1)으로 상승하는 구간을 통계적으로
식별한다. 비이성적 과열(irrational exuberance)의 계량적 탐지 장치로 국제적
표준이며, 댈러스 연준 International House Price Database의 exuberance
indicator가 이 방법을 사용한다.

방법 (Phillips, Shi & Yu, 2015, International Economic Review):

- 후방 확장 윈도우들에서 ADF t-통계량을 계산::

      Δy_t = α + β·y_{t-1} + Σ_{j=1..p} γ_j Δy_{t-j} + ε_t

  귀무가설 β = 0 (단위근), 대립가설 β > 0 (폭발적).
- BSADF(t) = sup_{시작점 s ≤ t-w0+1} ADF_{[s,t]}  (t 시점의 후방 sup ADF)
- GSADF = sup_t BSADF(t)  (표본 전체의 존재 검정)
- 시점 식별(date-stamping): BSADF(t)가 몬테카를로 임계값 수열을 최소 지속
  기간(≈ log T) 이상 연속 상회하면 그 구간을 폭발적 구간으로 판정.
- 최소 윈도우: w0 = ⌊T·(0.01 + 1.8/√T)⌋ (PSY 권고).

구현 노트
---------
- lag p = 0이면 프리픽스 합으로 각 윈도우의 OLS를 O(1)에 계산한다(빠름).
- 임계값은 표류항 없는 랜덤워크 귀무가설에서 lag 0으로 시뮬레이션한다.
  ADF t-통계량의 극한분포는 (올바른) 증강 차수와 무관하므로, lag p > 0의
  실증 통계량에 lag 0 임계값을 쓰는 것은 점근적으로 타당하다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def min_window(T: int) -> int:
    """PSY(2015) 권고 최소 윈도우 w0 = ⌊T(0.01 + 1.8/√T)⌋."""
    return int(np.floor(T * (0.01 + 1.8 / np.sqrt(T))))


def _adf_stats_all_starts(y: np.ndarray, end: int, w0: int) -> np.ndarray:
    """끝점 end 고정, 모든 시작점 s ∈ [0, end-w0+1]의 lag-0 ADF t-통계량.

    프리픽스 합으로 윈도우별 OLS(Δy ~ 1 + y_{-1})를 벡터화 계산한다.
    """
    # 회귀 관측치 i = 1..end : z_i = Δy_i, x_i = y_{i-1}
    z = np.diff(y[: end + 1])
    x = y[:end]
    czz = np.concatenate([[0.0], np.cumsum(z * z)])
    cz = np.concatenate([[0.0], np.cumsum(z)])
    cx = np.concatenate([[0.0], np.cumsum(x)])
    cxx = np.concatenate([[0.0], np.cumsum(x * x)])
    cxz = np.concatenate([[0.0], np.cumsum(x * z)])

    starts = np.arange(0, end - w0 + 2)  # 윈도우 [s, end], 길이 ≥ w0
    n = end - starts  # 회귀 표본 수
    Sz = cz[end] - cz[starts]
    Szz = czz[end] - czz[starts]
    Sx = cx[end] - cx[starts]
    Sxx = cxx[end] - cxx[starts]
    Sxz = cxz[end] - cxz[starts]

    det = n * Sxx - Sx * Sx
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (n * Sxz - Sx * Sz) / det
        a = (Sz - b * Sx) / n
        rss = Szz - a * Sz - b * Sxz
        sigma2 = rss / (n - 2)
        var_b = sigma2 * n / det
        t = b / np.sqrt(var_b)
    return t


def _adf_stat_lagged(y: np.ndarray, s: int, e: int, lags: int) -> float:
    """윈도우 [s, e]에서 lag p ≥ 1 ADF t-통계량 (명시적 OLS)."""
    dy = np.diff(y)
    rows = np.arange(s + 1 + lags, e + 1)  # 종속변수 시점
    if len(rows) < lags + 5:
        return np.nan
    Z = dy[rows - 1]
    X = [np.ones(len(rows)), y[rows - 1]]
    for j in range(1, lags + 1):
        X.append(dy[rows - 1 - j])
    X = np.column_stack(X)
    beta, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
    resid = Z - X @ beta
    dof = len(rows) - X.shape[1]
    if dof <= 0:
        return np.nan
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * xtx_inv[1, 1])
    return float(beta[1] / se)


def bsadf_sequence(y: np.ndarray, w0: int | None = None, lags: int = 0) -> np.ndarray:
    """BSADF(t) 수열. t < w0-1 구간은 NaN."""
    y = np.asarray(y, float)
    T = len(y)
    w0 = w0 or min_window(T)
    out = np.full(T, np.nan)
    for e in range(w0 - 1, T):
        if lags == 0:
            stats = _adf_stats_all_starts(y, e, w0)
            out[e] = np.nanmax(stats)
        else:
            vals = [
                _adf_stat_lagged(y, s, e, lags) for s in range(0, e - w0 + 2)
            ]
            out[e] = np.nanmax(vals) if vals else np.nan
    return out


def simulate_critical_values(
    T: int,
    w0: int | None = None,
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
    nreps: int = 499,
    seed: int = 42,
) -> dict:
    """랜덤워크 귀무가설에서 BSADF 임계값 수열과 GSADF 임계값을 시뮬레이션.

    Returns
    -------
    dict
        {"bsadf_cv": (len(quantiles), T) 배열, "gsadf_cv": 분위수 dict,
         "quantiles": quantiles}
    """
    w0 = w0 or min_window(T)
    rng = np.random.default_rng(seed)
    all_bsadf = np.full((nreps, T), np.nan)
    for r in range(nreps):
        e = rng.standard_normal(T)
        yr = np.cumsum(e)
        all_bsadf[r] = bsadf_sequence(yr, w0=w0, lags=0)
    bsadf_cv = np.full((len(quantiles), T), np.nan)
    valid = slice(w0 - 1, T)  # 최소 윈도우 이전 구간은 통계량이 정의되지 않음
    bsadf_cv[:, valid] = np.nanquantile(all_bsadf[:, valid], quantiles, axis=0)
    sup_stats = np.nanmax(all_bsadf[:, valid], axis=1)
    gsadf_cv = {q: float(np.quantile(sup_stats, q)) for q in quantiles}
    return {"bsadf_cv": bsadf_cv, "gsadf_cv": gsadf_cv, "quantiles": quantiles}


@dataclass
class ExuberanceResult:
    """GSADF 검정 결과와 폭발적 구간 식별.

    bsadf : BSADF(t) 수열 (index = 입력 시계열 index).
    cv : 분위수별 임계값 수열 DataFrame (columns 예: cv90, cv95, cv99).
    gsadf_stat, gsadf_cv : 전체 표본 존재 검정 통계량과 임계값.
    episodes : 폭발적 구간 [(시작, 끝)] 리스트 (95% 기준 + 지속기간 필터).
    margin : BSADF − cv95. 양수면 과열 임계 초과, 크기는 과열 강도.
    """

    bsadf: pd.Series
    cv: pd.DataFrame
    gsadf_stat: float
    gsadf_cv: dict
    episodes: list[tuple]
    margin: pd.Series

    @property
    def is_explosive_now(self) -> bool:
        return bool(self.margin.iloc[-1] > 0)


def gsadf_test(
    series: pd.Series,
    lags: int = 0,
    nreps: int = 499,
    min_duration: int | None = None,
    seed: int = 42,
) -> ExuberanceResult:
    """시계열에 GSADF 검정을 수행하고 폭발적 구간을 식별한다.

    Parameters
    ----------
    series : pd.Series
        검정 대상. 실질 가격지수(로그), 가격/소득, 가격/전세(임대료) 비율 등.
        비율 지표가 펀더멘털 대비 과열을 더 직접적으로 잡는다.
    min_duration : int | None
        폭발적 구간 최소 지속 기간. 기본값 ⌈log(T)⌉ (PSY 권고).
    """
    y = series.dropna()
    T = len(y)
    w0 = min_window(T)
    stat = bsadf_sequence(y.to_numpy(float), w0=w0, lags=lags)
    sim = simulate_critical_values(T, w0=w0, nreps=nreps, seed=seed)
    qs = sim["quantiles"]
    cv = pd.DataFrame(
        {f"cv{int(q * 100)}": sim["bsadf_cv"][i] for i, q in enumerate(qs)},
        index=y.index,
    )
    bsadf = pd.Series(stat, index=y.index, name="bsadf")
    margin = (bsadf - cv["cv95"]).rename("margin")

    min_dur = min_duration or int(np.ceil(np.log(T)))
    above = (margin > 0).to_numpy()
    episodes: list[tuple] = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_dur:
                episodes.append((y.index[start], y.index[i - 1]))
            start = None
    if start is not None and T - start >= min_dur:
        episodes.append((y.index[start], y.index[-1]))

    return ExuberanceResult(
        bsadf=bsadf,
        cv=cv,
        gsadf_stat=float(np.nanmax(stat)),
        gsadf_cv=sim["gsadf_cv"],
        episodes=episodes,
        margin=margin,
    )
