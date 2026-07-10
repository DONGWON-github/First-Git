"""신호접근법(signals approach) 기반 조기경보 임계값 산출.

"긴급한 정책이 필요한 경계지점"을 자의적으로 정하지 않고, 과거 데이터에서
가격 급등 사건을 가장 잘 예보하는 임계값을 통계적으로 역산한다.

절차 (Kaminsky & Reinhart 1999, AER; Alessi & Detken 2011, EJPE — ECB 표준):

1. 사건 정의: "향후 h기간 내 실질 주택가격 상승률이 g% 초과" 같은 급등
   사건을 이진 변수로 라벨링한다.
2. 각 지표(수급률, 재고갭, BSADF 여유폭 등)에 대해 임계값 후보를 훑으며
   신호/사건의 2×2 분할표를 만든다::

                     사건 발생(h기 내)   사건 없음
       신호 발령           A(적중)        B(오경보)
       신호 없음           C(누락)        D(정상침묵)

3. 최적 임계값 선택 기준
   - 잡음-신호비 NSR = [B/(B+D)] / [A/(A+C)] 최소화 (Kaminsky-Reinhart)
   - 정책자 손실 L(μ) = μ·누락률 + (1-μ)·오경보율 최소화, 상대적 유용성
     U_r = [min(μ,1-μ) − L] / min(μ,1-μ) (Alessi-Detken). μ는 누락(Type I)을
     오경보(Type II)보다 얼마나 싫어하는지의 정책 선호. 위기 예방이 급하면
     μ를 0.5보다 크게 둔다.
   - Youden J = 적중률 − 오경보율 최대화 (ROC 기준)

임계값이 산출되면 "현재 지표값과 임계값의 거리"가 곧 경계지점까지 남은
여유폭이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def label_surge_events(
    price: pd.Series, horizon: int, surge_growth: float, periods_per_year: int = 12
) -> pd.Series:
    """향후 horizon 기간 내 가격 급등 발생 여부를 시점별로 라벨링한다.

    사건 정의: t 이후 h기간 내 어느 시점의 '연율 환산 상승률'이 surge_growth를
    초과하면 event(t) = 1. (예: 월별 데이터, horizon=12, surge_growth=0.15
    → 향후 1년 내 전년동월비 15% 초과 급등 발생)

    Parameters
    ----------
    price : pd.Series
        실질 가격지수 (명목 지수는 물가로 디플레이트해 전달 권장).
    """
    yoy = price.pct_change(periods_per_year)
    fwd_max = (
        yoy.shift(-1)
        .rolling(window=horizon, min_periods=1)
        .max()
        .shift(-(horizon - 1))
    )
    event = (fwd_max > surge_growth).astype(float)
    event[fwd_max.isna()] = np.nan  # 표본 끝: 미래 미관측 → 라벨 불가
    return event.rename("event")


@dataclass
class ThresholdEvaluation:
    """단일 임계값의 예보 성능."""

    threshold: float
    hit_rate: float          # A/(A+C): 사건을 미리 잡아낸 비율
    false_alarm_rate: float  # B/(B+D): 평시에 잘못 울린 비율
    noise_to_signal: float   # 오경보율/적중률 (낮을수록 좋음)
    youden_j: float
    usefulness: float        # Alessi-Detken 상대적 유용성 U_r
    counts: tuple            # (A, B, C, D)


def evaluate_threshold(
    indicator: pd.Series,
    event: pd.Series,
    threshold: float,
    direction: str = "above",
    mu: float = 0.6,
) -> ThresholdEvaluation:
    """임계값 하나의 신호 성능을 평가한다.

    direction : "above"면 지표 ≥ 임계값일 때 신호(과열 지표),
                "below"면 지표 ≤ 임계값일 때 신호(수급률처럼 낮을수록 위험).
    mu : 정책자 선호 (누락 회피 가중치, 0.5 = 중립).
    """
    df = pd.concat({"x": indicator, "e": event}, axis=1).dropna()
    signal = df["x"] >= threshold if direction == "above" else df["x"] <= threshold
    ev = df["e"] > 0.5
    A = int((signal & ev).sum())
    B = int((signal & ~ev).sum())
    C = int((~signal & ev).sum())
    D = int((~signal & ~ev).sum())
    hit = A / (A + C) if (A + C) else np.nan
    fa = B / (B + D) if (B + D) else np.nan
    nsr = fa / hit if hit and hit > 0 else np.inf
    miss = 1 - hit if hit == hit else np.nan
    loss = mu * miss + (1 - mu) * fa
    u_abs = min(mu, 1 - mu) - loss
    u_rel = u_abs / min(mu, 1 - mu)
    return ThresholdEvaluation(
        threshold=float(threshold),
        hit_rate=hit,
        false_alarm_rate=fa,
        noise_to_signal=nsr,
        youden_j=(hit - fa) if hit == hit and fa == fa else np.nan,
        usefulness=u_rel,
        counts=(A, B, C, D),
    )


@dataclass
class OptimalThreshold:
    """지표 하나의 최적 임계값 탐색 결과.

    best : 선택된 임계값의 성능.
    grid : 후보 임계값 전체의 성능 표 (임계값 민감도 점검용).
    direction : 신호 방향.
    """

    best: ThresholdEvaluation
    grid: pd.DataFrame
    direction: str

    def distance_to_trigger(self, current_value: float) -> float:
        """현재 지표값에서 경계지점(임계값)까지 남은 여유폭.

        양수면 아직 임계값 안쪽(여유 있음), 음수면 이미 임계값을 넘었음.
        """
        if self.direction == "above":
            return self.best.threshold - current_value
        return current_value - self.best.threshold


def optimal_threshold(
    indicator: pd.Series,
    event: pd.Series,
    direction: str = "above",
    criterion: str = "usefulness",
    mu: float = 0.6,
    min_hit_rate: float = 0.5,
    n_grid: int = 81,
) -> OptimalThreshold:
    """분위수 그리드를 훑어 최적 임계값을 찾는다.

    criterion : "usefulness" | "nsr" | "youden"
    min_hit_rate : 적중률 하한. 경보가 사건 절반도 못 잡으면 임계값으로서
        의미가 없으므로 기본 50%를 강제한다.
    """
    x = indicator.dropna()
    grid_vals = np.unique(np.quantile(x, np.linspace(0.05, 0.95, n_grid)))
    rows = [
        evaluate_threshold(indicator, event, th, direction=direction, mu=mu)
        for th in grid_vals
    ]
    grid = pd.DataFrame(
        {
            "threshold": [r.threshold for r in rows],
            "hit_rate": [r.hit_rate for r in rows],
            "false_alarm_rate": [r.false_alarm_rate for r in rows],
            "noise_to_signal": [r.noise_to_signal for r in rows],
            "youden_j": [r.youden_j for r in rows],
            "usefulness": [r.usefulness for r in rows],
        }
    )
    feasible = grid[grid["hit_rate"] >= min_hit_rate]
    if feasible.empty:
        feasible = grid
    if criterion == "nsr":
        idx = feasible["noise_to_signal"].idxmin()
    elif criterion == "youden":
        idx = feasible["youden_j"].idxmax()
    else:
        idx = feasible["usefulness"].idxmax()
    return OptimalThreshold(best=rows[int(idx)], grid=grid, direction=direction)
