"""종합 조기경보지수(EWS)와 정책 개입 경계지점 판정.

개별 지표(수급률, 재고갭, BSADF 과열 여유폭, 가격 모멘텀 등)를 신호접근법으로
캘리브레이션한 뒤, 예보 성능 가중으로 합성해 하나의 경보 점수로 만든다.
(Kaminsky 1999는 1/NSR 가중을 쓰지만, 표본 내에서 한 지표가 과도하게
지배하는 것을 막기 위해 유계(0~1)인 Alessi-Detken 상대적 유용성을 기본
가중치로 쓴다.)

경보 점수 = Σ_i w_i·1[지표 i 신호 발령] / Σ_i w_i ∈ [0, 1],
w_i = max(U_r,i, 0)  (U_r = 상대적 유용성)

점수 구간을 4단계 경보로 사상한다 (구간은 설정 가능)::

    정상(GREEN) < 0.25 ≤ 주의(YELLOW) < 0.50 ≤ 경계(ORANGE) < 0.75 ≤ 심각(RED)

'심각'은 캘리브레이션된 지표 다수가 동시에 임계값을 넘은 상태로, 과거
데이터 기준 가격 급등이 임박했을 확률이 가장 높은 영역 — 긴급 정책 개입이
필요한 경계지점을 넘었다는 뜻이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .thresholds import OptimalThreshold

ALERT_LEVELS = ("GREEN", "YELLOW", "ORANGE", "RED")
ALERT_LABELS_KO = {"GREEN": "정상", "YELLOW": "주의", "ORANGE": "경계", "RED": "심각"}
DEFAULT_BOUNDS = (0.25, 0.50, 0.75)


@dataclass
class Indicator:
    """합성에 들어가는 개별 지표.

    name : 지표 이름 (보고서 표기용).
    series : 지표 시계열.
    calibration : optimal_threshold()로 얻은 임계값과 성능.
    """

    name: str
    series: pd.Series
    calibration: OptimalThreshold

    @property
    def weight(self) -> float:
        u = self.calibration.best.usefulness
        return max(float(u), 0.0) if u == u else 0.0

    def signal(self) -> pd.Series:
        """발령 여부 (1/0). 지표가 정의되지 않은 시점은 NaN 유지."""
        th = self.calibration.best.threshold
        if self.calibration.direction == "above":
            sig = (self.series >= th).astype(float)
        else:
            sig = (self.series <= th).astype(float)
        return sig.where(self.series.notna())


def composite_score(indicators: list[Indicator]) -> pd.Series:
    """가중 신호 합성 점수 (0~1).

    특정 시점에 정의되지 않은 지표(NaN)는 그 시점의 분모에서 제외해
    가용 지표만으로 재정규화한다.
    """
    sig_df = pd.concat({i.name: i.signal() for i in indicators}, axis=1)
    w = pd.Series({i.name: i.weight for i in indicators})
    if w.sum() == 0:
        raise ValueError("모든 지표의 가중치가 0입니다 (캘리브레이션 확인).")
    avail_w = sig_df.notna() @ w
    score = (sig_df.fillna(0.0) @ w) / avail_w.where(avail_w > 0)
    return score.rename("ews_score")


def alert_level(score: float, bounds: tuple = DEFAULT_BOUNDS) -> str:
    for level, b in zip(ALERT_LEVELS[:-1], bounds):
        if score < b:
            return level
    return ALERT_LEVELS[-1]


def policy_boundary_report(
    indicators: list[Indicator], bounds: tuple = DEFAULT_BOUNDS
) -> dict:
    """현재 시점의 경계지점 진단 보고서.

    Returns
    -------
    dict
        - table : 지표별 [현재값, 임계값, 방향, 경계까지 여유폭, 발령 여부,
          가중치] DataFrame
        - score : 현재 합성 점수
        - level / level_ko : 경보 단계
        - score_series : 합성 점수 시계열 (차트용)
    """
    rows = []
    for ind in indicators:
        cur = float(ind.series.dropna().iloc[-1])
        cal = ind.calibration
        rows.append(
            {
                "indicator": ind.name,
                "current": cur,
                "threshold": cal.best.threshold,
                "direction": "≥ 발령" if cal.direction == "above" else "≤ 발령",
                "margin_to_trigger": cal.distance_to_trigger(cur),
                "firing": bool(ind.signal().dropna().iloc[-1]),
                "weight": ind.weight,
                "hit_rate": cal.best.hit_rate,
                "false_alarm_rate": cal.best.false_alarm_rate,
            }
        )
    table = pd.DataFrame(rows)
    score_series = composite_score(indicators)
    score_now = float(score_series.dropna().iloc[-1])
    level = alert_level(score_now, bounds)
    return {
        "table": table,
        "score": score_now,
        "level": level,
        "level_ko": ALERT_LABELS_KO[level],
        "score_series": score_series,
        "bounds": bounds,
    }
