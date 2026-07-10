"""수급갭(공급-수요 불균형) 계산.

두 층위의 갭을 계산한다.

1. 기간별 갭(flow gap): 준공(또는 전망 준공) − 필요공급. 음수면 해당 기간
   공급 부족.
2. 누적 갭(stock gap): 기간별 갭의 누적합. 과거 과잉/부족이 재고에 누적되어
   가격 압력으로 작용한다는 stock-flow 모형(DiPasquale & Wheaton 1994)의
   재고 불균형 개념에 대응한다.

가격 압력 해석의 편의를 위해 필요공급 대비 비율(gap_ratio)도 제공한다.
gap_ratio = 공급/필요공급. 1.0 미만이면 부족, 예컨대 0.7이면 필요량의
70%만 공급된다는 뜻이다.
"""

from __future__ import annotations

import pandas as pd


def supply_demand_gap(
    supply: pd.Series, required: pd.Series, initial_stock_gap: float = 0.0
) -> pd.DataFrame:
    """기간별·누적 수급갭을 계산한다.

    Parameters
    ----------
    supply : pd.Series
        기간별 공급량 (실적 준공 + 전망 준공을 이어붙여 전달 가능).
    required : pd.Series
        기간별 필요공급량 (demand.required_supply의 'required' 열).
    initial_stock_gap : float
        분석 시작 시점 이전에 누적된 재고 과부족 (알고 있으면 지정).

    Returns
    -------
    pd.DataFrame
        columns = [supply, required, flow_gap, gap_ratio, stock_gap]
    """
    df = pd.concat({"supply": supply, "required": required}, axis=1).dropna()
    df["flow_gap"] = df["supply"] - df["required"]
    df["gap_ratio"] = df["supply"] / df["required"].where(df["required"] != 0)
    df["stock_gap"] = initial_stock_gap + df["flow_gap"].cumsum()
    return df


def rolling_gap_ratio(gap_df: pd.DataFrame, window: int) -> pd.Series:
    """이동 window 합 기준 수급률 (단기 변동 평활용).

    예: 월별 데이터에 window=12 → 최근 1년 공급/1년 필요공급.
    """
    s = gap_df["supply"].rolling(window).sum()
    r = gap_df["required"].rolling(window).sum()
    return (s / r).rename(f"gap_ratio_{window}")
