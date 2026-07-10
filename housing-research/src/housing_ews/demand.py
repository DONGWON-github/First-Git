"""인구·가구 통계 기반 신규 주택 필요공급량(요구수요) 추정.

표준 프레임(미국 JCHS·호주 NHSAC·영국 DLUHC의 housing need 산정과 동일한
회계 항등식)::

    필요공급(t) = 순가구증가(t) + 멸실(t) + 공가버퍼(t)

- 순가구증가: 통계청 장래가구추계(또는 주민등록 세대수 증감). 인구가 정체해도
  1~2인 가구 분화로 가구수는 증가할 수 있으므로 반드시 '가구' 기준을 쓴다.
- 멸실: 재건축·재개발·노후 소멸분. 재고 대비 멸실률(연 0.3~0.6% 수준)로
  근사하거나 국토부 멸실 통계를 직접 사용.
- 공가버퍼: 이사 마찰을 위한 자연공실을 유지하기 위한 추가분. 가구증가분에
  자연공실률을 곱해 근사한다.

Mankiw & Weil (1989, RSUE) 의 인구구조 기반 수요 추정과 달리, 이 모듈은
정책 실무에서 쓰는 회계적 필요량(housing need)을 계산한다. 두 접근은
보완적이며, 연구에서는 가격 모형의 수요 변수로 함께 쓸 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DemandAssumptions:
    """필요공급량 산정 가정.

    demolition_rate : 연간 재고 대비 멸실률 (예: 0.004 = 0.4%).
    natural_vacancy : 자연공실률 (예: 0.05). 가구증가분의 5%를 버퍼로 가산.
    """

    demolition_rate: float = 0.004
    natural_vacancy: float = 0.05


def required_supply(
    households: pd.Series,
    housing_stock: pd.Series,
    assumptions: DemandAssumptions | None = None,
    demolitions: pd.Series | None = None,
) -> pd.DataFrame:
    """기간별 신규 주택 필요공급량을 계산한다.

    Parameters
    ----------
    households : pd.Series
        가구수 수준(level) 시계열. 차분해 순가구증가를 계산한다.
    housing_stock : pd.Series
        주택 재고 시계열 (멸실률 적용 기준).
    demolitions : pd.Series | None
        실제 멸실 통계가 있으면 전달 (멸실률 근사 대신 사용).

    Returns
    -------
    pd.DataFrame
        columns = [household_growth, demolition, vacancy_buffer, required]
    """
    a = assumptions or DemandAssumptions()
    hh_growth = households.diff()
    if demolitions is None:
        # 연율 멸실률을 데이터 주기에 맞게 환산
        periods_per_year = _periods_per_year(households.index)
        demolitions = housing_stock * (a.demolition_rate / periods_per_year)
    vacancy_buffer = hh_growth.clip(lower=0) * a.natural_vacancy
    required = hh_growth + demolitions + vacancy_buffer
    out = pd.DataFrame(
        {
            "household_growth": hh_growth,
            "demolition": demolitions,
            "vacancy_buffer": vacancy_buffer,
            "required": required,
        }
    )
    return out.dropna()


def _periods_per_year(index: pd.Index) -> float:
    freq = getattr(index, "freq", None) or pd.infer_freq(index)
    if freq is None:
        raise ValueError("시계열 주기를 추론할 수 없습니다. 주기를 지정한 인덱스를 쓰세요.")
    code = freq.freqstr if hasattr(freq, "freqstr") else str(freq)
    code = code.upper()
    if code.startswith(("M", "ME", "MS")):
        return 12.0
    if code.startswith(("Q", "QE", "QS")):
        return 4.0
    if code.startswith(("A", "Y")):
        return 1.0
    raise ValueError(f"지원하지 않는 주기: {code} (M/Q/Y만 지원)")
