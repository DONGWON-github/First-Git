"""housing_ews — 주택 수급 불균형 조기경보 분석 패키지.

인허가→착공→준공 공급 파이프라인의 분포시차를 추정해 향후 준공(입주) 물량을
전망하고, 인구·가구 통계 기반 필요공급량과 비교해 수급갭을 계산한 뒤,
가격 지표의 폭발적 상승(exuberance) 검정과 결합해 긴급 정책 개입이 필요한
경계지점(임계값)을 산출한다.

모듈 구성
---------
- pipeline    : 인허가→착공→준공 분포시차 추정 및 준공 물량 전망
- demand      : 가구 증가·멸실·공가 버퍼 기반 필요공급량 추정
- gap         : 수급갭(기간별·누적) 계산
- exuberance  : Phillips-Shi-Yu SADF/GSADF 폭발적 근 검정 (버블 시점 식별)
- thresholds  : 신호접근법(Kaminsky-Reinhart) 기반 최적 임계값 산출
- ews         : 종합 조기경보지수 및 경보 단계(정상/주의/경계/심각) 판정
"""

from . import demand, ews, exuberance, gap, pipeline, thresholds

__all__ = ["pipeline", "demand", "gap", "exuberance", "thresholds", "ews"]
__version__ = "0.1.0"
