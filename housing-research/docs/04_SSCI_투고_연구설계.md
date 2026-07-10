# SCI/SSCI 투고를 위한 연구 설계: 혁신 방법론 × 한국 주택시장

최근(2018~2026) 국제 학술지의 혁신적 방법론을 조사·검증하고, 이를 한국
주택시장(공급 파이프라인 + 인구·가구 수요 + 비이성적 과열)에 접목한
**투고 가능한 연구 설계 5건**과 **타깃 저널 전략**을 제시한다. 인용된
문헌은 웹 조사에서 원문 대조로 검증된 것들이다.

---

## 0. 전략 총론 — 세 가지 원칙

1. **"한국 단독 연구도 SSCI에 실린다 — 단, 방법 혁신 또는 제도적
   고유성과 결합될 때."** 검증된 직접 선례:
   - Lee & Lee (2019, *J. of Housing Economics* 46, 101654) — KB 15개
     지역지수의 Diebold-Yilmaz 연결성 네트워크 (한국 단독)
   - Hwang & Suh (2021, *Emerging Markets Finance and Trade* 57(2),
     591-609) — TVP-VAR 동적 연결성 (한국 단독)
   - Pontines (2021, *Empirical Economics* 61(3), 1311-1350) — 한국
     LTV 규제의 실물 효과, 부호제약 SVAR (한국 단독)
   - Brunstein, Casamatta & Giannoni (2025, *J. of Housing Economics*
     67) — 코르시카 단일 지역이지만 DML+causal forest 방법 혁신으로 게재
   → 한국의 **전세·갭투자·청약·규제지역 제도는 세계에 없는 자연실험**
   이므로, 방법 혁신과 결합하면 단일국가 페널티를 상쇄한다.
2. **네거티브 결과를 아는 것이 설계 경쟁력이다.** 최신 벤치마크의 검증된
   교훈: 주택가격 "예측"에서는 GNN이 트리 기반(LightGBM/CatBoost)을
   이기지 못하고(Geerts 외, GNNs4HPP 벤치마크), 지리 인접 kNN 그래프의
   STGCN은 비그래프 베이스라인보다도 열세이며(HouseTS 벤치마크, arXiv
   2506.00765), 단순 선형(DLinear/ARDL)이 트랜스포머를 지배한다.
   → **예측 정확도로 승부하지 말 것.** 승부처는 (a) 경제적 연결
   구조의 식별(전이 네트워크), (b) 인과 효과의 이질성, (c) 새로운
   측정(텍스트·확정물량), (d) 정책 임계값 — 즉 "무엇을 새로 잴 수
   있는가"다.
3. **재현성 = 심사 통과율.** 본 저장소처럼 코드·데이터 파이프라인을
   공개하는 것이 최신 관행(GNNHAR·GNNs4HPP·HouseTS 모두 GitHub 공개;
   `exuber` R 패키지는 그 자체가 Dallas Fed WP로 출판됨).

---

## 1. 검증된 방법론 프런티어 지도

| 클러스터 | 검증된 대표 문헌 (저널) | 핵심 아이디어 | 한국 이식성 |
|---|---|---|---|
| 뉴스 감성→주택가격 | Soo (2018, *Review of Financial Studies*) — 지역 신문 논조로 34개 도시 주택 심리 정량화, 가격을 선행 | 텍스트가 기대심리의 고빈도 대리변수 | ◎ 네이버 뉴스·부동산 카페 |
| 검색량 예측력 | Møller, Pedersen, Schütte & Timmermann (2024, *Management Science*) — 구글 검색으로 주택가격 예측 가능성 | 검색 = 수요의 실시간 그림자 | ◎ 네이버 데이터랩 |
| LLM 기대 생성 | *J. of Monetary Economics* (2025) — LLM으로 인플레이션 기대 생성·보정 | 서베이 없는 기대 측정의 신영역 | ○ 한국어 LLM+뉴스 코퍼스 |
| 실시간 버블 모니터링 | Whitehouse, Harvey & Leybourne (2023, *Oxford Bulletin of Economics and Statistics*) — 실시간 CUSUM류 모니터링; PSY 대비 검정력 개선 | 폭발 시작을 더 빨리, 가짜 경보 없이 | ◎ 본 저장소 GSADF에 추가 |
| 패널 폭발근 검정 | Pavlidis 외 (2016, *JREFE* 53(4)) + `exuber` (Vasilopoulos·Pavlidis·Martínez-García, Dallas Fed WP) | 다지역 동시 과열 검정 | ◎ 시도 패널 |
| 연결성 네트워크 | Lee & Lee (2019, *JHE*); Hwang & Suh (2021, *EMFT*) — **연결성 급증은 붐 국면의 시그니처**, 강남3구 연결성이 전체를 수개월 선행 | 전이 구조 자체가 조기경보 정보 | ◎ 선례 확장 |
| GNN 스필오버 | Zhang, Pu, Cucuringu & Dong (2025, *International J. of Forecasting* 41) — GNNHAR: HAR을 내포하는 그래프 신경망 변동성 스필오버, 코드 공개 | 비선형 네트워크 전이의 검정 가능한 내포 구조 | ○ 지수 변동성으로 |
| 인과 ML 정책평가 | Brunstein 외 (2025, *JHE* 67) — DML+causal forest, Airbnb 탄력성 0.21, 공간 이질성; Miller (2020, *JEEM* 103, 102337) — causal forest의 **스태거드 도입·시간가변 효과** 확장; Grodecka-Messi & Hull (arXiv 2203.14751) — DML 세금 자본화(947개 고차원 통제로 추정치 2배+, Tiebout 이질성 검정); Chen 외 (arXiv 2606.02795) — EMCS 검증: causal ML은 처치 표본 ≥~3,000일 때 우위, naive TWFE 최악 | 평균효과가 아닌 **효과의 지도**를 그림 | ◎ 규제지역 지정 자연실험 |
| 시차의 내생성 | Oh & Yoon (2020, *J. of Financial Economics* 135(1)) — 공사기간의 실물옵션 채널, 침체기 준공 지연 | 파이프라인 시차는 상태변수 | ◎ 본 저장소 확장 |
| 버블 전염 | Bago 외 (2021, *JRFM* 14(7)) — GSADF+DCC-GARCH+버블 이동, 일본 사례 (단, MDPI/ESCI급 저널) | 과열의 공간 전파 추적 | ◎ 수도권→지방 |

---

## 2. 연구 설계 5건 (투고 단위)

### D1. "예정된 공급절벽": 파이프라인 확정물량 기반 주택시장 조기경보

- **연구질문**: 인허가→착공→준공의 분포시차로 계산한 *선행 확정공급*
  (locked-in supply)이 기존 지표(BSADF, 가격모멘텀, 신용)보다 가격 급등
  을 얼마나 먼저·정확히 예보하는가?
- **방법**: 본 저장소 프레임워크의 학술판 — ① NNLS 분포시차(+Oh-Yoon
  식 국면 의존 시차: 붐/침체 국면별 커널), ② 선행 수급률 지표, ③
  Alessi-Detken 손실함수 임계값, ④ Whitehouse-Harvey-Leybourne 실시간
  모니터링과의 경마(horse race), ⑤ 시도 패널 확장으로 사건 수 확보.
- **데이터**: 국토부 인허가·착공·준공(2011~, 아파트, 시도), 통계청
  장래가구추계, 부동산원 실거래지수, 한은 금리.
- **기여(novelty)**: 공급 파이프라인의 "이미 결정된 미래" 정보를 조기
  경보에 정식 편입한 연구는 국제적으로 공백. 미국(인허가→착공 1.4개월)
  과 달리 한국 아파트는 시차가 길어 정보가치가 극대화되는 시장.
- **타깃 저널**: 1순위 *Journal of Housing Economics* (한국 단독+방법
  결합 선례 확인) / 2순위 *Journal of Real Estate Finance and
  Economics* (exuberance 계보) / 3순위 *International Journal of
  Housing Markets and Analysis* (지역시장 특화, 등재등급은 SSCI 아닌
  ESCI급이므로 최후순위).
- **리스크**: 급등 사건 수 부족 → 시도 패널 필수. 착공통계 2011년 이후
  제약 → 인허가→준공 단일 단계로 1990년대까지 소급하는 강건성 부록.

### D2. 서사가 가격을 움직이는가: 한국 주택시장의 텍스트 기대지표와 과열의 인과 동학

- **연구질문**: 뉴스·커뮤니티 텍스트로 만든 서사(narrative) 지표가
  (a) BOK 전망CSI를 얼마나 선행하는가, (b) GSADF 폭발 구간의 시작을
  예측하는가, (c) 기대→가격의 인과 기여는 어느 정도인가?
- **방법**: Soo(2018, RFS)의 지역 신문 감성 사전 접근 + LLM 기반 서사
  분류(영끌/공급부족/규제/전세불안 등 주제별 서사 강도)를 결합. 검색량
  (Møller 외 2024, Management Science 계보)을 고빈도 검증축으로. BOK
  이슈노트 2025-15의 반사실(기대 중립 시 상승률 24%→11%)을 텍스트
  지표로 재추정 — 서베이 기반 결과의 텍스트 기반 재현이라는 명확한
  검증 구도.
- **데이터**: 네이버 뉴스 아카이브(지역 태깅), 부동산 커뮤니티, 네이버
  데이터랩 검색량, 전망CSI, 부동산원 주간지수(고빈도 매칭).
- **기여**: 비영어권·고과열 시장에서 서사→가격 인과 경로의 최초 정량화
  급. LLM으로 한국어 서사를 측정하는 방법론 자체가 수출 가능.
- **타깃 저널**: 1순위 *Real Estate Economics* (Cepni 외 2026 뉴스-주택
  예측력 논문 게재 — 주제 적합 확인) / 2순위 *Journal of Housing
  Economics* / 3순위 *Housing Studies* (서사·담론 전통).
- **리스크**: 텍스트→가격의 역인과 — 검색량·뉴스의 고빈도 선행성과
  외생적 뉴스 충격(정책 발표일) 이벤트 스터디로 방어.

### D3. 과열은 네트워크로 온다: 경제적 연결 그래프 기반 전이 조기경보

- **연구질문**: 한국 지역 주택시장의 가격·과열 전이는 지리 인접이
  아니라 **경제적 연결**(갭투자 자금 흐름, 통근권, 학군 대체성, 청약
  경쟁 중복)을 따라 일어나는가? 연결성의 급증은 과열의 선행 신호인가?
- **방법**: Lee & Lee(2019, JHE)·Hwang & Suh(2021, EMFT)의 연결성
  선례를 3중 확장 — ① 패널 GSADF(`exuber`)로 지역별 폭발 시점 식별,
  ② TVP-VAR 연결성과 **버블 전이 타이밍**의 결합(과열이 어느 노드에서
  어느 노드로 이동하는가), ③ GNNHAR(Zhang 외 2025, IJF — 코드 공개)식
  그래프 모형에서 지리 엣지 vs 경제 엣지의 성능·구조 비교. HouseTS의
  네거티브 결과(지리 kNN 그래프 열세)가 "경제 엣지" 설계의 공백 근거.
- **데이터**: 부동산원/KB 시군구 지수, 자금조달계획서 기반 갭투자 비율,
  통근 OD(통계청), 학군·청약 데이터.
- **기여**: "그래프를 학습이 아니라 **경제 제도**로 설계"하는 관점 전환.
  강남3구 연결성의 선행성(Hwang & Suh)을 전이 예측으로 격상.
- **타깃 저널**: 1순위 *International Journal of Forecasting* (GNNHAR
  게재 — 방법 확장 적합) / 2순위 *Emerging Markets Finance and Trade*
  (한국 선례) / 3순위 *Journal of Housing Economics*.
- **리스크**: GNN 성능 우위가 안 나올 수 있음 — 예측이 아닌 **전이
  구조 식별·조기경보 리드타임**을 주장의 축으로 설계(원칙 2).

### D4. 규제지역 지정의 인과 효과 지도: 스태거드 자연실험 × causal forest

- **연구질문**: 조정대상지역·투기과열지구 지정(2017~2023 수십 차례
  시차 지정·해제)의 가격·거래·풍선효과는 어디서, 누구에게, 얼마나
  이질적인가? "지정"이라는 신호 자체의 기대 효과는?
- **방법**: 시군구×월 패널의 스태거드 DID(최신 이질성-강건 추정량) +
  causal forest CATE(Brunstein 외 2025 JHE의 설계 이식). **스태거드
  도입에서 코호트·노출기간·횡단면 효과를 동시에 가변화하는 causal
  forest 확장은 Miller(2020, JEEM 103)가 정확한 방법 템플릿** — 규제
  지역 지정이 바로 그 구조다. Chen 외의 EMCS 교훈 반영 — 처치
  시군구×기간 표본이 충분(≥3,000 처치 관측)함을 사전 확인,
  generalized DID를 벤치마크로 병행. Pontines(2021, Empirical
  Economics)의 거시 SVAR 결과를 미시 인과 지도로 보완하는 구도.
- **데이터**: 국토부 실거래 마이크로데이터(지번·계약일), 지정·해제
  고시일, 자금조달계획서, 주민등록 이동.
- **기여**: 세계적으로 희귀한 "반복·시차·공간 명시적" 주택규제 자연
  실험의 이질 효과 지도 — 풍선효과의 인과 정량화.
- **타깃 저널**: 1순위 *Journal of Housing Economics* / 2순위
  *Empirical Economics* (한국 LTV 선례) / 3순위 *Regional Science and
  Urban Economics* (공간 정책 평가 전통, 난도 높음).
- **리스크**: 지정 기준이 가격 급등 자체(내생) — 지정 기준 점수의
  경계 불연속(RD)과 병행하면 식별이 단단해짐.

### D5. 전세 레버리지 사이클: 내재 LTV와 버블의 점화·전염

- **연구질문**: 전세가율(=갭투자의 내재 LTV)의 상승은 폭발적 가격
  구간의 발생 확률·강도·전염 속도를 높이는가? 임계 전세가율이 존재
  하는가?
- **방법**: ① 시군구 패널 GSADF로 폭발 에피소드 라벨링, ② 전세가율·
  갭투자 비중을 상태변수로 한 threshold/스무드 전환 패널 로짓(임계
  전세가율 추정), ③ Bago 외(2021)식 버블 전염 분석을 국내 지역 간
  (수도권→지방)으로 이식, ④ Geanakoplos(2012 AER)·BoE ABM의 레버리지
  명제를 전세 채널로 검정하는 프레이밍.
- **데이터**: 부동산원 매매·전세지수(전세가율), 확정일자 전세 계약
  DB, 자금조달계획서 갭투자 식별, 보증사고(역전세) 통계.
- **기여**: 전세는 문헌에 없는 **비은행 내재 레버리지의 자연실험** —
  "레버리지가 버블을 만든다"는 국제 명제를 은행 규제 밖 채널로 검정
  하는 유일한 시장. 임계 전세가율은 그대로 거시건전성 트리거가 됨
  (본 저장소 §7 절차로 캘리브레이션).
- **타깃 저널**: 1순위 *Journal of Real Estate Finance and Economics*
  (exuberance 방법 계보) / 2순위 *Journal of Housing Economics* /
  3순위 *Emerging Markets Finance and Trade*.
- **리스크**: 전세가율의 내생성(가격 기대가 전세가율을 움직임) —
  전세 공급 충격(입주물량 = D1의 확정물량!)을 도구변수로 사용, 설계
  D1과 자연 연결.

---

## 3. 타깃 저널 매핑 (검증 사례 기반)

| 저널 | 등재 | 성향 | 한국 단독 수용성 (근거) |
|---|---|---|---|
| J. of Housing Economics (Elsevier) | SSCI | 주택 특화 실증·정책 | **높음** — Lee & Lee 2019 (한국), Brunstein 2025 (단일 지역+방법) |
| J. of Real Estate Finance and Economics (Springer) | SSCI | 부동산 금융·exuberance 계보 | 중간 — Pavlidis 2016 등 방법론 전통, 단일국은 방법 기여 필요 |
| Real Estate Economics (Wiley) | SSCI | 최상위 부동산 저널 | 중하 — Cepni 2026(뉴스-예측력) 등 주제 적합 시 도전 가치 |
| Empirical Economics (Springer) | SSCI | 응용 계량 전반 | **높음** — Pontines 2021 (한국 LTV) |
| Emerging Markets Finance and Trade | SSCI | 신흥시장 실증 | **높음** — Hwang & Suh 2021 (한국) |
| International J. of Forecasting | SCIE/SSCI급 | 예측 방법론 | 방법 기여가 축이면 국가 무관 — Zhang 2025 (GNNHAR) |
| Oxford Bulletin of Economics and Statistics | SSCI | 계량 방법 | 실시간 모니터링 방법 확장 시 — Whitehouse 외 2023 |
| Housing Studies / Urban Studies (T&F/Sage) | SSCI | 정책·제도·담론 포함 | 높음 — 제도 연구(전세·청약) 서사 결합에 적합 |
| Regional Science and Urban Economics | SSCI | 도시·공간 경제 상위 | 중 — 식별 엄격, D4의 도전 타깃 |
| Int. J. of Housing Markets and Analysis (Emerald) | ESCI급* | 지역 주택시장 사례 | 매우 높음 — 안전판 |
| J. of Risk and Financial Management (MDPI) | ESCI급* | 오픈액세스 | 높음 — 단 SSCI 아님 (Bago 2021), 권장하지 않음 |

\* 등재 구분은 변동 가능 — 투고 시점에 Web of Science Master Journal
List에서 재확인할 것.

## 4. 투고 전략 체크리스트

1. **페어링 전략**: 각 설계마다 "방법 저널(상위) ↔ 지역 저널(안전판)"
   2단 티어로 커버레터·프레이밍을 이원화. 상위 티어엔 방법 기여를,
   지역 티어엔 제도 고유성·정책 함의를 전면에.
2. **한국 제도의 세계화**: 전세·청약·규제지역은 각주가 아니라 **식별
   전략**으로 팔 것 ("a natural experiment unavailable elsewhere").
3. **재현 패키지 동봉**: 본 저장소(`housing_ews`)를 논문 부록 코드로
   발전 — GNNHAR·exuber·HouseTS처럼 공개 코드가 표준인 심사 환경.
4. **베이스라인 정직성**: DLinear/ARDL·트리 모델·generalized DID 등
   "강한 단순 모형"을 반드시 벤치마크에 포함 (원칙 2의 네거티브 결과
   문헌을 심사자가 알고 있음).
5. **순서 제안**: D1(본 저장소가 이미 80% 구현) → D5(D1의 확정물량을
   IV로 재사용) → D2(데이터 구축 병행) → D3 → D4(마이크로데이터 확보
   후). D1과 D5는 코드·데이터가 공유되어 한 해에 2편 투고 가능한 조합.
