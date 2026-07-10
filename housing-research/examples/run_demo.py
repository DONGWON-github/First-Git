"""수급 불균형 조기경보 파이프라인 end-to-end 데모.

합성(한국형) 데이터로 다음을 순서대로 수행하고 output/에 차트·표를 남긴다.

1. 인허가→착공→준공 분포시차 추정 (진짜 시차 복원 검증 포함)
2. 향후 36개월 준공 물량 전망 (확정분 locked-in vs 시나리오 의존분 분해)
3. 가구 기반 필요공급량·수급갭 계산 + 선행 수급률(fwd_gap12) 산출
4. 실질가격 GSADF 폭발적 상승 검정 (과열 구간 시점 식별)
5. 신호접근법으로 지표별 최적 임계값(경계지점) 계산
6. 종합 조기경보지수 + 현재 경보 단계·경계지점까지 여유폭 보고

실행:  python examples/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from housing_ews import demand, ews, exuberance, gap, pipeline, thresholds
from synthetic_data import make_dataset

OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# ---- 차트 스타일 (dataviz 검증 팔레트, 라이트 모드) ----
C = {
    "permits": "#2a78d6",    # slot1 blue
    "starts": "#1baf7a",     # slot2 aqua
    "completions": "#eda100",  # slot3 yellow
    "required": "#008300",   # slot4 green
    "price": "#4a3aa7",      # slot5 violet
    "bsadf": "#e34948",      # slot6 red
    "gap_back": "#e87ba4",   # slot7 magenta (사후 수급률)
    "gap_fwd": "#eb6834",    # slot8 orange (선행 수급률)
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
    "st_good": "#0ca30c",
    "st_warn": "#fab219",
    "st_serious": "#ec835a",
    "st_critical": "#d03b3b",
}


def setup_fonts() -> None:
    for p in Path("/usr/share/fonts/truetype/nanum").glob("NanumGothic*.ttf"):
        fm.fontManager.addfont(str(p))
    plt.rcParams.update(
        {
            "font.family": "NanumGothic",
            "axes.unicode_minus": False,
            "figure.facecolor": C["surface"],
            "axes.facecolor": C["surface"],
            "savefig.facecolor": C["surface"],
            "text.color": C["ink"],
            "axes.labelcolor": C["ink2"],
            "xtick.color": C["muted"],
            "ytick.color": C["muted"],
            "axes.edgecolor": C["axis"],
            "font.size": 10,
        }
    )


def style_ax(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=C["ink"])
    ax.grid(axis="y", color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(C["axis"])
    ax.tick_params(length=0)


def end_label(ax, series: pd.Series, text: str, color: str, dy: float = 0.0) -> None:
    s = series.dropna()
    ax.annotate(
        text,
        xy=(s.index[-1], s.iloc[-1]),
        xytext=(6, dy),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="bold",
        va="center",
    )


def forward_locked_kernel(weights: np.ndarray, horizon: int) -> np.ndarray:
    """향후 horizon기간 하류 물량 중 '현재까지 관측된 상류'가 만드는 부분의
    컨볼루션 커널. kernel[j] = Σ_{h=1..H} w_{h+j}."""
    K = len(weights)
    kern = np.zeros(K)
    for j in range(K):
        hi = min(K, j + horizon + 1)
        kern[j] = weights[j + 1 : hi].sum()
    return kern


def forward_locked_series(
    permits: pd.Series,
    starts: pd.Series,
    p2s: np.ndarray,
    s2c: np.ndarray,
    horizon: int,
) -> pd.Series:
    """각 시점 t에서, 이미 관측된 착공·인허가만으로 확정된 향후 horizon개월
    준공 합계. (착공분 커널 + 미착공 인허가분 커널의 컨볼루션)"""
    k_starts = forward_locked_kernel(s2c, horizon)
    # 인허가 → (미래 착공 m≥1) → 준공 결합 커널
    K = len(p2s) + len(s2c)
    k_perm = np.zeros(K)
    for j in range(K):
        acc = 0.0
        for h in range(1, horizon + 1):
            for m in range(1, h + 1):
                if j + m < len(p2s) and h - m < len(s2c):
                    acc += p2s[j + m] * s2c[h - m]
        k_perm[j] = acc
    s_arr = starts.to_numpy(float)
    p_arr = permits.to_numpy(float)
    from_starts = np.convolve(s_arr, k_starts)[: len(s_arr)]
    from_permits = np.convolve(p_arr, k_perm)[: len(p_arr)]
    return pd.Series(from_starts + from_permits, index=starts.index)


def main() -> None:
    setup_fonts()
    data = make_dataset()
    idx = data["permits"].index
    now = idx[-1]

    # ================= 1. 파이프라인 시차 추정 =================
    model = pipeline.PipelineModel(
        permit_to_start_maxlag=30, start_to_completion_maxlag=48
    )
    model.fit(data["permits"], data["starts"], data["completions"])
    p2s, s2c = model.permit_to_start, model.start_to_completion

    def lag_stats(dist, true_w):
        lags = np.arange(len(true_w))
        t_total = true_w.sum()
        t_mean = (lags * true_w).sum() / t_total
        return (
            f"실현율 추정 {dist.realization_rate:.2f} (진값 {t_total:.2f}), "
            f"평균시차 추정 {dist.mean_lag:.1f}개월 (진값 {t_mean:.1f}), "
            f"R²={dist.r2:.3f}"
        )

    print("=" * 72)
    print("[1] 파이프라인 분포시차 추정")
    print("  인허가→착공 :", lag_stats(p2s, data["true_p2s"]))
    print("  착공→준공   :", lag_stats(s2c, data["true_s2c"]))

    # ================= 2. 준공 전망 (36개월) =================
    H = 36
    fcst = model.forecast_completions(H, permit_scenario=None)
    li_share_12 = fcst.locked_in.head(12).sum() / fcst.total.head(12).sum()
    li_share_24 = fcst.locked_in.head(24).sum() / fcst.total.head(24).sum()
    li_share_36 = fcst.locked_in.head(36).sum() / fcst.total.head(36).sum()
    print("\n[2] 준공(입주) 물량 전망 — 이미 결정된 비중(locked-in share)")
    print(f"  향후 12개월 {li_share_12:.0%} / 24개월 {li_share_24:.0%} / 36개월 {li_share_36:.0%}")
    print(f"  → 향후 1~2년 입주물량은 과거 착공·인허가로 사실상 확정되어 있음")

    # ================= 3. 수요·수급갭 =================
    req = demand.required_supply(data["households"], data["stock"])
    g = gap.supply_demand_gap(data["completions"], req["required"])
    gap12 = gap.rolling_gap_ratio(g, 12)

    locked12 = forward_locked_series(
        data["permits"], data["starts"], p2s.weights, s2c.weights, horizon=12
    )
    # 교차 검증: 표본 끝 시점의 커널 계산 == PipelineModel locked-in 전망
    ref = fcst.locked_in.head(12).sum()
    assert np.isclose(locked12.iloc[-1], ref, rtol=1e-6), (locked12.iloc[-1], ref)
    # 표본 시작 이전의 착공·인허가 이력이 없어 초기 구간은 확정물량이
    # 과소평가된다 → 최대 시차만큼 마스킹 (캘리브레이션 왜곡 방지)
    locked12.iloc[:60] = np.nan

    req_fwd12 = req["required"].rolling(12).sum().shift(-12)
    req_fwd12 = req_fwd12.fillna(float(req["required"].tail(6).mean()) * 12)
    fwd_gap12 = (locked12 / req_fwd12).rename("fwd_gap12")

    print("\n[3] 수급 현황")
    print(f"  최근 12개월 수급률(사후)  : {gap12.dropna().iloc[-1]:.2f}")
    print(f"  향후 12개월 수급률(선행)  : {fwd_gap12.dropna().iloc[-1]:.2f}"
          f"  ← 확정물량 {locked12.iloc[-1]/1e4:.1f}만호 / 필요 {req_fwd12.iloc[-1]/1e4:.1f}만호")

    # ================= 4. GSADF 과열 검정 =================
    log_price = np.log(data["price_real"])
    ex = exuberance.gsadf_test(log_price, lags=0, nreps=299, seed=11)
    print("\n[4] GSADF 폭발적 상승 검정 (실질가격, 로그)")
    print(f"  GSADF 통계량 {ex.gsadf_stat:.2f} vs 95% 임계값 {ex.gsadf_cv[0.95]:.2f}"
          f" → {'폭발적 구간 존재' if ex.gsadf_stat > ex.gsadf_cv[0.95] else '증거 없음'}")
    for s, e in ex.episodes:
        print(f"  폭발적 구간: {s:%Y-%m} ~ {e:%Y-%m}")
    print(f"  현재 BSADF 여유폭(margin) {ex.margin.iloc[-1]:+.2f} "
          f"({'임계 초과' if ex.is_explosive_now else '임계 이내'})")

    # ================= 5. 임계값 캘리브레이션 =================
    event = thresholds.label_surge_events(
        data["price_real"], horizon=12, surge_growth=0.08
    )
    mom6 = (np.log(data["price_real"]).diff(6) * 2).rename("mom6")  # 연율 환산

    cal_fwd = thresholds.optimal_threshold(fwd_gap12, event, direction="below")
    cal_back = thresholds.optimal_threshold(gap12, event, direction="below")
    cal_bsadf = thresholds.optimal_threshold(ex.margin, event, direction="above")
    cal_mom = thresholds.optimal_threshold(mom6, event, direction="above")

    inds = [
        ews.Indicator("선행 수급률(확정물량/필요, 12M)", fwd_gap12, cal_fwd),
        ews.Indicator("사후 수급률(준공/필요, 12M)", gap12, cal_back),
        ews.Indicator("GSADF 과열 여유폭(BSADF-cv95)", ex.margin, cal_bsadf),
        ews.Indicator("실질가격 모멘텀(6M 연율)", mom6, cal_mom),
    ]

    print("\n[5] 신호접근법 최적 임계값 (= 긴급 정책 경계지점)")
    for ind in inds:
        b = ind.calibration.best
        d = "이하" if ind.calibration.direction == "below" else "이상"
        print(
            f"  {ind.name:<28s} 임계값 {b.threshold:+.3f} {d} 발령 | "
            f"적중률 {b.hit_rate:.0%}, 오경보율 {b.false_alarm_rate:.0%}, "
            f"NSR {b.noise_to_signal:.2f}, 유용성 {b.usefulness:.2f}"
        )

    # ================= 6. 종합 경보·경계지점 보고 =================
    report = ews.policy_boundary_report(inds)
    print("\n[6] 현재 경보 판정")
    print(f"  종합 EWS 점수 {report['score']:.2f} → 경보 단계 "
          f"{report['level']} ({report['level_ko']})")
    tbl = report["table"].copy()
    print(tbl.to_string(index=False,
                        float_format=lambda v: f"{v:+.3f}" if abs(v) < 10 else f"{v:.1f}"))
    tbl.to_csv(OUT / "boundary_report.csv", index=False, encoding="utf-8-sig")
    report["score_series"].to_csv(OUT / "ews_score.csv", encoding="utf-8-sig")

    # ================= 차트 =================
    draw_charts(data, model, fcst, req, gap12, fwd_gap12, ex, report, now)
    print(f"\n차트/표 저장 완료 → {OUT}")


def draw_charts(data, model, fcst, req, gap12, fwd_gap12, ex, report, now) -> None:
    scale = 1e-3  # 호 → 천호

    # ---- 1. 파이프라인 흐름 + 준공 전망 ----
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=150)
    sm = lambda s: s.rolling(6, min_periods=1).mean() * scale
    ax.plot(sm(data["permits"]), color=C["permits"], lw=2)
    ax.plot(sm(data["starts"]), color=C["starts"], lw=2)
    ax.plot(sm(data["completions"]), color=C["completions"], lw=2)
    ax.plot(fcst.total * scale, color=C["completions"], lw=2, ls=(0, (4, 3)))
    ax.plot(fcst.locked_in * scale, color=C["ink2"], lw=1.6, ls=(0, (1.5, 2)))
    ax.axvline(now, color=C["axis"], lw=1)
    ax.annotate("현재", xy=(now, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", color=C["muted"], fontsize=9)
    end_label(ax, sm(data["permits"]), "인허가", C["permits"], 10)
    end_label(ax, sm(data["starts"]), "착공", C["starts"], -2)
    end_label(ax, fcst.total * scale, "준공 전망", C["completions"], 8)
    end_label(ax, fcst.locked_in * scale, "확정분", C["ink2"], -8)
    ax.legend(["인허가", "착공", "준공(실적)", "준공(전망)", "준공(확정분만)"],
              frameon=False, fontsize=9, loc="upper left", labelcolor=C["ink2"])
    style_ax(ax, "공급 파이프라인: 인허가 → 착공 → 준공 (월별, 천호, 6M 평활)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "01_pipeline.png")
    plt.close(fig)

    # ---- 2. 시차 분포 ----
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=150)
    for ax, dist, true_w, name in [
        (axes[0], model.permit_to_start, data["true_p2s"], "인허가 → 착공"),
        (axes[1], model.start_to_completion, data["true_s2c"], "착공 → 준공"),
    ]:
        lags = np.arange(len(dist.weights))
        bars = ax.bar(lags, dist.weights, color=C["permits"], width=0.75)
        (line,) = ax.plot(np.arange(len(true_w)), true_w, color=C["ink"],
                          lw=1.4, ls=(0, (3, 2)))
        style_ax(ax, f"{name} 시차 분포")
        ax.set_xlabel("시차 (개월)", fontsize=9)
        ax.annotate(
            f"평균 {dist.mean_lag:.0f}개월 · 실현율 {dist.realization_rate:.0%}",
            xy=(0.97, 0.9), xycoords="axes fraction", ha="right",
            fontsize=9, color=C["ink2"])
    axes[0].legend([bars, line], ["추정(NNLS)", "진짜 분포"], frameon=False,
                   fontsize=9, labelcolor=C["ink2"])
    fig.tight_layout()
    fig.savefig(OUT / "02_lag_distribution.png")
    plt.close(fig)

    # ---- 3. 공급 vs 필요공급, 수급률 ----
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.6), dpi=150, sharex=True)
    a0, a1 = axes
    sup12 = data["completions"].rolling(12).sum() * 1e-4
    req12 = req["required"].rolling(12).sum() * 1e-4
    a0.plot(sup12, color=C["completions"], lw=2)
    a0.plot(req12, color=C["required"], lw=2)
    a0.legend(["준공(12M 누계)", "필요공급(12M 누계)"], frameon=False,
              fontsize=9, labelcolor=C["ink2"])
    style_ax(a0, "연간 공급 vs 필요공급 (만호)")
    a0.set_ylim(bottom=0)

    a1.plot(gap12, color=C["gap_back"], lw=2)
    a1.plot(fwd_gap12, color=C["gap_fwd"], lw=2)
    a1.axhline(1.0, color=C["axis"], lw=1)
    th = report["table"].set_index("indicator")
    back_th = th.loc["사후 수급률(준공/필요, 12M)", "threshold"]
    a1.axhline(back_th, color=C["st_critical"], lw=1.2, ls=(0, (4, 3)))
    a1.annotate(f"사후 수급률 경계지점 {back_th:.2f} (캘리브레이션)",
                xy=(gap12.index[30], back_th), xytext=(0, -14),
                textcoords="offset points",
                color=C["st_critical"], fontsize=9, fontweight="bold")
    end_label(a1, gap12, "사후 수급률", C["gap_back"], 6)
    end_label(a1, fwd_gap12, "선행 수급률(확정물량)", C["gap_fwd"], -6)
    a1.legend(["사후(준공/필요)", "선행(확정물량/필요)"], frameon=False,
              fontsize=9, labelcolor=C["ink2"])
    style_ax(a1, "수급률: 1.0 미만 = 공급 부족 (12M 이동합 기준)")
    fig.tight_layout()
    fig.savefig(OUT / "03_gap.png")
    plt.close(fig)

    # ---- 4. 가격 + GSADF ----
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.6), dpi=150, sharex=True)
    a0, a1 = axes
    a0.plot(data["price_real"], color=C["price"], lw=2)
    for s, e in ex.episodes:
        a0.axvspan(s, e, color=C["st_critical"], alpha=0.10)
    if ex.episodes:
        s0, e0 = ex.episodes[0]
        a0.annotate("폭발적 구간(GSADF 95%)", xy=(s0, data["price_real"].max()),
                    xytext=(2, -2), textcoords="offset points",
                    color=C["st_critical"], fontsize=9, fontweight="bold")
    end_label(a0, data["price_real"], "실질가격", C["price"])
    style_ax(a0, "실질 주택가격지수와 폭발적 상승 구간 (2005=100)")

    a1.plot(ex.bsadf, color=C["bsadf"], lw=2)
    a1.plot(ex.cv["cv95"], color=C["muted"], lw=1.4, ls=(0, (4, 3)))
    end_label(a1, ex.bsadf, "BSADF", C["bsadf"], 6)
    end_label(a1, ex.cv["cv95"], "95% 임계값", C["muted"], -6)
    a1.legend(["BSADF(t)", "임계값(MC 95%)"], frameon=False, fontsize=9,
              labelcolor=C["ink2"])
    style_ax(a1, "BSADF 통계량 vs 임계값: 상회 = 과열(폭발) 신호")
    fig.tight_layout()
    fig.savefig(OUT / "04_gsadf.png")
    plt.close(fig)

    # ---- 5. 종합 EWS ----
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=150)
    score = report["score_series"]
    bands = [(0.0, 0.25, C["st_good"], "정상"), (0.25, 0.5, C["st_warn"], "주의"),
             (0.5, 0.75, C["st_serious"], "경계"), (0.75, 1.0, C["st_critical"], "심각")]
    for lo, hi, col, lab in bands:
        ax.axhspan(lo, hi, color=col, alpha=0.07)
        ax.annotate(lab, xy=(score.index[3], (lo + hi) / 2), fontsize=9,
                    color=col, fontweight="bold", va="center")
    (raw_line,) = ax.plot(score, color=C["permits"], lw=1.1, alpha=0.35,
                          drawstyle="steps-post")
    (ma_line,) = ax.plot(score.rolling(6, min_periods=1).mean(),
                         color=C["permits"], lw=2.2)
    ax.legend([raw_line, ma_line], ["월별 점수", "6M 평활"], frameon=False,
              fontsize=9, labelcolor=C["ink2"], loc="upper center")
    cur = float(score.dropna().iloc[-1])
    ax.plot([score.dropna().index[-1]], [cur], "o", color=C["ink"], ms=6)
    ax.annotate(f"현재 {cur:.2f} ({report['level_ko']})",
                xy=(score.dropna().index[-1], cur), xytext=(-8, 12),
                textcoords="offset points", ha="right", fontsize=10,
                fontweight="bold", color=C["ink"])
    style_ax(ax, "종합 조기경보지수: 가중 신호 합성 점수와 경보 단계")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    fig.tight_layout()
    fig.savefig(OUT / "05_ews.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
