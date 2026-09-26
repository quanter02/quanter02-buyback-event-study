# =====================================================================
# 추가 검증: README 6장(한계)·8장(다음 단계)에서 남겨 둔 점검
#   E1. 거래대금 5분위별 매매 수익 (동일가중 대비) — 전체 기간
#   E2. 거래대금 하위 40% 제외 포트폴리오
#   E3. 군집을 고려한 t값: 공시일·월 단위 군집, 포트폴리오 Newey-West
# run_local.py 가 7·8단계(+상장폐지 복원)를 돌린 뒤 같은 전역 공간에서 실행한다.
# =====================================================================

def nw_t(x, lags=20):
    """Newey-West(Bartlett) 표준오차로 계산한 평균의 t값. 보유기간이 겹치는 일별 수익용."""
    x = pd.Series(x).dropna().values
    n = len(x)
    if n < lags + 5:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(s / n)

def cluster_t(x, keys):
    """군집별 평균을 낸 뒤 t (노트북 summary_table 의 't(날짜군집)'과 같은 방식)."""
    s = pd.Series(np.asarray(x, dtype=float), index=range(len(x)))
    k = pd.Series(np.asarray(keys), index=s.index)
    ok = s.notna()
    return tstat(s[ok].groupby(k[ok]).mean())

TGT = ev_res[(ev_res.market == "KOSDAQ") & (ev_res.type == "신탁")].copy()
RET = "trade_ew" if "trade_ew" in TGT else "trade_ret"

print(f"\n{'#'*20} 추가 검증 {'#'*20}")
print(f"대상: 코스닥 신탁 {TGT[RET].notna().sum()}건, 수익 기준 = {RET} (동일가중 대비, 비용 {COST:.1%})")

# ---- E1. 거래대금 5분위 ----
lab = ["1(최소)", "2", "3", "4", "5(최대)"]
q_all = pd.qcut(ev_res.liq, 5, labels=lab)                              # 노트북 6-5와 같은 기준(전체 공시)
q_yr = ev_res.groupby("year").liq.transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
print("\n【E1-a. 거래대금 5분위 (전체 공시 기준 분위, 코스닥 신탁)】")
display(TGT.assign(q=q_all.loc[TGT.index]).groupby("q", observed=True)[RET].apply(trade_stats).unstack())
print("\n【E1-b. 거래대금 5분위 (연도 안에서 분위 → 거래대금의 시간 추세 제거)】")
display(TGT.assign(q=q_yr.loc[TGT.index].map(dict(enumerate(lab)))).groupby("q")[RET].apply(trade_stats).unstack())

cut40 = ev_res.liq.quantile(0.4)
liq_ok = TGT[TGT.liq >= cut40]
print(f"\n【E1-c. 하위 40% 제외 (거래대금 ≥ {cut40/1e8:,.1f}억 원/일)】")
display(pd.DataFrame({"전체": trade_stats(TGT[RET]), "하위40% 제외": trade_stats(liq_ok[RET])}).T)
print("연도별 (하위 40% 제외):")
display(liq_ok.groupby("year")[RET].apply(trade_stats).unstack()[["N", "평균%", "중앙값%"]])

# 플라시보도 같은 종목군(유동성)으로 맞춰 비교
if "plc_ew" in globals():
    p_all = plc_ew[(plc_ew.market == "KOSDAQ") & (plc_ew.type == "신탁")]
    liq_codes = set(liq_ok.stock_code)
    p_liq = p_all[p_all.stock_code.isin(liq_codes)] if "stock_code" in p_all else p_all
    print(f"  하위40% 제외 종목의 무작위 날짜 평균 {p_liq.trade_ret.mean()*100:.2f}% → "
          f"차이 {(liq_ok[RET].mean()-p_liq.trade_ret.mean())*100:+.2f}%p, t {welch_t(liq_ok[RET], p_liq.trade_ret):.2f} (단순)")

# ---- E2. 유동성 필터 포트폴리오 ----
sig_all = TGT[TGT.status.isin(["ok", "기간밖"])]
sig_liq = sig_all[sig_all.liq >= cut40]
print("\n【E2. 포트폴리오: 전체 vs 거래대금 하위 40% 제외 (슬롯 20)】")
rows = []
for name, s_ in [("전체", sig_all), ("하위40% 제외", sig_liq)]:
    for c in [0.003, 0.01]:
        p_ = portfolio_backtest(s_, delay=1, hold=HOLD, cost=c, slots=20)
        with use_benchmark(EW):
            h_ = portfolio_backtest(s_, delay=1, hold=HOLD, cost=c, slots=20)
        a, b = perf(p_.ret), perf(h_.hedged)
        rows.append({"신호": name, "비용": f"{c:.1%}", "N": len(s_), "주식만 연%": a["연수익률%"], "주식만 샤프": a["샤프"],
                     "주식만 MDD%": a["최대낙폭%"], "헤지 연%": b["연수익률%"], "헤지 샤프": b["샤프"],
                     "헤지 NW t": round(nw_t(h_.hedged), 2), "평균 투자비중%": round(np.minimum(p_.n / 20, 1).mean() * 100)})
display(pd.DataFrame(rows).set_index(["신호", "비용"]))

# ---- E3. 군집 t ----
x = TGT[RET]
print("\n【E3. t값 비교 (코스닥 신탁, 동일가중 대비 매매 수익)】")
tt = {"단순 t (매매 독립 가정)": tstat(x),
      "공시일 군집 t": cluster_t(x, TGT.event_date.dt.strftime("%Y-%m-%d")),
      "주(week) 군집 t": cluster_t(x, TGT.event_date.dt.strftime("%G-%V")),
      "월 군집 t": cluster_t(x, TGT.event_date.dt.strftime("%Y-%m"))}
with use_benchmark(EW):
    h_ = portfolio_backtest(sig_all, delay=1, hold=HOLD, cost=COST, slots=20)
    hp_ = portfolio_backtest(plc_ev, delay=1, hold=HOLD, cost=COST, slots=20)
diff = (h_.hedged - hp_.hedged.reindex(h_.index).fillna(0))
tt["포트폴리오 헤지 수익 NW t (lag 20)"] = nw_t(h_.hedged)
tt["포트폴리오 (전략−플라시보) NW t (lag 20)"] = nw_t(diff)
display(pd.Series(tt).round(2).to_frame("t"))
print("→ 월 군집·NW t가 2를 넘으면, 겹치는 보유기간·공시 몰림을 감안해도 유의하다고 볼 수 있음.")
print("\n추가 검증 끝 ✅")
