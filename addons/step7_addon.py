# =====================================================================
# 7단계 추가 코드 (v2 노트북에 붙여넣기용) — 이 셀 하나로 전부 실행됩니다
# 조건: 1~3단계(가격 처리)를 먼저 실행한 상태여야 합니다.
# 붙여넣은 뒤 맨 마지막 줄이 "# ===== 끝 =====" 인지 확인하세요 (잘림 방지).
# =====================================================================
import matplotlib.pyplot as plt
# ---------- 6. 포트폴리오 백테스트 ----------
def placebo_events(ev, draws=1, hold=20, gap=40, seed=0):
    """플라시보용 가짜 이벤트(같은 종목, 공시와 gap거래일 이상 떨어진 무작위 거래일)를 이벤트 형식으로 반환."""
    _need_cache()
    rng = np.random.default_rng(seed)
    rows = []
    for code, sub in ev.groupby("stock_code"):
        p = PX.get(code)
        if p is None:
            continue
        mk = sub.market.iloc[0]
        idx = align(p["Close"], IDX[mk])[0].index
        ev_pos = [event_pos(idx, d) for d in sub.event_date]
        cand = [k for k in range(25, len(idx) - hold - 2) if all(abs(k - e) > gap for e in ev_pos)]
        for _, r in sub.iterrows():
            if cand:
                for k in rng.choice(cand, size=min(draws, len(cand)), replace=False):
                    rows.append({"stock_code": code, "market": mk, "type": r["type"],
                                 "event_date": idx[k], "liq": r.get("liq", np.nan)})
    return pd.DataFrame(rows)

def position_returns(ev, delay=1, hold=20, cost=0.003):
    """각 매매의 일별 수익(보유 중인 날만). 진입: t0+delay 종가, 청산: 그로부터 hold일 뒤 종가.
    비용은 진입 다음 날과 청산일에 절반씩 차감."""
    _need_cache()
    rows = []
    for j, (i, r) in enumerate(ev.iterrows()):
        p = PX.get(r.stock_code)
        if p is None:
            continue
        s, m = align(p["Close"], IDX[r.market])
        pos = event_pos(s.index, r.event_date)
        a, b = pos + delay, pos + delay + hold
        if b >= len(s):
            continue
        rs = s.pct_change().iloc[a + 1:b + 1]
        if np.any(np.abs(rs.values) > DAILY_LIMIT):
            continue
        rm = m.pct_change().iloc[a + 1:b + 1]
        c = np.zeros(len(rs)); c[0] += cost / 2; c[-1] += cost / 2
        rows.append(pd.DataFrame({"date": rs.index, "pid": j,
                                  "ret": rs.values - c, "excess": rs.values - rm.values - c}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "pid", "ret", "excess"])

def portfolio_backtest(ev, delay=1, hold=20, cost=0.003, slots=20):
    """매일 보유 중인 종목에 1/max(보유 종목 수, slots)씩 배분. 빈 슬롯은 현금(수익 0).
    반환: 일별 DataFrame(ret=주식만 보유, hedged=지수 헤지 시 초과수익, n=보유 종목 수)."""
    pr = position_returns(ev, delay, hold, cost)
    cal = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for v in IDX.values()])))
    if pr.empty:
        return pd.DataFrame(index=cal, data={"ret": 0.0, "hedged": 0.0, "n": 0})
    g = pr.groupby("date").agg(ret=("ret", "sum"), hedged=("excess", "sum"), n=("pid", "count"))
    cal = cal[(cal >= g.index.min()) & (cal <= g.index.max())]
    g = g.reindex(cal).fillna(0)
    w = np.maximum(g["n"], slots)
    return pd.DataFrame({"ret": g["ret"] / w, "hedged": g["hedged"] / w, "n": g["n"]}, index=cal)

def perf(r, name=""):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    yrs = max(len(r) / 252, 1e-9)
    return pd.Series({"연수익률%": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
                      "변동성%": round(r.std() * np.sqrt(252) * 100, 2),
                      "샤프": round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else np.nan,
                      "최대낙폭%": round((eq / eq.cummax() - 1).min() * 100, 2),
                      "누적%": round((eq.iloc[-1] - 1) * 100, 1)}, name=name)

def test_portfolio_backtest():
    _px, _idx = dict(PX), dict(IDX)          # 3단계 캐시 백업
    try:
        idx = pd.bdate_range("2024-01-01", periods=60)
        r = np.full(60, 0.01); r[30] = 0.2; r[31] = 0.2      # t0, t0+1 급등 → 제외돼야 함
        PX.clear(); IDX.clear()
        PX["000001"] = pd.DataFrame({"Close": 100 * np.cumprod(1 + r), "Volume": 1.0}, index=idx)
        IDX["KOSDAQ"] = pd.Series(100.0, index=idx)
        ev = pd.DataFrame({"stock_code": ["000001"], "market": ["KOSDAQ"], "event_date": [idx[30]]})
        p1 = portfolio_backtest(ev, delay=1, hold=5, cost=0.0, slots=1)
        assert abs((1 + p1.ret).prod() - 1.01 ** 5) < 1e-9
        p2 = portfolio_backtest(ev, delay=1, hold=5, cost=0.0, slots=2)
        assert abs((1 + p2.ret).prod() - 1.005 ** 5) < 1e-9
        pc = portfolio_backtest(ev, delay=1, hold=5, cost=0.01, slots=1)
        assert (1 + pc.ret).prod() < (1 + p1.ret).prod()
    finally:
        PX.clear(); PX.update(_px); IDX.clear(); IDX.update(_idx)   # 캐시 복원
    print("통과: test_portfolio_backtest ✅")

test_portfolio_backtest()

# ============================================================
# 7-0
# ===== 7단계 설정 =====
SIG_TYPES   = ['신탁']               # 매매할 공시 유형
SIG_MARKETS = ['KOSDAQ']    # 매매할 시장 (['KOSDAQ']만 넣어 비교해 보세요)
SLOTS       = 20                     # 최대 동시 보유 종목 수 (종목당 최대 비중 = 1/SLOTS)
PARTICIPATION = 0.05                 # 하루 거래대금의 5% 이상은 사지 않는다고 가정 (운용 규모 계산용)

sig = ev_res[ev_res.type.isin(SIG_TYPES) & ev_res.market.isin(SIG_MARKETS) & (ev_res.status == 'ok')]
plc_ev = placebo_events(sig, draws=1, hold=HOLD, seed=0)
print(f'신호 {len(sig)}건, 플라시보 {len(plc_ev)}건')

# ============================================================
# 7-1
# ===== 7-1. 성과 요약 =====
pf  = portfolio_backtest(sig,    delay=1, hold=HOLD, cost=COST, slots=SLOTS)
pfp = portfolio_backtest(plc_ev, delay=1, hold=HOLD, cost=COST, slots=SLOTS)
bench_mk = SIG_MARKETS[0] if len(SIG_MARKETS) == 1 else 'KOSDAQ'
bench = IDX[bench_mk].pct_change().reindex(pf.index)

table = pd.concat([perf(pf.ret, '전략(주식만)'), perf(pf.hedged, '전략(지수 헤지)'),
                   perf(pfp.ret.reindex(pf.index), '플라시보(주식만)'),
                   perf(pfp.hedged.reindex(pf.index), '플라시보(지수 헤지)'),
                   perf(bench, f'{bench_mk} 지수')], axis=1).T
display(table)
print(f'평균 보유 종목 {pf.n.mean():.1f}개 / 투자 비중 평균 {np.minimum(pf.n / SLOTS, 1).mean():.0%} '
      f'(나머지는 현금) / 종목이 하나도 없던 날 {(pf.n == 0).mean():.0%}')

# ============================================================
# 7-2
# ===== 7-2. 수익 곡선 =====
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
for s, lab in [(pf.ret, '전략(주식만)'), (pf.hedged, '전략(지수 헤지)'),
               (pfp.hedged.reindex(pf.index), '플라시보(지수 헤지)'), (bench, f'{bench_mk} 지수')]:
    ax[0].plot((1 + s.fillna(0)).cumprod(), label=lab)
ax[0].set_ylabel('누적 (시작=1)'); ax[0].legend(); ax[0].set_title('신탁 자사주 공시 포트폴리오')
ax[1].fill_between(pf.index, pf.n, step='mid', alpha=0.5); ax[1].axhline(SLOTS, ls='--', c='gray')
ax[1].set_ylabel('보유 종목 수'); plt.show()

# ============================================================
# 7-3
# ===== 7-3. 연도별 =====
yearly = pd.DataFrame({
    '전략(주식만)%':   pf.ret.groupby(pf.index.year).apply(lambda r: (1 + r).prod() - 1),
    '전략(헤지)%':     pf.hedged.groupby(pf.index.year).apply(lambda r: (1 + r).prod() - 1),
    '플라시보(헤지)%': pfp.hedged.reindex(pf.index).groupby(pf.index.year).apply(lambda r: (1 + r).prod() - 1),
    f'{bench_mk}%':    bench.groupby(pf.index.year).apply(lambda r: (1 + r.fillna(0)).prod() - 1),
    '평균 보유 종목':  pf.n.groupby(pf.index.year).mean()}).round(3)
yearly.iloc[:, :4] = (yearly.iloc[:, :4] * 100).round(1)
display(yearly)

# ============================================================
# 7-4
# ===== 7-4. 민감도: 슬롯 수 · 비용 =====
rows = []
for sl in [10, 20, 40]:
    for c in [0.003, 0.01]:
        p_ = portfolio_backtest(sig, delay=1, hold=HOLD, cost=c, slots=sl)
        rows.append(perf(p_.hedged, f'슬롯 {sl} / 비용 {c:.1%}'))
display(pd.DataFrame(rows))

# ============================================================
# 7-5
# ===== 7-5. 운용 가능 규모 =====
buy_cap = (sig.liq * PARTICIPATION).dropna()
q = buy_cap.quantile([0.25, 0.5])
print(f'하루 거래대금의 {PARTICIPATION:.0%}만 산다고 할 때 한 종목에 넣을 수 있는 금액')
print(f'  하위 25%: {q[0.25]/1e6:,.0f}백만 원 / 중앙값: {q[0.5]/1e6:,.0f}백만 원')
print(f'→ 종목당 비중 1/{SLOTS}이면, 대부분의 종목을 제대로 살 수 있는 포트폴리오 규모 ≈ '
      f'{q[0.25] * SLOTS / 1e8:,.1f}억 원 (하위 25% 기준)')
print('  이보다 크면 소형주에서 원하는 만큼 못 사거나, 사는 순간 가격을 밀어 올려 수익이 줄어듭니다.')

print('\n7단계 완료 ✅')
# ===== 끝 =====
