# =====================================================================
# 8단계 추가 코드: 동일가중 벤치마크로 다시 검증 + 지수 데이터 점검 + 공시 몰림 분석
# 조건: 1~3단계와 7단계를 먼저 실행한 상태여야 합니다.
# 실행: %run -i step8_addon.py   (붙여넣기라면 맨 마지막 줄 "# ===== 끝 =====" 확인)
# =====================================================================
import contextlib
import matplotlib.pyplot as plt

# ---------- 함수 ----------
def build_ew_benchmark(ev, min_stocks=20):
    """자사주 공시 기업 전체를 매일 같은 비중으로 들고 있는 시장별 동일가중 지수(가격 수준).
    가격제한폭을 넘는 일간 변동(기업행위)은 제외. 그날 종목이 min_stocks 미만이면 비움."""
    code_mk = ev.drop_duplicates("stock_code").set_index("stock_code")["market"]
    rets = {}
    for code, p in PX.items():
        r = p["Close"].pct_change()
        rets[code] = r.where(r.abs() <= DAILY_LIMIT)
    R = pd.DataFrame(rets).sort_index()
    ew, counts = {}, {}
    for mk in ev.market.unique():
        cols = [c for c in R.columns if code_mk.get(c) == mk]
        n = R[cols].notna().sum(axis=1)
        daily = R[cols].mean(axis=1).where(n >= min_stocks).fillna(0)
        ew[mk] = 100 * (1 + daily).cumprod()
        counts[mk] = n
    return ew, counts

@contextlib.contextmanager
def use_benchmark(bm):
    """IDX(비교 지수)를 잠시 다른 벤치마크로 바꿔서 기존 함수를 그대로 재사용. 끝나면 원래대로 복원."""
    saved = dict(IDX)
    try:
        IDX.clear(); IDX.update(bm); yield
    finally:
        IDX.clear(); IDX.update(saved)

def test_ew_benchmark():
    _px, _idx = dict(PX), dict(IDX)
    try:
        idx = pd.bdate_range("2024-01-01", periods=10)
        PX.clear()
        PX["A"] = pd.DataFrame({"Close": 100 * 1.01 ** np.arange(10), "Volume": 1.0}, index=idx)
        PX["B"] = pd.DataFrame({"Close": 100 * 1.03 ** np.arange(10), "Volume": 1.0}, index=idx)
        c = 100 * 1.02 ** np.arange(10); c[5:] *= 0.5            # 5일째 -50% (기업행위) → 제외돼야 함
        PX["C"] = pd.DataFrame({"Close": c, "Volume": 1.0}, index=idx)
        ev = pd.DataFrame({"stock_code": ["A", "B", "C"], "market": ["KOSDAQ"] * 3})
        ew, _ = build_ew_benchmark(ev, min_stocks=2)
        r = ew["KOSDAQ"].pct_change()
        assert abs(r.iloc[1] - 0.02) < 1e-12          # (1%+3%+2%)/3
        assert abs(r.iloc[5] - 0.02) < 1e-12          # C 제외 → (1%+3%)/2
        with use_benchmark({"X": pd.Series([1.0])}):
            assert list(IDX) == ["X"]
    finally:
        PX.clear(); PX.update(_px); IDX.clear(); IDX.update(_idx)
    assert set(IDX) == set(_idx), "IDX 복원 실패"
    print("통과: test_ew_benchmark ✅")

test_ew_benchmark()

# 7단계 변수가 없으면 다시 만든다
if "sig" not in globals() or "plc_ev" not in globals():
    SIG_TYPES, SIG_MARKETS, SLOTS = ['신탁'], ['KOSDAQ', 'KOSPI'], 20
    sig = ev_res[ev_res.type.isin(SIG_TYPES) & ev_res.market.isin(SIG_MARKETS) & (ev_res.status == 'ok')]
    plc_ev = placebo_events(sig, draws=1, hold=HOLD, seed=0)

# =====================================================================
# 8-1. 지수 데이터 점검: 튀는 값이 있나
# =====================================================================
print('【8-1. 시가총액 지수 점검】')
for mk, s in IDX.items():
    r = s.pct_change().dropna()
    vol = r.groupby(r.index.year).std().mul(np.sqrt(252) * 100).round(1)
    print(f'\n{mk}: 연도별 변동성(%) {vol.to_dict()}')
    big = r[r.abs() > 0.08]
    if len(big):
        print(f'  ⚠️ 하루 ±8% 넘는 날 {len(big)}일 — 실제 폭락/폭등인지, 데이터 오류인지 확인 필요:')
        display((big * 100).round(2).to_frame('일간 변동%').head(10))
    else:
        print('  하루 ±8% 넘는 날 없음')

# =====================================================================
# 8-2. 동일가중 벤치마크 만들기
# =====================================================================
EW, EW_N = build_ew_benchmark(ev_res)
print('\n【8-2. 동일가중 벤치마크】')
for mk in EW:
    print(f'{mk}: 하루 평균 {EW_N[mk].mean():.0f}개 종목으로 구성')
print('※ 자사주 공시 기업 전체(미래에 공시할 기업 포함)로 만든 비교 기준입니다. 매매 신호가 아니라 비교용이므로 문제없습니다.')

# =====================================================================
# 8-3. 개별 매매: 동일가중 대비 초과수익 + 플라시보
# =====================================================================
with use_benchmark(EW):
    ev_res['trade_ew'] = trades_with(ev_res, delay=1, hold=HOLD, cost=COST)
    plc_ew = placebo_trades(ev_res, draws=5, hold=HOLD, cost=COST)
rows = []
for (mk, tp), sub in ev_res.groupby(['market', 'type']):
    e = sub.trade_ew.dropna()
    p = plc_ew[(plc_ew.market == mk) & (plc_ew.type == tp)].trade_ret.dropna()
    rows.append({'그룹': f'{mk}-{tp}', '공시 N': len(e), '공시 평균%': round(e.mean()*100, 2),
                 '공시 중앙값%': round(e.median()*100, 2), '무작위 평균%': round(p.mean()*100, 2),
                 '차이%p': round((e.mean() - p.mean())*100, 2), '차이 t': round(welch_t(e, p), 2)})
print('\n【8-3. 동일가중 대비 매매 수익 (t0+1 진입, 20일 보유, 비용 반영)】')
display(pd.DataFrame(rows).set_index('그룹'))
print('→ 이제 "무작위 평균"이 0 근처여야 정상. 여전히 크게 마이너스면 벤치마크로도 설명 안 되는 무언가가 있음.')

# =====================================================================
# 8-4. 포트폴리오: 동일가중 헤지
# =====================================================================
with use_benchmark(EW):
    pf_ew  = portfolio_backtest(sig,    delay=1, hold=HOLD, cost=COST, slots=SLOTS)
    pfp_ew = portfolio_backtest(plc_ev, delay=1, hold=HOLD, cost=COST, slots=SLOTS)
ew_mix = pd.concat([EW[mk].pct_change() for mk in EW], axis=1).mean(axis=1).reindex(pf_ew.index)
table8 = pd.concat([perf(pf_ew.ret, '전략(주식만)'), perf(pf_ew.hedged, '전략(동일가중 헤지)'),
                    perf(pfp_ew.ret.reindex(pf_ew.index), '플라시보(주식만)'),
                    perf(pfp_ew.hedged.reindex(pf_ew.index), '플라시보(동일가중 헤지)'),
                    perf(ew_mix, '동일가중 벤치마크')], axis=1).T
print('\n【8-4. 포트폴리오 성과 (동일가중 기준)】')
display(table8)

fig, ax = plt.subplots(figsize=(10, 5))
for s, lab in [(pf_ew.ret, '전략(주식만)'), (pf_ew.hedged, '전략(동일가중 헤지)'),
               (pfp_ew.hedged.reindex(pf_ew.index), '플라시보(동일가중 헤지)'), (ew_mix, '동일가중 벤치마크')]:
    ax.plot((1 + s.fillna(0)).cumprod(), label=lab)
ax.axhline(1, c='gray', lw=0.8); ax.legend(); ax.set_title('동일가중 벤치마크 기준 수익 곡선'); plt.show()

# =====================================================================
# 8-5. 연도별
# =====================================================================
yr8 = pd.DataFrame({
    '전략(헤지)%':     pf_ew.hedged.groupby(pf_ew.index.year).apply(lambda r: (1 + r).prod() - 1),
    '플라시보(헤지)%': pfp_ew.hedged.reindex(pf_ew.index).groupby(pf_ew.index.year).apply(lambda r: (1 + r).prod() - 1),
    '동일가중%':       ew_mix.groupby(pf_ew.index.year).apply(lambda r: (1 + r.fillna(0)).prod() - 1),
}).mul(100).round(1)
yr8['전략-플라시보 %p'] = (yr8['전략(헤지)%'] - yr8['플라시보(헤지)%']).round(1)
print('\n【8-5. 연도별 (동일가중 기준)】')
display(yr8)

# =====================================================================
# 8-6. 공시 몰림: 신탁 공시가 한꺼번에 쏟아질 때 성과가 나쁜가
# =====================================================================
all_sig_dates = ev_res.loc[ev_res.type.isin(SIG_TYPES), 'event_date'].sort_values().values
def crowd(d, days=30):   # 공시일 이전 30일 동안 나온 신탁 공시 수 (과거 정보만 사용)
    d = np.datetime64(d)
    return int(((all_sig_dates < d) & (all_sig_dates >= d - np.timedelta64(days, 'D'))).sum())
sig8 = ev_res.loc[sig.index].copy()
sig8['몰림'] = sig8.event_date.apply(crowd)
sig8['몰림 구간'] = pd.qcut(sig8['몰림'].rank(method='first'), 3, labels=['적음', '보통', '많음'])
print('\n【8-6. 직전 30일 신탁 공시 수 구간별 매매 수익 (동일가중 대비)】')
display(sig8.groupby('몰림 구간').trade_ew.apply(trade_stats).unstack()
        .join(sig8.groupby('몰림 구간')['몰림'].agg(['min', 'max']).rename(columns={'min': '공시수 최소', 'max': '공시수 최대'})))
print('→ "많음" 구간이 확실히 나쁘면, 공시가 몰리는 시기(보통 시장 급락기)를 피하는 규칙을 검토할 수 있음.')
print('  단, 이건 결과를 보고 규칙을 만드는 것이라 전체 기간에서 다시 확인해야 합니다.')

print('\n8단계 완료 ✅')
# ===== 끝 =====
