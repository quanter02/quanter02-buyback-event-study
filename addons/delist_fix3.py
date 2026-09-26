# ===== 상장폐지 복원 재계산 v3: 최악 가정 + 스트레스 테스트 (%run -i) =====
import numpy as np, pandas as pd
# ===== 상장폐지 기업 복원 후 재계산 (전체 기간 셀을 실행한 같은 세션에서) =====
# 1) 지금은 '기타(E)'로 분류된 과거 상장사 공시를 되살린다
# 거래소(KRX) 데이터가 로그인 없이는 막혀 있어서 공시 당시 시장을 알 수 없음
# → 가장 보수적으로: 복원 공시를 전부 '코스닥'으로 넣는다 (코스닥 신탁 그룹에 불리한 최악 가정)
x = raw[(raw.corp_cls == 'E') & (raw.stock_code.fillna('').str.strip().str.len() == 6)].copy()
x['corp_cls'] = 'K'
extra = clean_events(x, DEDUP_DAYS)
extra['period'] = np.where(extra.event_date < SPLIT_DATE, f'~{SPLIT_DATE[:4]} 이전', f'{SPLIT_DATE[:4]}~')
extra['복원'] = True
print('복원 대상 이벤트 (전부 코스닥으로 가정):'); print(extra.groupby('type').size().to_string())

# ---- 스트레스 테스트: 데이터 없이도 할 수 있는 계산 ----
# 빠진 신탁 공시들이 평균 얼마나 잃어야 코스닥 신탁의 우위가 사라지나?
_obs = ev_res[(ev_res.market == 'KOSDAQ') & (ev_res.type == '신탁')].trade_ew.dropna()
_plc = plc_ew[(plc_ew.market == 'KOSDAQ') & (plc_ew.type == '신탁')].trade_ret.dropna().mean() if 'plc_ew' in globals() else 0.0
_n_miss = int((extra.type == '신탁').sum())
_edge = _obs.mean() - _plc
_breakeven = _plc - _edge * len(_obs) / max(_n_miss, 1)
print('\n=== 스트레스 테스트 ===')
print(f'관측된 코스닥 신탁 {len(_obs)}건: 평균 {_obs.mean()*100:.2f}%, 무작위 대비 우위 {_edge*100:.2f}%p')
print(f'빠진 신탁 공시 {_n_miss}건이 전부 코스닥이라고 가정할 때,')
print(f'→ 우위가 0이 되려면 빠진 공시들이 20일 동안 평균 {_breakeven*100:.1f}% (동일가중 대비) 잃어야 함')
print(f'   참고: 관측된 매매 중 {_breakeven*100:.0f}% 이하 손실 비율 = {(_obs <= _breakeven).mean():.1%}')

# ---- 주가를 받을 수 있는지 시험 ----
_codes = list(extra.stock_code.unique())
_ok = 0
for c in _codes[:5]:
    t = extra[extra.stock_code == c]
    p = _load_px_orig(c, t.event_date.min() - pd.Timedelta(days=30), t.event_date.min() + pd.Timedelta(days=30))
    _ok += int(p is not None and len(p) > 0)
print(f'\n상장폐지 종목 가격 시험: 5개 중 {_ok}개 성공')
if _ok == 0:
    print('→ 상장폐지 종목 가격을 구할 수 없어 여기서 멈춤. 위 스트레스 테스트로 판단하세요.')
else:
    new = [c for c in extra.stock_code.unique() if c not in PX_STORE and c not in MISSING]
    print(f'새로 받을 종목 {len(new)}개')
    for k, code in enumerate(new, 1):
        sub = extra[extra.stock_code == code]
        p = _load_px_orig(code, sub.event_date.min() - pd.Timedelta(days=250), sub.event_date.max() + pd.Timedelta(days=250))
        if p is None: MISSING.add(code)
        else: PX_STORE[code] = p
        if k % 50 == 0: _save()
    _save()
    print(f"복원 종목 중 가격 없음: {sum(c in MISSING for c in extra.stock_code.unique())}개")

    # 3) 상장폐지를 반영하는 매매 계산: 보유 중 가격이 끊기면 마지막 가격으로 청산 (정리매매 폭락도 그대로 반영)
    DATA_END = max(p.index.max() for p in PX_STORE.values())
    def _truncated(s, b):
        return len(s) > 0 and b >= len(s) and s.index[-1] < DATA_END - pd.Timedelta(days=10)

    def tradable_return(stock, mkt, event_date, hold=20, cost=0.003, delay=1):
        s, m = align(stock, mkt)
        pos = event_pos(s.index, event_date)
        a, b = pos + delay, pos + delay + hold
        if a >= len(s): return np.nan
        trunc = _truncated(s, b)
        if b >= len(s):
            if not trunc: return np.nan            # 표본 끝이라 아직 결과 없음
            b = len(s) - 1
            if b <= a: return -cost
        rs = s.pct_change().iloc[a + 1:b + 1]
        if not trunc and np.any(np.abs(rs.values) > DAILY_LIMIT): return np.nan
        return (s.iloc[b] / s.iloc[a] - 1) - (m.iloc[b] / m.iloc[a] - 1) - cost

    def position_returns(ev, delay=1, hold=20, cost=0.003):
        rows = []
        for j, (i, r) in enumerate(ev.iterrows()):
            p = PX.get(r.stock_code)
            if p is None: continue
            s, m = align(p["Close"], IDX[r.market])
            pos = event_pos(s.index, r.event_date)
            a, b = pos + delay, pos + delay + hold
            if a >= len(s): continue
            trunc = _truncated(s, b)
            if b >= len(s):
                if not trunc: continue
                b = len(s) - 1
                if b <= a: continue
            rs = s.pct_change().iloc[a + 1:b + 1]
            if not trunc and np.any(np.abs(rs.values) > DAILY_LIMIT): continue
            rm = m.pct_change().iloc[a + 1:b + 1]
            c = np.zeros(len(rs)); c[0] += cost / 2; c[-1] += cost / 2
            rows.append(pd.DataFrame({"date": rs.index, "pid": j, "ret": rs.values - c, "excess": rs.values - rm.values - c}))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "pid", "ret", "excess"])

    # 4) 전체 다시 계산
    events_all = pd.concat([events.assign(복원=False), extra], ignore_index=True)
    ev_res, ar_mat = build_ar_matrix(events_all, PRE, POST, HOLD, COST, verbose=False)
    EW, EW_N = build_ew_benchmark(ev_res)
    with use_benchmark(EW):
        ev_res['trade_ew'] = trades_with(ev_res, delay=1, hold=HOLD, cost=COST)
        plc_ew = placebo_trades(ev_res, draws=3, hold=HOLD, cost=COST)

    # 보유 중 상장폐지된 매매
    def _hit_delist(r):
        p = PX.get(r.stock_code)
        if p is None: return False
        s = p['Close']; pos = event_pos(s.index, r.event_date)
        if pos + 1 >= len(s): return False            # 공시 전에 이미 상장폐지 → 매매 불가
        return _truncated(s, pos + 1 + HOLD)
    ev_res['보유중상폐'] = ev_res.apply(_hit_delist, axis=1)

    print('\n\n########## 상장폐지 반영 결과 ##########')
    print(f"전체 이벤트 {len(ev_res)}건 (복원 {int(ev_res.복원.sum())}건)")
    print(f"보유 20일 안에 상장폐지된 매매 {int(ev_res.보유중상폐.sum())}건")
    if ev_res.보유중상폐.any():
        print(ev_res[ev_res.보유중상폐][['corp_name', 'market', 'type', 'event_date', 'trade_ew']].to_string())

    print('\n=== A. 복원 이벤트만의 매매 (동일가중 대비) ===')
    print(ev_res[ev_res.복원].groupby(['market', 'type']).trade_ew.apply(trade_stats).unstack().to_string())

    print('\n=== B. 8-3 다시: 그룹별 공시 vs 무작위 ===')
    r83 = []
    for (mk_, tp), sub in ev_res.groupby(['market', 'type']):
        e = sub.trade_ew.dropna(); p = plc_ew[(plc_ew.market == mk_) & (plc_ew.type == tp)].trade_ret.dropna()
        r83.append({'그룹': f'{mk_}-{tp}', 'N': len(e), '평균%': round(e.mean()*100, 2), '중앙값%': round(e.median()*100, 2),
                    '무작위%': round(p.mean()*100, 2), '차이%p': round((e.mean()-p.mean())*100, 2), '차이t': round(welch_t(e, p), 2)})
    print(pd.DataFrame(r83).set_index('그룹').to_string())

    sig = ev_res[(ev_res.type == '신탁') & (ev_res.market == 'KOSDAQ') & ev_res.status.isin(['ok', '기간밖'])]
    plc_ev = placebo_events(sig, draws=1, hold=HOLD, seed=0)
    print(f'\n=== C. 포트폴리오 (코스닥 신탁 {len(sig)}건, 슬롯 20) ===')
    rows = []
    for c in [0.003, 0.01]:
        p_ = portfolio_backtest(sig, delay=1, hold=HOLD, cost=c, slots=20)
        with use_benchmark(EW):
            h_ = portfolio_backtest(sig, delay=1, hold=HOLD, cost=c, slots=20)
            hp = portfolio_backtest(plc_ev, delay=1, hold=HOLD, cost=c, slots=20)
        a, b, pb = perf(p_.ret), perf(h_.hedged), perf(hp.hedged.reindex(h_.index))
        rows.append({'비용': f'{c:.1%}', '주식만 연%': a['연수익률%'], '주식만 샤프': a['샤프'], '주식만 MDD%': a['최대낙폭%'],
                     '헤지 연%': b['연수익률%'], '헤지 샤프': b['샤프'], '플라시보헤지 연%': pb['연수익률%']})
        if c == 0.003: h03, hp03 = h_, hp
    print(pd.DataFrame(rows).set_index('비용').to_string())

    print('\n=== D. 연도별 전략-플라시보 (동일가중 헤지, 비용 0.3%) ===')
    yy = pd.DataFrame({'전략%': h03.hedged.groupby(h03.index.year).apply(lambda r: (1 + r).prod() - 1),
                       '플라시보%': hp03.hedged.reindex(h03.index).groupby(h03.index.year).apply(lambda r: (1 + r).prod() - 1)}).mul(100).round(1)
    yy['차이%p'] = (yy['전략%'] - yy['플라시보%']).round(1)
    print(yy.to_string())
    print(f"플러스 연도 {int((yy['차이%p'] > 0).sum())}/{len(yy)}")
    print('########## 끝 ##########')
# ===== 끝 =====
