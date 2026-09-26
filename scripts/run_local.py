# =====================================================================
# Colab 없이 로컬(오프라인)에서 노트북 파이프라인을 다시 돌리는 실행기
#
#   python scripts/run_local.py --period quick     # 2024-07 ~ 2026-08 (노트북 1~8단계와 비교)
#   python scripts/run_local.py --period full      # 2016-01 ~ 2026-08 (+ 상장폐지 복원)
#
# 필요한 파일 (data/cache/, git에는 올리지 않음):
#   raw_{START}_{END}.csv        공시 원자료 (full은 dart_list/list_*.csv 조각도 가능)
#   prices_{START}_{END}.pkl     {'px': {종목코드: DataFrame(Close, Volume)}, 'missing': set}
#   index_{START}_{END}.pkl      {'KOSPI': Series, 'KOSDAQ': Series}  (종가)
#   (quick 캐시가 없으면 전체 기간 prices/index 캐시를 대신 쓴다)
# 네트워크를 쓰지 않는다. 캐시에 없는 종목은 '가격없음'으로 처리된다.
# =====================================================================
import argparse, glob, json, os, pickle, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")

ap = argparse.ArgumentParser()
ap.add_argument("--period", choices=["quick", "full"], default="quick")
ap.add_argument("--no-delist", action="store_true", help="full에서 상장폐지 복원(delist_fix3) 생략")
args = ap.parse_args()

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")

G = globals()
G["display"] = lambda x: print(x.to_string() if hasattr(x, "to_string") else x)

# ---- 노트북의 설정 셀과 핵심 함수 셀을 그대로 가져온다 ----
nb = json.load(open(os.path.join(ROOT, "notebooks", "buyback_event_study.ipynb")))
src = lambda i: "".join(nb["cells"][i]["source"])
assert "핵심 함수 모음" in src(5), "노트북 셀 순서가 바뀌었습니다 (5번 셀 = 핵심 함수)"
exec(src(5), G)

PRE, POST, HOLD, COST, DEDUP_DAYS, SPLIT_DATE = 5, 20, 20, 0.003, 30, "2021-01-01"
START, END = ("2024-07-01", "2026-08-31") if args.period == "quick" else ("2016-01-01", "2026-08-31")
TAG = f"{START}_{END}"

FULL_TAG = "2016-01-01_2026-08-31"          # 전체 기간 캐시가 있으면 2년(quick)도 그걸로 돌린다
def _load_bundle():
    """scripts/export_cache_colab.py 가 만든 조각(bundle_*.pkl.gz.partNN)을 이어 붙여 읽는다.
    pandas 버전 차이를 피하려고 numpy 배열로만 저장돼 있다."""
    import gzip
    parts = sorted(glob.glob(os.path.join(CACHE, f"bundle_{FULL_TAG}.pkl.gz.part*")))
    if not parts:
        return None
    need = int(parts[0].rsplit("of", 1)[1])
    assert len(parts) == need, f"조각 {len(parts)}개 / 필요 {need}개 — 빠진 조각이 있습니다"
    b = pickle.loads(gzip.decompress(b"".join(open(p, "rb").read() for p in parts)))
    os.makedirs(os.path.join(CACHE, "dart_list"), exist_ok=True)
    for name, data in b.get("dart_list", {}).items():          # 공시 CSV 조각도 함께 풀어 둔다
        open(os.path.join(CACHE, "dart_list", name), "wb").write(data)
    px = {c: pd.DataFrame({"Close": v["c"], "Volume": v["v"]}, index=pd.DatetimeIndex(v["d"]))
          for c, v in b["px"].items()}
    idx = {m: pd.Series(v["c"], index=pd.DatetimeIndex(v["d"])) for m, v in b["idx"].items()}
    print(f"조각 {len(parts)}개에서 가격 {len(px)}종목·지수·공시 {len(b.get('dart_list', {}))}개 파일 복원")
    return px, set(b["missing"]), idx

bundle = _load_bundle()

# ---- 공시 ----
raw_path = os.path.join(CACHE, f"raw_{TAG}.csv")
if os.path.exists(raw_path):
    raw = pd.read_csv(raw_path, dtype=str)
else:
    parts = sorted(glob.glob(os.path.join(CACHE, "dart_list", "list_*.csv")))
    if not parts:
        print(f"⚠️ 공시 파일이 없습니다: {os.path.relpath(raw_path, ROOT)} 또는 data/cache/dart_list/ "
              f"(또는 bundle 조각 — scripts/export_cache_colab.py)")
        sys.exit(0)
    raw = pd.concat([pd.read_csv(f, dtype=str) for f in parts], ignore_index=True)
events = clean_events(raw, DEDUP_DAYS)
events["period"] = np.where(events.event_date < SPLIT_DATE, f"~{SPLIT_DATE[:4]} 이전", f"{SPLIT_DATE[:4]}~")
cancel_path = os.path.join(CACHE, f"cancel_{TAG}.csv")
if os.path.exists(cancel_path):
    cm = pd.read_csv(cancel_path, dtype={"rcept_no": str})
    events["cancel"] = events.rcept_no.map(dict(zip(cm.rcept_no, cm.cancel)))
    events["소각"] = events["cancel"].map({True: "소각 언급", False: "소각 없음"}).fillna("확인불가")
print(f"원자료 {len(raw)}건 → 정제 후 이벤트 {len(events)}건 ({events.stock_code.nunique()}종목)")
print(events.groupby(["market", "type"]).size().to_string())

# ---- 가격 / 지수 (캐시 전용) ----
def _cache(kind):
    for t in (TAG, FULL_TAG):
        p = os.path.join(CACHE, f"{kind}_{t}.pkl")
        if os.path.exists(p):
            return p
    return os.path.join(CACHE, f"{kind}_{TAG}.pkl")
if bundle:
    PX_STORE, MISSING, IDX_STORE = bundle
else:
    px_path, idx_path = _cache("prices"), _cache("index")
    missing_files = [p for p in (px_path, idx_path) if not os.path.exists(p)]
    if missing_files:
        print("\n⚠️ 가격 캐시가 없어 여기서 멈춥니다. 필요한 파일:")
        print(f"   data/cache/bundle_{FULL_TAG}.pkl.gz.part00ofNN ...  (scripts/export_cache_colab.py 로 생성)")
        for p in missing_files: print("   또는", os.path.relpath(p, ROOT))
        sys.exit(0)
    d = pickle.load(open(px_path, "rb"))
    PX_STORE, MISSING = d["px"], set(d.get("missing", set()))
    IDX_STORE = pickle.load(open(idx_path, "rb"))
print(f"가격 캐시 {len(PX_STORE)}종목, 지수 {list(IDX_STORE)}")

def load_px(code, start, end):
    return PX_STORE.get(code)
_load_px_orig = load_px
def load_index(market, start, end):
    return IDX_STORE[market]
def _save():
    pass

# ---- 3단계 ----
t0 = time.time()
ev_res, ar_mat = build_ar_matrix(events, PRE, POST, HOLD, COST, verbose=False)
print(f"\n【3단계】 처리 결과 ({time.time()-t0:.0f}s)"); print(ev_res.status.value_counts().to_string())
print(f"가격없음 비율 {(ev_res.status == '가격없음').mean():.1%}")

print("\n【5~6단계】 시장 × 유형 (지수 대비, 비용 반영)")
display(by_group(ev_res, ev_res.trade_ret))

# ---- 7·8단계 ----
def run_addon(name):
    code = open(os.path.join(ROOT, "addons", name), encoding="utf-8").read()
    print(f"\n{'#'*20} {name} {'#'*20}")
    exec(compile(code, name, "exec"), G)

run_addon("step7_addon.py")
run_addon("step8_addon.py")
if args.period == "full" and not args.no_delist:
    run_addon("delist_fix3.py")

# ---- 추가 검증 (README 6·8장) ----
exec(open(os.path.join(ROOT, "scripts", "extra_checks.py"), encoding="utf-8").read(), G)
