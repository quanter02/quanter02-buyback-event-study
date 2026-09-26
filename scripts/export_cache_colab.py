# =====================================================================
# Colab에서 실행: 드라이브 캐시를 로컬 실행기(run_local.py)용 조각 파일로 내보낸다.
#   - 가격(prices_*.pkl) + 코스피·코스닥 지수 + dart_list 공시 CSV를 한 묶음으로
#   - pandas 객체 대신 numpy 배열로 저장 (Colab과 로컬의 pandas 버전이 달라도 읽힘)
#   - gzip 압축 후 PART_MB 단위로 쪼갠다 (파일 이름에 전체 조각 수 표시: part00of04) → MyDrive/disclosure_cache/upload/
# 사용: 새 셀에  %run -i export_cache_colab.py   (또는 셀에 붙여넣기)
# =====================================================================
import gzip, os, pickle, glob
import numpy as np, pandas as pd

WORK = '/content/drive/MyDrive/disclosure_cache'
TAG = '2016-01-01_2026-08-31'
PART_MB = 8                                   # 조각 크기 (업로드 한도에 맞게 조절)

try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception:
    pass

d = pickle.load(open(f'{WORK}/cache/prices_{TAG}.pkl', 'rb'))
px = {c: {'d': p.index.values.astype('datetime64[ns]'), 'c': p['Close'].to_numpy('float64'),
          'v': p['Volume'].to_numpy('float64')} for c, p in d['px'].items()}

import FinanceDataReader as fdr               # 노트북 load_index 와 같은 출처
idx = {}
for m, code in [('KOSPI', 'KS11'), ('KOSDAQ', 'KQ11')]:
    s = fdr.DataReader(code, '2015-01-01', '2027-04-30')['Close'].astype(float)
    idx[m] = {'d': s.index.values.astype('datetime64[ns]'), 'c': s.to_numpy('float64')}

dart = {os.path.basename(f): open(f, 'rb').read() for f in sorted(glob.glob(f'{WORK}/cache/dart_list/list_*.csv'))}

bundle = {'px': px, 'missing': sorted(d.get('missing', [])), 'idx': idx, 'dart_list': dart}
gz = gzip.compress(pickle.dumps(bundle, protocol=4), 9)
step = PART_MB * 1_000_000
n = (len(gz) + step - 1) // step

out = f'{WORK}/upload'
os.makedirs(out, exist_ok=True)
for f in glob.glob(f'{out}/bundle_*'): os.remove(f)
for i in range(n):
    open(f'{out}/bundle_{TAG}.pkl.gz.part{i:02d}of{n:02d}', 'wb').write(gz[i * step:(i + 1) * step])

print(f'종목 {len(px)}개, 지수 {[(m, str(v["d"][0])[:10], str(v["d"][-1])[:10]) for m, v in idx.items()]}, 공시 파일 {len(dart)}개')
print(f'압축 {len(gz)/1e6:.1f}MB → {n}조각 ({PART_MB}MB 단위): {out}')
