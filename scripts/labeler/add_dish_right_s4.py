"""Add source-verified dishwasher-right s4; target jitter is j0."""
import argparse,json
from pathlib import Path
from add_bread_rack_scenes import extend

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--grid-root',type=Path,required=True);a=ap.parse_args()
    h,stats=extend(a.index.read_text(),a.grid_root,[('4ab360df2b71','worker1','DishwasherRack/out-right',4,0,46)])
    note='DishwasherRack/out-right s4 추가: 전체 50판, 대상 j0 10판(실패 9·성공 1). '
    needle='<div class="help">추가 bread s4·rack-L s3 등록 완료.'
    assert needle in h
    h=h.replace(needle,'<div class="help">'+note+'추가 bread s4·rack-L s3 등록 완료.')
    a.out.write_text(h);print(json.dumps(stats))
