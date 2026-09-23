"""Add point/interval editing to a live index, preserving embedded cells/labels."""
import argparse
from pathlib import Path

def extend(html):
    assert 'id="bRangeStart"' not in html,'interval UI already present'
    def put(old,new):
        nonlocal html
        assert old in html,old
        html=html.replace(old,new)
    put('</style>', '''.interval-band{position:absolute;top:0;bottom:0;background:var(--mark-soft);opacity:.6;pointer-events:none;border:1px solid var(--mark)}
.interval-band.selected{opacity:.85}.event-label{flex:1;overflow-wrap:anywhere}.boundary-fields{display:flex;gap:6px}.boundary-fields input{width:90px}
</style>''')
    put('<h2>선택 마크 메모</h2>', '''<h2>구간 라벨</h2>
    <div class="rowb"><button id="bRangeStart">구간 시작 (I)</button><button id="bRangeEnd">현재 시점으로 종료 (O)</button><button id="bRangeStartMove">시작/점 이동</button></div>
    <div class="boundary-fields"><label>시작/점 프레임 <input id="rangeStartFrame" type="number" min="0" step="1"></label><label>종료 프레임 <input id="rangeEndFrame" type="number" min="0" step="1"></label></div>
    <div class="help">지속 상태: I로 시작 → 영상을 이동 → O로 종료. 선택한 점 마크에 O를 누르면 구간으로 전환됩니다. 구간 클릭 후 경계를 수정할 수 있습니다. 여러 구간·재발·겹치는 사건도 기록 가능합니다. 종료는 상태가 끝난 첫 프레임입니다. 끝까지 관측되지 않은 종료를 임의로 추정하지 마세요.</div>
    <h2>선택 마크 메모</h2>''')
    put('CSV 내보내기</button>', '판 단위 CSV</button><button id="btnEventsCSV">사건별 CSV</button><button id="btnEventsTSV">사건별 TSV</button>')
    put('// ---- init',Path(__file__).with_name('interval_labels.js').read_text()+'\n// ---- init')
    return html

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.write_text(extend(a.index.read_text()))
