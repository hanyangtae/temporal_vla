"""Upgrade the existing labeler HTML without touching cells or stored labels."""
import argparse
from pathlib import Path


def upgrade(html):
    if '["successful_retry"' in html:
        raise ValueError('successful_retry already installed; inspect before changing')
    changes = [
        ('v6 실패 시점 라벨러', 'v6 사건 시점 라벨러'),
        ('  ["other","기타","메모에 설명"],', '  ["other","기타","메모에 설명"],\n  ["successful_retry","성공적 재시도","재시도가 성공한 시점 · 최종 task 성공과 별개"],'),
        ('<kbd>1</kbd>~<kbd>7</kbd>', '<kbd>1</kbd>~<kbd>8</kbd>'),
        ('/^[1-7]$/', '/^[1-8]$/'),
        ('마크는 여러 개 가능(각각 유형·메모). 첫 마크 = 물리적 실패가 <b>처음 확정되는 프레임</b>(예: 그리퍼가 대상을 놓치는 순간, 물체가 떨어지는 순간). 애매하면 메모에 남기고 저장.',
         '마크는 실패와 성공적 재시도 등 <b>관측한 사건</b>을 기록합니다. 실패 사건과 재시도가 성공한 시점을 각각 표시하세요. 8번은 최종 task 성공 여부를 바꾸지 않습니다. 애매하면 메모에 남기고 저장.'),
        ('실패가 확정되는 프레임마다 <kbd>M</kbd>.', '기록할 사건 시점에서 <kbd>M</kbd>, 성공적 재시도는 <kbd>8</kbd>.'),
        ('물리적 실패 없음 · 표류/timeout (X)', '물리적 실패 없음 · 마크 전체 지움 (X)'),
        ('const rec={rel:c.rel', 'const firstFailure=ms.find(m=>m.type!=="successful_retry")||null;\n  const rec={rel:c.rel'),
        ('t_fail:ms.length?ms[0].t:null,frame:ms.length?ms[0].frame:null', 't_fail:firstFailure?firstFailure.t:null,frame:firstFailure?firstFailure.frame:null'),
        ('type:ms.length?ms[0].type:null', 'type:firstFailure?firstFailure.type:null'),
    ]
    for old, new in changes:
        if old not in html:
            raise ValueError('Unexpected HTML: missing '+old[:80])
        html=html.replace(old,new)
    return html


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.write_text(upgrade(a.index.read_text(encoding='utf-8')),encoding='utf-8')
