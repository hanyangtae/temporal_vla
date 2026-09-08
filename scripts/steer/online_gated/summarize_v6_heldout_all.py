#!/usr/bin/env python3
"""고정 manifest와 새 base에 쌍대응해 구제·파괴·전체 SR를 집계한다 (stdlib)."""
import argparse
import collections
import csv
from pathlib import Path

ARMS = ['base', 'plain_b08', 'jfair_b09', 'reseed', 'reseed_plain_b08', 'reseed_jfair_b09']

def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))

def coordinates(r, manifest=False):
    slug = r['grid_instruction'].replace('/','_') if manifest else r['slug']
    return (slug, int(r['scene_idx']), int(r['jitter_idx']), int(r['noise_idx']))

def metrics(rows, expected):
    paired = [r for r in rows if r['baseline_success'] != '']
    fail = [r for r in paired if int(r['baseline_success']) == 0]
    succ = [r for r in paired if int(r['baseline_success']) == 1]
    rescue = sum(int(r['success']) for r in fail)
    harm = sum(1-int(r['success']) for r in succ)
    return dict(completed=len(rows), expected=expected, paired=len(paired),
                fail_n=len(fail), rescued=rescue, rescue_rate=rescue/len(fail) if fail else '',
                succ_n=len(succ), destroyed=harm, destruction_rate=harm/len(succ) if succ else '',
                success=sum(int(r['success']) for r in rows),
                sr=sum(int(r['success']) for r in rows)/len(rows) if rows else '',
                paired_delta_sr=(rescue-harm)/len(paired) if paired else '')

def summarize(roots, out):
    manifest, results = {}, {}
    for root in roots:
        local = {coordinates(r,True):r for r in read(root/'manifest.tsv')}
        for c,r in local.items():
            k=(r['machine'],)+c
            if k in manifest: raise ValueError(f'duplicate manifest episode: {k}')
            manifest[k]=r
        if not (root/'episodes.tsv').exists(): continue
        for r in read(root/'episodes.tsv'):
            c=coordinates(r); e=local[c]; k=(e['machine'],)+c+(r['eval_arm'],)
            if k in results: raise ValueError(f'duplicate result: {k}')
            if r['eval_arm'] not in ARMS: raise ValueError(r['eval_arm'])
            if int(r['collection_success']) != int(e['success']): raise ValueError(f'label mismatch {k}')
            results[k]=dict(r, machine=e['machine'], artifact_slug=e['artifact_slug'])
    baselines={k[:-1]:r for k,r in results.items() if k[-1]=='base'}
    mismatch=sum(int(r['success']) != int(manifest[k]['success']) for k,r in baselines.items())
    # Always derive pairs from the actual base rows, never trust a stale per-machine summary.
    for k,r in results.items():
        b=baselines.get(k[:-1]); r['baseline_success']=b['success'] if b else ''
    rows=[]
    for arm in ARMS:
        armrows=[r for k,r in results.items() if k[-1]==arm]
        rows.append(dict(scope='overall',slug='',scene='',arm=arm,**metrics(armrows,len(manifest))))
        tasks=sorted({r['artifact_slug'] for r in manifest.values()})
        for slug in tasks:
            er=[r for r in manifest.values() if r['artifact_slug']==slug]
            rr=[r for r in armrows if r['artifact_slug']==slug]
            rows.append(dict(scope='instruction',slug=slug,scene='',arm=arm,**metrics(rr,len(er))))
            for scene in sorted({int(r['scene_idx']) for r in er}):
                es=[r for r in er if int(r['scene_idx'])==scene]
                rs=[r for r in rr if int(r['scene_idx'])==scene]
                rows.append(dict(scope='scene',slug=slug,scene=scene,arm=arm,**metrics(rs,len(es))))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'summary.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t'); w.writeheader();w.writerows(rows)
    lines=['# v6 heldout detector 성공·실패 전판 평가','',
           '범위: detector는 대상 j를 제외. 기존 연산자는 대상 실패판 fit 포함, β는 기존 실패 평가에서 선정. 전체 파이프 독립 holdout 결과가 아니다.','',
           f'- 기준 판 {len(manifest)}개; base 완료 {len(baselines)}개; 수집/base 불일치 {mismatch}개.',
           '- 미완료 판은 분모에 넣지 않는다. arm별 완료 수와 paired 분모를 함께 확인한다.','',
           '| arm | 완료/예정 | 쌍 | 구제/실패 | 파괴/성공 | 전체 SR | paired ΔSR |',
           '|---|---|---|---|---|---|---|']
    for r in rows:
        if r['scope']!='overall':continue
        fmt=lambda x: f'{x:.4f}' if x!='' else 'NA'
        lines.append(f"| {r['arm']} | {r['completed']}/{r['expected']} | {r['paired']} | {r['rescued']}/{r['fail_n']} | {r['destroyed']}/{r['succ_n']} | {fmt(r['sr'])} | {fmt(r['paired_delta_sr'])} |")
    lines += ['', 'instruction·scene별 분해는 summary.tsv 참조. 파괴율은 새 base 성공판 중 arm 실패 비율이다. 결과 해석 전 runbook의 confound audit를 갱신한다.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();summarize(a.results_root,a.out)

if __name__=='__main__':main()
