"""Audited in-place completion of an EXISTING rollout_latents_v2 output dir (CPU, no model, no GPU).

Case (2026-11-02, srv50 commit 61a66541): live smoke passed (exact_49), 200/200 records written, but the converter's
final count also counted DIAG_V2.json (201 != 200) and refused COMPLETE.json. The records themselves were produced by
the converter whose sha is in contract.json (script_sha256). The fixed converter has a DIFFERENT sha, so resuming with it
would (correctly) refuse the provenance. This tool finishes the dir WITHOUT recomputing, moving, deleting or editing
anything that exists (contract.json, PARITY_SMOKE_V2.json, DIAG_V2.json, records, npz keep their bytes):
  1 contract.json is a rollout_latents_v2 contract for --token-contract; sha(PARITY_SMOKE_V2.json) ==
    contract.parity_smoke_v2_sha256; smoke provenance == the contract's provenance part; smoke rows re-judged
    (rollout_latents_v2.smoke_verdict, not the file's 'passed' flag)
  2 contract.episodes_contract_sha256 == sha(<episodes>/contract.json)
  3 --converter-source FILE (the exact converter bytes, e.g. `git show 61a66541:<path> > FILE`) must hash to
    contract.script_sha256 -> binds the records to the code that produced them (mandatory)
  4 latents_v2_audit.audit(...) ok: explicit episode identity set == --expect-records, no missing / mismatched /
    duplicate / unexpected record, npz/latents verified
  5 writes, exclusive-create only: COMPLETE_AUDIT_V2.json (full report) and COMPLETE.json
    {contract (as read), episodes, completed_by: {tool + sha, audit module sha, converter sha + commit, reason}, audit}
Without --write it only reports (exit 0 = would complete, 3 = refused).

    python finalize_latents_v2.py --episodes <full_episodes/srv50_bread_s4> --out <latents_v2_t49/srv50_bread_s4> \
        --token-contract 49 --expect-records 200 --converter-source <file> --converter-commit 61a66541 [--write]
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import latents_v2_audit as AU      # noqa: E402
import rollout_latents_v2 as R2    # noqa: E402

CONTRACT_EXTRA = ('parity_smoke_v2_sha256', 'episodes_contract_sha256', 'cross_device_validated_sha256')
REASON = ('converter counted DIAG_V2.json as a latent record (201 != 200) and wrote no COMPLETE.json; records audited in '
          'place by explicit episode identity, none recomputed, no existing file modified')


def check(episodes, out, token_contract, expect, converter_source):
    out, episodes = Path(out), Path(episodes)
    why = []
    cp, sp = out / 'contract.json', out / 'PARITY_SMOKE_V2.json'
    if not cp.exists() or not sp.exists():
        return None, None, [f'missing {[p.name for p in (cp, sp) if not p.exists()]}']
    contract, smoke = json.loads(cp.read_text()), json.loads(sp.read_text())
    if contract.get('version') != R2.VERSION or contract.get('token_contract') != token_contract:
        why.append(f'contract version/token_contract {contract.get("version")}/{contract.get("token_contract")}')
    if AU.sha(sp) != contract.get('parity_smoke_v2_sha256'):
        why.append('PARITY_SMOKE_V2.json bytes differ from the sha recorded in contract.json')
    prov = {k: v for k, v in contract.items() if k not in CONTRACT_EXTRA}
    if smoke.get('provenance') != prov:
        diff = sorted(k for k in set(prov) | set(smoke.get('provenance') or {}) if (smoke.get('provenance') or {}).get(k) != prov.get(k))
        why.append(f'smoke provenance != contract provenance in {diff}')
    try:
        R2.smoke_verdict(smoke.get('rows'), token_contract)
    except SystemExit as e:
        why.append(f'smoke rows do not pass: {e}')
    if AU.sha(episodes / 'contract.json') != contract.get('episodes_contract_sha256'):
        why.append('episodes contract sha differs from contract.json')
    src = Path(converter_source)
    if not src.is_file() or AU.sha(src) != contract.get('script_sha256'):
        why.append('--converter-source does not hash to contract.script_sha256 (records not bound to their code)')
    rep = AU.audit(episodes, out, token_contract, expect=expect)
    if not rep['ok']:
        why.append(f'audit: {json.dumps({k: v for k, v in rep["problems"].items() if v})[:1500]}')
    for name in ('COMPLETE.json', 'COMPLETE_AUDIT_V2.json'):
        if (out / name).exists():
            why.append(f'{name} already exists (never overwritten)')
    return contract, rep, why


def finalize(episodes, out, token_contract, expect, converter_source, converter_commit, write=False):
    contract, rep, why = check(episodes, out, token_contract, expect, converter_source)
    result = {'refused': why, 'audit': rep, 'would_complete': not why}
    if why or not write:
        return result
    out = Path(out)
    completed_by = {'tool': 'finalize_latents_v2', 'tool_sha256': AU.sha(Path(__file__)),
                    'audit_module_sha256': AU.sha(Path(AU.__file__)), 'rollout_latents_v2_current_sha256': AU.sha(Path(R2.__file__)),
                    'converter_script_sha256': contract['script_sha256'], 'converter_commit': converter_commit,
                    'converter_source_verified': True, 'reason': REASON}
    audit_doc = {'completed_by': completed_by, 'audit': rep}
    with open(out / 'COMPLETE_AUDIT_V2.json', 'x') as f:
        f.write(json.dumps(audit_doc, indent=1))
    with open(out / 'COMPLETE.json', 'x') as f:
        f.write(json.dumps({'contract': contract, 'episodes': rep['verified_records'], 'completed_by': completed_by,
                            'audit_report_sha256': AU.sha(out / 'COMPLETE_AUDIT_V2.json'),
                            'audit': {k: rep[k] for k in ('expected', 'verified_records', 'non_record_present')}}, indent=1))
    result['written'] = ['COMPLETE_AUDIT_V2.json', 'COMPLETE.json']
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--token-contract', type=int, choices=(16, 49), required=True)
    ap.add_argument('--expect-records', type=int, required=True)
    ap.add_argument('--converter-source', type=Path, required=True); ap.add_argument('--converter-commit', required=True)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args(argv)
    r = finalize(a.episodes, a.out, a.token_contract, a.expect_records, a.converter_source, a.converter_commit, a.write)
    print(json.dumps({'refused': r['refused'], 'would_complete': r['would_complete'], 'written': r.get('written'),
                      'verified_records': (r['audit'] or {}).get('verified_records')}, indent=1))
    return 0 if r['would_complete'] else 3


if __name__ == '__main__':
    sys.exit(main())
