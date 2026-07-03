#!/usr/bin/env python3
"""Download arXiv PDFs for the ★ deep-review set, verify %PDF magic, emit manifest."""
import os
import sys
import time
import urllib.request

REPO = "/home/dongkyu/pkt_ws/temporal_vla"
B = f"{REPO}/docs/Activation_steering_basic"
R = f"{REPO}/docs/references"

# (key, arxiv_id, folder)
PAPERS = [
    # §1
    ("ParkLRH", "2311.03658", B),
    ("ToyModelsSuperposition", "2209.10652", B),
    ("RepE", "2310.01405", B),
    ("BolukbasiDebias", "1607.06520", B),
    # §2
    ("CunninghamSAE", "2309.08600", B),
    ("GeometryOfTruth", "2310.06824", B),
    ("ROME", "2202.05262", B),
    ("IOI", "2211.00593", B),
    ("TopKSAE", "2406.04093", B),
    # §3
    ("Conceptors", "2410.16314", B),
    ("ActAdd", "2308.10248", B),
    ("CAA", "2312.06681", B),
    ("ITI", "2306.03341", B),
    ("FunctionVectors", "2310.15213", B),
    ("ReFT", "2404.03592", B),
    ("TanSteeringReliability", "2407.12404", B),
    ("AxBench", "2501.17148", B),
    # §4
    ("ArditiRefusal", "2406.11717", B),
    ("Sycophancy", "2310.13548", B),
    ("PersonaVectors", "2507.21509", B),
    ("VTI_VLMHallucination", "2410.15778", B),
    # §5
    ("CircuitBreakers", "2406.04313", B),
    ("GemmaScope", "2408.05147", B),
    # §6 (references)
    ("SAE_VLA_pi05", "2603.19183", R),
    ("LAE_LatentActivationEditing", "2509.20623", R),
    ("VLS_SteerViaVLM", "2602.03973", R),
    # §7 (references)
    ("Sentinel_RuntimeMonitor", "2410.04640", R),
    ("FIPER_FailurePrediction", "2510.09459", R),
    ("KnowNo_AskForHelp", "2307.01928", R),
]

UA = {"User-Agent": "Mozilla/5.0 (research paper fetch)"}


def fetch(arxiv_id, dest):
    urls = [f"https://arxiv.org/pdf/{arxiv_id}.pdf", f"https://arxiv.org/pdf/{arxiv_id}"]
    for u in urls:
        try:
            req = urllib.request.Request(u, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if data[:5] == b"%PDF-":
                with open(dest, "wb") as f:
                    f.write(data)
                return True, len(data)
            else:
                last = f"not-pdf(head={data[:8]!r})"
        except Exception as e:  # noqa
            last = str(e)[:80]
    return False, last


def main():
    rows = []
    for key, aid, folder in PAPERS:
        fname = f"{key}_{aid}.pdf"
        dest = os.path.join(folder, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 10000:
            rows.append((key, aid, "SKIP(exists)", os.path.getsize(dest)))
            continue
        ok, info = fetch(aid, dest)
        rows.append((key, aid, "OK" if ok else "FAIL", info))
        print(f"{'OK ' if ok else 'FAIL'} {key:32s} {aid:12s} -> {info}")
        time.sleep(1.2)
    fails = [r for r in rows if r[2] == "FAIL"]
    print("\n=== SUMMARY ===")
    print(f"total={len(rows)} ok={sum(1 for r in rows if r[2]=='OK')} "
          f"skip={sum(1 for r in rows if r[2].startswith('SKIP'))} fail={len(fails)}")
    if fails:
        print("FAILED (verify id / find alt url):")
        for k, a, _, info in fails:
            print(f"  - {k} {a}: {info}")


if __name__ == "__main__":
    main()
