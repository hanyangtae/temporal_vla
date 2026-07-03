#!/usr/bin/env python3
"""Download §5 rework new-source PDFs to Activation_steering_basic, verify %PDF."""
import os, sys, time, urllib.request
B = "/home/dongkyu/pkt_ws/temporal_vla/docs/Activation_steering_basic"
PAPERS = [
    ("CAST", "2409.05907"),
    ("RogueScalpel", "2509.22067"),
    ("GoogleSteerableChatbots", "2505.04260"),
    ("ReliableEvalSteering", "2410.17245"),
    ("SEGA", "2301.12247"),
    ("OpenCharacterTraining", "2511.01689"),
    ("Asyrp", "2210.10960"),
    ("SafeLatentDiffusion", "2211.05105"),
    ("MinimizingCollateralDamage", "2605.01167"),
    ("ASA_ToolCallingRepE", "2602.04935"),
]
UA = {"User-Agent": "Mozilla/5.0 (research paper fetch)"}
def fetch(aid, dest):
    for u in (f"https://arxiv.org/pdf/{aid}.pdf", f"https://arxiv.org/pdf/{aid}"):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60) as r:
                data = r.read()
            if data[:5] == b"%PDF-":
                open(dest, "wb").write(data); return True, len(data)
            last = f"not-pdf({data[:8]!r})"
        except Exception as e:
            last = str(e)[:90]
    return False, last
fails = []
for key, aid in PAPERS:
    dest = f"{B}/{key}_{aid}.pdf"
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        print(f"SKIP {key}"); continue
    ok, info = fetch(aid, dest)
    print(f"{'OK ' if ok else 'FAIL'} {key:30s} {aid:12s} {info}")
    if not ok: fails.append((key, aid, info))
    time.sleep(1.2)
print("\nFAILS:", fails if fails else "none")
