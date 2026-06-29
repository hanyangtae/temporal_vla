"""Phase A1 검증 리포트: FFN value-vector logit-lens 결과가 논문 주장을 재현하는지 확인.

실행:
  conda activate openvla-interp
  python ~/pkt_ws/mechanistic-steering-vlas/verify_a1.py

확인 포인트(논문 Section 3 / Fig 2):
  (1) value vector가 의미(semantic) 클러스터를 가진다  -> 예시 출력
  (2) action token이 후반 레이어에 집중된다 (Fig 2b)    -> 레이어별 action-token 비율 표
  (3) 'up'/'fast'/'slow' 등 제어개념 neuron이 실재한다  -> 개념별 개수+예시
  (4) 우리 투영 == 저자 공개 클러스터(up_10)             -> 일치 확인
"""
import os, re, yaml

ART = os.path.expanduser(
    "~/pkt_ws/mechanistic-steering-vlas/openvla/ffn_value_vectors/artifacts/ffn_value_projection")
TXT = os.path.join(ART, "top_tokens_output.txt")
SHIPPED = os.path.expanduser(
    "~/pkt_ws/mechanistic-steering-vlas/openvla/libero_experiments/configs/interventions/dictionaries.yaml")
INTER = 11008  # llama2-7b FFN intermediate size

line_re = re.compile(r"\[(\d+)\]\s*(.*)")

def parse(rest):
    out = []
    for it in rest.split(", "):
        it = it.strip()
        if it.startswith("[action"):
            out.append(("action", it))
        elif len(it) >= 2 and it[0] == "'" and it[-1] == "'":
            out.append(("tok", it[1:-1]))
        elif it:
            out.append(("tok", it))
    return out

rows = {}
with open(TXT) as f:
    for line in f:
        m = line_re.match(line)
        if m:
            rows[int(m.group(1))] = parse(m.group(2))
n = len(rows)
nlayers = (max(rows) // INTER) + 1
print(f"총 value vector: {n}  (= {nlayers} layers x {INTER})  [논문값 ~352,255]\n")

# (1) 의미 클러스터 예시
print("=" * 70)
print("(1) 의미(semantic) value vector 예시 — top token이 하나의 개념으로 묶임")
print("=" * 70)
showcase = [160000, 287222, 229524, 302565]  # pastness / update / higher-lower / north
for idx in showcase:
    if idx in rows:
        toks = [t[1] for t in rows[idx][:8]]
        print(f"  [{idx}] L{idx//INTER:>2}: {toks}")

# (2) Fig 2b: 레이어별 action-token 비율 (top-10 기준)
print("\n" + "=" * 70)
print("(2) Fig 2b 재현 — 레이어별 action-token 비율 (top-10 token 중)")
print("=" * 70)
per_layer = {}
for idx, toks in rows.items():
    L = idx // INTER
    top10 = toks[:10]
    a = sum(1 for k, _ in top10 if k == "action")
    d = per_layer.setdefault(L, [0, 0])
    d[0] += a; d[1] += len(top10)
print("  layer : action-token 비율")
for L in range(nlayers):
    a, tot = per_layer.get(L, [0, 1])
    frac = a / tot
    bar = "#" * int(frac * 50)
    print(f"   {L:>2}   : {frac*100:5.1f}%  {bar}")
print("  -> 후반 레이어로 갈수록 비율이 커지면 Fig 2b 재현 성공.")

# (3) 제어개념 neuron
print("\n" + "=" * 70)
print("(3) 제어개념 neuron (top token이 해당 단어로 시작)")
print("=" * 70)
def norm(s): return s.lstrip("▁ ").lower()
for concept, prefixes in [("up", ["up"]), ("fast", ["fast", "quick", "rapid"]),
                          ("slow", ["slow"]), ("left", ["left"]), ("right", ["right"])]:
    pre = tuple(prefixes)
    def kw_count(toks):  # top-10 중 개념 단어로 시작하는 token 수
        return sum(1 for k, s in toks[:10] if k == "tok" and norm(s).startswith(pre))
    hits = [idx for idx, toks in rows.items()
            if toks and toks[0][0] == "tok" and norm(toks[0][1]).startswith(pre)]
    best = max(hits, key=lambda i: kw_count(rows[i])) if hits else None
    extok = [t[1] for t in rows[best][:6]] if best is not None else []
    print(f"  {concept:5s}: {len(hits):3d}개   가장 선명한 neuron 예) {extok}")

# (4) 저자 공개 up_10 클러스터와 일치 확인
print("\n" + "=" * 70)
print("(4) 우리 투영 == 저자 공개 up_10 클러스터?  (같은 체크포인트 -> 같아야 정상)")
print("=" * 70)
ship = yaml.safe_load(open(SHIPPED))
ok = 0
for flat in list(ship["up_10"].keys()):
    toks = [t[1] for t in rows.get(int(flat), [])[:5]]
    up_like = any(norm(t).startswith("up") or t in ("raised","raise","above","higher","high","North","north")
                  for t in toks)
    ok += up_like
    print(f"  flat {flat} (L{int(flat)//INTER}): {toks}  {'OK' if up_like else '??'}")
print(f"  -> {ok}/{len(ship['up_10'])} 개가 up/높이 개념 토큰. 대부분 OK면 투영 신뢰.")
print("\n[완료] 위 4개가 모두 그럴듯하면 A1(해석분석) 재현 성공입니다.")
