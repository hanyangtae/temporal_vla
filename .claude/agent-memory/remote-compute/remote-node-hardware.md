---
name: remote-node-hardware
description: 승준 원격 노드(166.104.146.37:11112) 실측 사양 — 8 core / 31GB RAM, 데이터는 HDD라 대형 npz 로드가 I/O 병목
metadata:
  type: reference
---

원격 compute 노드 실측 (2026-08-27): `nproc` = **8**, RAM **31GB** (가용 ~26GB), swap 0.
데이터(`~/datasets/temporal_vla_store/...`)는 HDD.

**How to apply:**
- OMP/OPENBLAS/MKL_NUM_THREADS는 8을 넘겨봐야 의미 없다 (헬퍼 기본 8이 적정). 상위 세션이
  16을 지시해도 8로 낮추고 보고할 것.
- RAM 31GB 제약: grid_phase segA shard npz는 개당 3~11GB(v2 segA 총 ~70GB)이고 `load_shard`가
  shard 하나를 통째로 메모리에 올린 뒤 [n,1536] f32로 슬라이스한다 → peak ≈ 최대 shard 크기
  (v2 CoffeeSetupMug 10.7GB)로 묶여 있어 통과하지만 여유는 크지 않다. 동시 대형 job 2개 금지.
- HDD I/O가 실제 병목: 70GB shard 로드만 10~20분, pkl(개당 ~600MB) 스캔은 케이스 수에 비례해
  급증 → 매니페스트 dir를 glob하는 스크립트는 필요한 케이스만 담은 subset dir를 만들어 넘길 것.
- 로컬 kanu(64코어) 기준 CPU cap 메모리와 혼동하지 말 것.
- 볼륨 분리 (2026-08-27 `df -h` 실측): `~/datasets` = /dev/sda2 1.8T HDD (여유 ~369G) ·
  `~/workspace`(repo) = NVMe root 468G 인데 **여유 68G뿐**. 수십 GB 규모 shard 사본/재작성
  산출물은 반드시 `~/datasets` 쪽에 두고, `~/workspace/outputs`에는 소용량 결과만 쓸 것.
