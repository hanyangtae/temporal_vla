---
name: remote-shard-stream-crc
description: 원격 HDD로 수십 GB shard npz를 스트리밍 복사하면 드물게 조용한 손상이 난다 — 쓰고 나서 멤버 CRC 대조를 반드시 넣을 것
metadata:
  type: feedback
---

원격 노드에서 shard npz 를 행 단위로 스트리밍 복사·재작성한 뒤에는 **멤버 CRC 대조를
직접 돌려 확인**한다. `zipfile.ZipFile.open(name)` 으로 끝까지 읽고 `zlib.crc32` 를
`ZipInfo.CRC` 와 비교하면 된다(수 GB 라도 몇 분).

**Why:** 2026-09-01 v4r step1 에서 7 shard 중 `PPCC_bread.npz` 하나만 `X.npy` CRC 불일치
(`BadZipFile: Bad CRC-32`)가 나 다음 단계(ae_cluster)가 죽었다. 파일 크기·헤더는 정상이었고
같은 코드로 만든 나머지 6개는 멀쩡했다 → 로직 버그가 아니라 **HDD 쓰기/읽기 단계의 일회성
손상**. 그 파일만 지우고 같은 명령으로 재생성하니 CRC 일치했다(재현 안 됨).

**How to apply:** 대형 shard 를 만든 직후 CRC 스캔 → 불일치 파일만 지우고 재생성(빌더를
idempotent 하게: 대상이 있으면 skip-write). 실패를 다음 스텝에서 만나면 원인 추적에
시간이 크게 든다. [[remote-node-hardware]] [[v4r-recollect-round]]
