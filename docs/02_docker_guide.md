# Docker 사용 가이드 (temporal_vla)

이 프로젝트의 Docker 환경을 이해하고 사용하기 위한 가이드입니다. 컨테이너 구조와 실행/운영 절차에 초점을 둡니다.

> 통일 HTTP API 계약 (endpoint, sub-key, 모델 × 벤치마크 호환 매트릭스, 운영 패턴) 은 [`01_serving_interface.md`](01_serving_interface.md) 가 단일 출처입니다. 이 문서는 그 위에서 컨테이너를 어떻게 띄우고 관리하는지를 다룹니다.

---

## 1. Docker 기본 개념

이 프로젝트에서 사용하는 Docker 개념을 간단히 설명합니다.

| 개념 | 설명 | 이 프로젝트에서의 예시 |
|------|------|----------------------|
| **이미지(Image)** | 소프트웨어가 설치된 스냅샷. Dockerfile로 정의 | `docker/robocasa/Dockerfile` → robocasa 이미지 |
| **컨테이너(Container)** | 이미지를 실행한 것. 독립된 환경에서 동작 | `robocasa`, `groot`, `groot_n15`, `xvla`, `dreamvla` |
| **볼륨 마운트(Volume)** | 호스트 폴더를 컨테이너 안에 연결 | 프로젝트 폴더 `./` → 컨테이너 내 `/temporal_vla` |
| **docker-compose** | 여러 컨테이너를 하나의 설정 파일로 관리 | `docker-compose.yml` |
| **서비스(Service)** | compose 파일에 정의된 각 컨테이너 단위 | `robocasa`, `groot`, `groot_n15`, `xvla`, `dreamvla` |

**왜 컨테이너를 분리하는가?**

- `robocasa`는 Python 3.11 + MuJoCo + GUI(KasmVNC)가 필요
- `xvla`는 Python 3.10 + LeRobot이 필요
- `dreamvla`는 Python 3.10 + flamingo_pytorch + CLIP이 필요
- 이들의 의존성이 서로 충돌하므로 각각 별도의 컨테이너를 사용합니다. GR00T도 N1.6은 `groot`, N1.5는 `groot_n15`로 분리합니다.

**볼륨 마운트의 의미:**

호스트(내 PC)에서 코드를 수정하면 컨테이너 안에도 즉시 반영됩니다.
별도로 파일을 복사할 필요가 없습니다. 코드 편집은 호스트에서, 실행은 컨테이너에서 합니다.

---

## 2. 컨테이너 구성

### 2.1 robocasa

| 항목 | 내용 |
|------|------|
| 역할 | RoboCasa 시뮬레이션 실행, 데이터셋 평가, VLA 모델 평가 (HTTP로 모델 서버 호출) |
| Python | 3.11 |
| 주요 패키지 | MuJoCo, robosuite, robocasa, lerobot, torch 2.5.1 |
| GUI | KasmVNC (원격) 또는 X11 (로컬) |
| 상태 | 항상 실행 (기본 컨테이너) |

이 컨테이너에서 시뮬레이션을 실행하고, 모델이 예측한 행동(action)을 받아 환경에 적용합니다.
모델 추론은 xvla/dreamvla 컨테이너의 서버에 HTTP 요청을 보내서 수행합니다.

### 2.2 groot

| 항목 | 내용 |
|------|------|
| 역할 | GR00T N1.6 학습/추론, ZMQ model server |
| Python | 3.10 |
| 주요 패키지 | Isaac-GR00T N1.6, torch 2.7.1, flash-attn |
| 서비스명 | `groot` |

N1.6 RoboCasa fine-tuning과 ZMQ 평가 server에 사용합니다.

### 2.3 groot_n15

| 항목 | 내용 |
|------|------|
| 역할 | GR00T N1.5 공식 RoboCasa recipe 및 PandaOmron sanity fine-tuning |
| Python | 3.10 |
| 주요 패키지 | Isaac-GR00T-N1.5, torch 2.5.1, flash-attn |
| 서비스명 | `groot_n15` |

기존 `groot`는 N1.6 전용이므로 N1.5는 `groot_n15`로 분리합니다.

### 2.4 xvla

| 항목 | 내용 |
|------|------|
| 역할 | X-VLA 모델 학습 (`lerobot-train`) 및 추론 서버 (FastAPI) |
| Python | 3.10 |
| 주요 패키지 | `lerobot[xvla]` |
| 서버 포트 | 8100 |
| 활성화 | `--profile xvla` 필요 |

LeRobot v3.0 형식의 데이터셋을 사용합니다 (`data/datasets_v3/`).

### 2.5 dreamvla

| 항목 | 내용 |
|------|------|
| 역할 | DreamVLA 모델 학습 및 추론 서버 (FastAPI) |
| Python | 3.10 |
| 주요 패키지 | flamingo_pytorch, CLIP, transformers 4.40.2 |
| 서버 포트 | 8200 |
| 활성화 | `--profile dreamvla` 필요 |

DreamVLA 코드를 별도로 clone해야 합니다:

```bash
git clone https://github.com/Zhangwenyao1/DreamVLA dreamvla
```

---

## 3. 컨테이너 시작 / 중지 / 접속

### 3.1 이미지 빌드

컨테이너를 처음 사용하거나 Dockerfile이 변경되었을 때 빌드합니다.

```bash
# robocasa (필수)
docker compose build robocasa

# GR00T N1.6
docker compose build groot

# GR00T N1.5
docker compose build groot_n15

# xvla (X-VLA 사용 시)
docker compose build xvla

# dreamvla (DreamVLA 사용 시)
docker compose build dreamvla

# 캐시 무시하고 처음부터 재빌드
docker compose build --no-cache robocasa
```

> 첫 빌드는 PyTorch, Flash Attention 등 빌드로 **30분 이상** 걸릴 수 있습니다.
> 이후에는 캐시 덕분에 빠릅니다.

### 3.2 컨테이너 시작

```bash
# robocasa 시작 (백그라운드)
docker compose up -d robocasa

# xvla 시작
docker compose up -d xvla

# GR00T N1.6 시작
docker compose up -d groot

# GR00T N1.5 시작
docker compose up -d groot_n15

# dreamvla 시작
docker compose up -d dreamvla

# 모든 컨테이너 한번에 시작
docker compose up -d
```

`-d` 옵션은 백그라운드 실행을 의미합니다. 터미널을 닫아도 컨테이너가 유지됩니다.

### 3.3 컨테이너 접속 (셸 진입)

```bash
# 실행 중인 robocasa에 접속
docker compose exec robocasa bash

# xvla에 접속
docker compose exec xvla bash

# GR00T N1.6에 접속
docker compose exec groot bash

# GR00T N1.5에 접속
docker compose exec groot_n15 bash
```

`exec`은 **이미 실행 중인** 컨테이너에 새 셸을 연결합니다.
컨테이너가 `up` 상태가 아니면 에러가 발생합니다.

### 3.4 일회성 명령 실행 (run --rm)

```bash
# xvla 컨테이너에서 학습 스크립트 실행 후 자동 삭제
docker compose run --rm xvla \
  bash /temporal_vla/scripts/train/xvla.sh
```

`run --rm`은 새 컨테이너를 만들어 명령을 실행하고, 끝나면 컨테이너를 자동 삭제합니다.
학습, 데이터 변환 등 **일회성 작업**에 적합합니다.

> **`exec` vs `run --rm` 정리:**
> - `exec`: 이미 실행 중인 컨테이너에 접속. 상태가 유지됨.
> - `run --rm`: 새로 컨테이너를 만들어 실행 후 삭제. 일회성 작업용.

### 3.5 컨테이너 중지 / 삭제

```bash
# robocasa만 중지
docker compose stop robocasa

# 모든 컨테이너 중지 및 삭제
docker compose down

# 이미지까지 삭제 (재빌드 필요)
docker compose down --rmi all
```

### 3.6 상태 확인

```bash
docker compose ps                    # 실행 중인 컨테이너 목록
docker compose logs robocasa         # 로그 확인
docker compose logs -f robocasa      # 실시간 로그 스트리밍 (Ctrl+C로 종료)
```

---

## 4. 디스플레이 설정 (VNC vs X11)

robocasa 컨테이너에서 GUI(시뮬레이션 화면)를 보려면 디스플레이 설정이 필요합니다.
사용 환경에 따라 두 가지 방법 중 하나를 선택합니다.

### 비교 표

| | SSH 원격 접속 (VNC) | 로컬 우분투 PC (X11) |
|---|---|---|
| **사용 상황** | SSH로 서버에 접속해서 작업 | 모니터가 연결된 PC에서 직접 작업 |
| **GUI 접근** | 웹 브라우저로 VNC 접속 | 호스트 화면에 직접 출력 |
| **설정 파일** | `docker-compose.override.example.yml` | `docker-compose.override.local.example.yml` |
| **필요한 .env 변수** | `VNC_PW` | `DISPLAY` |

### 4.1 VNC 모드 (SSH 원격 접속)

1. override 파일 복사:

   ```bash
   cp docker-compose.override.example.yml docker-compose.override.yml
   ```

2. `.env` 파일에서 `VNC_PW` 설정:

   ```
   VNC_PW=원하는비밀번호
   ```

3. 컨테이너 시작:

   ```bash
   docker compose up -d robocasa
   ```

4. 웹 브라우저에서 접속:

   ```
   https://<서버IP>:8444
   ```

   - 자체 서명 인증서를 사용하므로 브라우저에서 경고가 나옵니다 → "고급" → "계속 진행"
   - 사용자명: `.env`의 `USER_NAME`
   - 비밀번호: `.env`의 `VNC_PW`

### 4.2 X11 모드 (로컬 우분투 PC)

1. override 파일 복사:

   ```bash
   cp docker-compose.override.local.example.yml docker-compose.override.yml
   ```

2. `.env` 파일에서 `DISPLAY` 설정:

   ```
   DISPLAY=:0
   ```

   확인 방법: 호스트 터미널에서 `echo $DISPLAY`

3. 호스트에서 Docker의 X11 접근 허용 (로그인할 때마다 한 번 실행):

   ```bash
   xhost +local:docker
   ```

4. 컨테이너 시작:

   ```bash
   docker compose up -d robocasa
   ```

5. 컨테이너 내 GUI 앱이 호스트 화면에 직접 표시됩니다.

---

## 5. GPU 설정

`docker-compose.override.yml`에서 각 컨테이너의 GPU 할당을 관리합니다.
기본 설정은 모든 컨테이너가 GPU 0을 사용합니다.

### GPU 번호 변경

`docker-compose.override.yml`에서 두 곳을 변경합니다:

```yaml
services:
  robocasa:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']       # ← GPU 번호 변경
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0      # ← 같은 번호로 변경
```

### Multi-GPU 서버에서 각 컨테이너에 다른 GPU 할당

```yaml
services:
  robocasa:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']       # GPU 0
              capabilities: [gpu]
  xvla:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']       # GPU 1
              capabilities: [gpu]
  dreamvla:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2']       # GPU 2
              capabilities: [gpu]
```

### GPU 확인

```bash
# 호스트에서 확인
nvidia-smi

# 컨테이너 내에서 확인
docker compose exec robocasa nvidia-smi
```

---

## 6. Troubleshooting

### `docker compose build` 시 "permission denied"

**원인:** Docker 그룹에 사용자가 추가되지 않음

**해결:**
```bash
sudo usermod -aG docker $USER
```
실행 후 **로그아웃 → 재로그인** 필요

---

### `nvidia-smi` 실패 (GPU를 찾을 수 없음)

**원인:** NVIDIA Container Toolkit 미설치 또는 드라이버 문제

**해결:**
```bash
# 1. 먼저 호스트에서 nvidia-smi 확인
nvidia-smi

# 2. Container Toolkit 설치 확인
dpkg -l | grep nvidia-container-toolkit

# 3. Docker 재시작
sudo systemctl restart docker
```

---

### `Error response from daemon: could not select device driver "nvidia"`

**원인:** Docker가 NVIDIA 런타임을 인식하지 못함

**해결:**
```bash
# NVIDIA Container Toolkit 재설치
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### X11 모드에서 GUI 화면이 안 뜸

**원인:** `xhost +local:docker` 미실행

**해결:**
```bash
# 호스트에서 실행
xhost +local:docker
```

---

### VNC 접속이 안 됨 (Connection refused on :8444)

**원인:** 포트가 노출되지 않았거나 VNC 서버 시작 실패

**해결:**
```bash
# 1. override 파일이 VNC 템플릿인지 확인
cat docker-compose.override.yml | grep 8444

# 2. VNC 서버 시작 로그 확인
docker compose logs robocasa
```

---

### `docker compose exec robocasa bash` → "no such service"

**원인:** 컨테이너가 아직 실행 중이 아님

**해결:**
```bash
# 먼저 컨테이너 시작
docker compose up -d robocasa

# 상태 확인
docker compose ps
```

---

### 컨테이너 내에서 파일 권한 문제 (Permission denied)

**원인:** `.env`의 `USER_ID`/`GROUP_ID`가 호스트 사용자와 불일치

**해결:**
```bash
# 호스트에서 확인
id -u    # → USER_ID에 입력
id -g    # → GROUP_ID에 입력
```

`.env` 수정 후 이미지를 다시 빌드해야 합니다:
```bash
docker compose build robocasa
```

---

### 디스크 공간 부족

**해결:**
```bash
# Docker 디스크 사용량 확인
docker system df

# 사용하지 않는 이미지/캐시 정리
docker system prune -a
```

---

### 모델 서버 연결 실패 (Connection refused on :8100 / :8200)

**원인:** 해당 모델 컨테이너가 실행 중이 아니거나, 서버 스크립트가 시작되지 않음

**해결:**
```bash
# 1. 컨테이너 시작
docker compose up -d xvla

# 2. 서버 실행
docker compose exec xvla python /temporal_vla/scripts/serve/xvla.py \
  --model-path lerobot/xvla-base

# 3. health check으로 확인
curl http://localhost:8100/health
```

---

## 7. 유용한 명령어 모음

| 명령어 | 설명 |
|--------|------|
| `docker compose ps` | 실행 중인 컨테이너 확인 |
| `docker compose logs -f <service>` | 실시간 로그 |
| `docker compose exec <service> bash` | 컨테이너 셸 접속 |
| `docker compose run --rm xvla <cmd>` | 일회성 명령 실행 |
| `docker compose up -d <service>` | 컨테이너 백그라운드 시작 |
| `docker compose stop <service>` | 컨테이너 중지 |
| `docker compose down` | 모든 컨테이너 중지 + 삭제 |
| `docker compose build --no-cache <service>` | 캐시 없이 재빌드 |
| `docker system df` | Docker 디스크 사용량 |
| `docker system prune -a` | 미사용 이미지/캐시 정리 |
