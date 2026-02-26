# Contributing Guide

## 개발 워크플로우

### 1. 작업 시작

```bash
# dev 브랜치에서 최신 코드 pull
git checkout dev
git pull origin dev

# 작업 브랜치 생성
git checkout -b feat/my-feature    # 기능 개발
git checkout -b exp/my-experiment  # 실험
git checkout -b fix/my-fix         # 버그 수정
```

### 2. 작업 중

```bash
# 주기적으로 커밋
git add .
git commit -m "feat: add new model architecture"

# 원격에 push
git push origin feat/my-feature
```

### 3. 머지 요청 (PR)

1. GitHub에서 Pull Request 생성 (`feat/my-feature` → `dev`)
2. 팀원 최소 1명 리뷰
3. 충돌 해결 후 머지
4. 머지된 브랜치 삭제

### 4. 릴리즈 (main 머지)

- 주요 마일스톤 달성 시 `dev` → `main` PR 생성
- 전원 합의 후 머지
- 태그 생성: `v0.1.0` 등

## 커밋 메시지 컨벤션

```
<type>: <description>

타입 목록:
- feat: 새로운 기능/모델 추가
- fix: 버그 수정
- exp: 실험 관련 (config, 실험 스크립트 등)
- docs: 문서 변경
- refactor: 리팩토링
- config: 설정 변경
- script: 스크립트 추가/수정
```

## 실험 관리

### 실험 브랜치 네이밍

```
exp/<date>-<short-description>
exp/0301-peract-lr-sweep
exp/0305-lerobot-policy
```

### 실험 기록

실험 결과는 팀 공유 문서(Notion/Google Docs)에 기록:
- 실험 목적
- 변경 사항 (하이퍼파라미터, 모델 구조 등)
- 결과 (metrics, 정성적 관찰)
- 다음 단계

### 체크포인트 관리

- 체크포인트는 git에 커밋하지 않음 (.gitignore)
- 공유가 필요한 경우 Google Drive 또는 서버의 공유 디렉토리 사용
- 중요 체크포인트는 명확한 네이밍: `{model}_{task}_{date}_{metric}.pth`