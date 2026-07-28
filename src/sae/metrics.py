"""평가 지표 — U-coefficient + clock(시간분위) 기준선.

출처: task_classification@88543a2 `phase/metrics/{base,structure,uncertainty,purity,
      silhouette,registry}.py` (https://github.com/robots-oh/task_classification)
이식 근거: docs/steering/29_sae_port_review.md §5.A
      ("EvalContext 에 clock(시간분위) 기준선 내장 → 길이 통제 대조군을 공짜로 얻음")
이식 계획: docs/steering/30_sae_g1_port_handout.md §2.1, §4 Phase A3

[원본 대비 변경]
- 6개 모듈(base/structure/uncertainty/purity/silhouette/registry)을 한 파일로 합쳤다
  (우리 쪽 소비자가 적고 상호참조가 촘촘해서 패키지로 쪼갤 이득이 없다).
- `boundary.py`(BoundaryF1)·`self_transition.py` 는 phase 경계 전용 지표라 이번 lift 범위 밖
  (29 §5.A 이식 목록에도 없음). 필요해지면 같은 출처에서 추가로 가져올 것.
- scipy 의존 없음(원본도 없음). numpy + sklearn(silhouette_score)만 사용 —
  원격 노드(승준)에 scipy 가 없으므로 이 제약은 계속 유지해야 한다.

[우리 용도 메모 — G1/G2]
`EvalContext.cell` 은 동료 기준 task×object 정체성(nuisance 진단)이었다. 우리는 같은 자리에
**scene 라벨(scenario_seed/layout_id/style_id)** 을 넣어 U(scene|z) 를 잰다. 필드 이름은
원본을 유지한다(호출부에서 무엇을 넣었는지 명시할 것). 주의: cell 과 scenario_seed 는
같은 게 아니라 계층이다 — 핸드아웃 §2.2-4.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np

TIERS = ("primary", "auxiliary", "diagnostic")

SAMPLE_CAP = 4000   # silhouette 이 이상이면 서브샘플 (O(N^2) 억제)
SAMPLE_SEED = 0


# ================================================================ 에피소드 구조 헬퍼
# 출처: phase/metrics/structure.py
# 입력은 step 단위다: 상태열 `path [N]`과 에피소드 인덱스 `episode [N]`. 행이 (episode, t)
# 정렬이므로 에피소드는 연속 구간이고, 에피소드 내 이웃은 인접 행이다. 패딩도 mask도 없다.
def episode_bounds(episode):
    """[N] 에피소드 인덱스 → 각 에피소드의 시작 오프셋 [B+1] (연속 구간 전제)."""
    episode = np.asarray(episode)
    starts = np.concatenate([[0], np.flatnonzero(np.diff(episode)) + 1, [len(episode)]])
    return starts


def within_episode(episode):
    """[N-1] bool — i번째와 i+1번째 행이 같은 에피소드인가 (인접 비교 마스크)."""
    episode = np.asarray(episode)
    return episode[1:] == episode[:-1]


def time_fraction(episode):
    """[N] 에피소드 내 진행도 0..1. '시계' 기준선을 만드는 데 쓴다."""
    b = episode_bounds(episode)
    out = np.empty(len(episode), np.float64)
    for s, e in zip(b[:-1], b[1:]):
        n = e - s
        out[s:e] = np.arange(n) / max(n - 1, 1)
    return out


def clock_clusters(tfrac, n_states):
    """같은 스텝 집합 위, 같은 상태 개수의 '시계' 클러스터 (매칭된 대조군).

    시계(에피소드 내 시간 분위)는 시간만 세는 분할이다 — 발견한 상태열이 이걸 못 넘으면
    그 분할은 시간 축의 재명명일 뿐이라는 뜻.

    [우리 용도] 길이/진행도 confound 통제의 기본 대조군 (confound-audit 규약과 정합).
    """
    n_states = max(int(n_states), 2)
    edges = np.quantile(tfrac, np.linspace(0, 1, n_states + 1)[1:-1])
    return np.digitize(tfrac, edges)


def segments(path, episode):
    """[(episode, start, end, state)] — 상태열의 연속 구간 (start/end는 전역 행 인덱스)."""
    path = np.asarray(path)
    b = episode_bounds(episode)
    out = []
    for i, (s, e) in enumerate(zip(b[:-1], b[1:])):
        seg = path[s:e]
        cuts = np.concatenate([[0], np.flatnonzero(np.diff(seg)) + 1, [len(seg)]])
        out += [(i, s + a, s + z, int(seg[a])) for a, z in zip(cuts[:-1], cuts[1:])]
    return out


# ================================================================ 공통 뼈대
# 출처: phase/metrics/base.py
@dataclass
class EvalContext:
    """한 평가 대상(한 split, 혹은 그 부분집합)의 모든 지표 입력.

    지표마다 필요한 입력이 다르다: Uncertainty Coefficient는 (상태열, 라벨)을, Silhouette은
    (잠재, 상태열)을 본다. 시그니처를 지표마다 다르게 두면 다형성을 못 쓰고 호출부가
    분기 지옥이 된다. 그래서 필요한 입력을 한데 묶어 주고, 각 지표가 거기서 필요한 것만
    꺼내 쓰게 한다 — 모든 지표가 `metric.compute(ctx)` 하나로 호출된다.

        ctx = EvalContext(path=..., phase=..., cell=..., episode=..., z=...)
        row = {m.key: m.compute(ctx) for m in registry if m.available(ctx)}

    path/phase/cell/episode는 필수, z(모델 잠재)는 특성공간 지표(Silhouette)에만 필요해
    선택이다. clock·시간분위 같은 파생값은 처음 쓸 때 한 번 계산해 캐시한다.
    """

    path: np.ndarray            # 발견 상태열 [N] — 평가 대상 클러스터링
    phase: np.ndarray           # GT phase [N] (우리 G2 에선 succ/fail 등 outcome 라벨 자리로도 씀)
    cell: np.ndarray            # nuisance 정체성 [N] — 우리 G1 에선 **scene 라벨**을 넣는다
    episode: np.ndarray         # 에피소드 인덱스 [N] (경계 계산용)
    z: np.ndarray | None = None  # 모델 잠재 [N, d] — 없으면 특성공간 지표는 스킵

    def __post_init__(self):
        self.path = np.asarray(self.path)
        self.phase = np.asarray(self.phase)
        self.cell = np.asarray(self.cell)
        self.episode = np.asarray(self.episode)
        if self.z is not None:
            self.z = np.asarray(self.z)

    def __len__(self):
        return len(self.path)

    @cached_property
    def n_states(self) -> int:
        """발견 상태열이 실제로 쓴 상태 개수 (시계 대조군의 상태 수를 여기 맞춘다)."""
        return int(len(np.unique(self.path)))

    @cached_property
    def bounds(self) -> np.ndarray:
        return episode_bounds(self.episode)

    @cached_property
    def clock(self) -> np.ndarray:
        """같은 상태 수의 시계(시간분위) 대조군 [N]."""
        return clock_clusters(time_fraction(self.episode), self.n_states)

    def field(self, name: str) -> np.ndarray:
        """이름으로 상태열/라벨 배열을 고른다 — 지표가 source/target을 문자열로 지정."""
        if name == "clock":
            return self.clock
        arr = getattr(self, name, None)
        if arr is None:
            raise KeyError(f"EvalContext에 '{name}' 필드가 없습니다")
        return arr


class Metric(ABC):
    """모든 평가지표의 베이스. 하위 클래스는 `compute(ctx)` 하나만 구현하면 된다.

    key    metrics.json에 남길 키 (예: "u_phase").
    tier   리포트 표의 그룹·정렬용 태그. 계산 인터페이스와는 무관하다.
    """

    key: str = ""
    tier: str = "diagnostic"

    def available(self, ctx: EvalContext) -> bool:
        """이 지표가 이 ctx에서 계산 가능한가 (예: Silhouette은 z가 있어야)."""
        return True

    @abstractmethod
    def compute(self, ctx: EvalContext):
        """float 하나, 또는 여러 키를 담은 dict를 돌려준다."""
        ...


# ================================================================ Uncertainty Coefficient
# 출처: phase/metrics/uncertainty.py
#
# U(y|z) = (H(y)-H(y|z))/H(y). z가 y의 엔트로피를 몇 % 설명하는가. target 자신의
# 엔트로피로 정규화하므로 같은 z에 대해 U(phase|z)와 U(cell|z)를 **직접 비교**할 수 있다
# (NMI로는 H(cell)!=H(phase) 때문에 불가능).
#
#     u_phase        = U(phase | path)   발견 상태열이 정답 라벨을 얼마나 설명하는가
#     u_cell         = U(cell  | path)   상태열이 nuisance(우리: scene) 정체성을 잡았는지
#     u_clock        = U(clock | path)   상태열이 시계(시간분위)를 얼마나 설명하는가
#     clock_u_phase  = U(phase | clock)  시계 기준선 — 못 넘으면 시간만 센 것
#     cell_u_phase   = U(phase | cell)   cell만으로 라벨이 얼마나 설명되는가 (상한 참고)
def uncertainty_coef(z, y):
    """U(y|z) = (H(y) - H(y|z)) / H(y). z/y는 정수 라벨 배열 [N]."""
    z, y = np.asarray(z), np.asarray(y)
    _, zi = np.unique(z, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    n = len(z)
    cm = np.zeros((zi.max() + 1, yi.max() + 1))
    np.add.at(cm, (zi, yi), 1)
    pij = cm / n
    pz, py = pij.sum(1, keepdims=True), pij.sum(0, keepdims=True)
    nz = pij > 0
    mi = (pij[nz] * np.log(pij[nz] / (pz @ py)[nz])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    # y가 (거의) 상수면 H(y)=0 → U(y|z) 정의 불가. 부동소수점 탓에 hy가 정확히 0이
    # 아니라 ~1e-16이 되면 mi/hy가 0/0으로 폭주하므로 epsilon으로 막는다.
    return 0.0 if hy <= 1e-12 else float(mi / hy)


class UncertaintyCoefficient(Metric):
    def __init__(self, source: str, target: str, key: str, tier: str = "diagnostic"):
        self.source = source   # 예측 역할 배열의 필드명 (path/clock/cell)
        self.target = target   # 정답 역할 배열의 필드명 (phase/cell/clock)
        self.key = key
        self.tier = tier

    def compute(self, ctx: EvalContext) -> float:
        return uncertainty_coef(ctx.field(self.source), ctx.field(self.target))


# ================================================================ Purity
# 출처: phase/metrics/purity.py
def purity(z, y):
    """sum_k max_c |{i: z_i=k, y_i=c}| / N.

    한 클러스터가 얼마나 단일 라벨로 순수한가 (0..1). 단, 상태 수 K가 커질수록 자동으로
    올라가므로 K가 다른 분할끼리는 비교하면 안 된다 — 주 지표는 U-coefficient이고
    purity는 진단으로만 병기한다.
    """
    z, y = np.asarray(z), np.asarray(y)
    _, yi = np.unique(y, return_inverse=True)
    return sum(np.bincount(yi[z == k]).max() for k in np.unique(z)) / len(z)


class Purity(Metric):
    key = "purity_phase"

    def __init__(self, tier: str = "diagnostic"):
        self.tier = tier

    def compute(self, ctx: EvalContext) -> float:
        return float(purity(ctx.path, ctx.phase))


# ================================================================ Silhouette
# 출처: phase/metrics/silhouette.py
def silhouette(z, labels, sample=SAMPLE_CAP, seed=SAMPLE_SEED):
    """잠재 [N,d]와 군집 라벨 [N] → 평균 silhouette. 라벨이 <2종이면 정의 불가 → nan.

    라벨 없는 군집 품질 (-1..1). GT를 안 쓰므로 U-coef(정답 대조)와 상보적이다.
    O(N^2)라 N이 크면 결정적 서브샘플(seed 고정)로 자른다.
    """
    from sklearn.metrics import silhouette_score

    z = np.asarray(z)
    labels = np.asarray(labels)
    if len(z) > sample:
        idx = np.random.default_rng(seed).choice(len(z), sample, replace=False)
        z, labels = z[idx], labels[idx]
    # silhouette은 2 <= 라벨종수 <= N-1 에서만 정의된다.
    if not (2 <= len(np.unique(labels)) <= len(labels) - 1):
        return float("nan")
    return float(silhouette_score(z, labels))


class Silhouette(Metric):
    key = "silhouette"

    def __init__(self, sample: int = SAMPLE_CAP, seed: int = SAMPLE_SEED,
                 tier: str = "primary"):
        self.sample = sample
        self.seed = seed
        self.tier = tier

    def available(self, ctx: EvalContext) -> bool:
        # 특성공간 지표라 잠재 z가 반드시 필요하고, 상태가 최소 2종은 있어야 한다.
        return ctx.z is not None and ctx.n_states >= 2

    def compute(self, ctx: EvalContext) -> float:
        return silhouette(ctx.z, ctx.path, self.sample, self.seed)


# ================================================================ 레지스트리
# 출처: phase/metrics/registry.py (BoundaryF1·SelfTransition 항목은 미이식이라 제외)
U = UncertaintyCoefficient

# 순서는 리포트 가독성용 (계산엔 무관). 티어: primary → auxiliary → diagnostic.
DEFAULT_METRICS = [
    U("path", "phase", key="u_phase", tier="primary"),
    Silhouette(tier="primary"),
    U("clock", "phase", key="clock_u_phase", tier="auxiliary"),
    U("path", "cell", key="u_cell", tier="diagnostic"),
    U("path", "clock", key="u_clock", tier="diagnostic"),
    U("cell", "phase", key="cell_u_phase", tier="diagnostic"),
    Purity(tier="diagnostic"),
]


def derived_flags(row: dict) -> dict:
    """지표 dict에서 파생 판정을 계산한다 (지표 사이의 비교라 개별 Metric이 아님).

    beats_clock   상태열이 시계 기준선을 넘었는가 (U(phase|path) > U(phase|clock)).
    cell_dominant 상태열이 phase보다 cell 정체성을 더 잡았는가.
                  (동료에겐 nuisance 경보였지만 **우리 G1 에선 오히려 목표 신호**다 —
                   cell 자리에 scene 라벨을 넣었을 때 이게 True 여야 scene 인코딩 증거.)
    """
    out = {}
    if "u_phase" in row and "clock_u_phase" in row:
        out["beats_clock"] = row["u_phase"] > row["clock_u_phase"]
    if "u_cell" in row and "u_phase" in row:
        out["cell_dominant"] = row["u_cell"] > row["u_phase"]
    return out


def evaluate(ctx: EvalContext, metrics=None) -> dict:
    """ctx 하나에 대해 모든 가용 지표 + 파생 플래그를 계산한다 (우리 추가 편의 함수)."""
    metrics = DEFAULT_METRICS if metrics is None else metrics
    row: dict = {}
    for m in metrics:
        if not m.available(ctx):
            continue
        val = m.compute(ctx)
        if isinstance(val, dict):
            row.update(val)
        else:
            row[m.key] = val
    row.update(derived_flags(row))
    return row
