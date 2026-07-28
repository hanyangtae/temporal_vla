# conceptor vs 평균차이 — 연산자 비교

## Conceptor (현행 exp3(구 pq3))

```
R_c = (1/N) (X_c − μ_c)ᵀ (X_c − μ_c)
```

클래스별 자기 평균으로 중심화 → μ 차이 소멸

```
C_c = R_c (R_c + α⁻² I)⁻¹                    c ∈ {success, failure}

C_steer = C_succ ∧ ¬C_fail
        = (C_succ⁻¹ + (I − C_fail)⁻¹ − I)⁻¹

h′ = h Mᵀ ,   M = (1−β) I + β C_steer ≈ (1−β) I
```

C_steer 가 데이터 위에서 ≈0 → 전방향 균일 축소

## 평균차이 + directional ablation (제안)

```
r̂ = normalize(μ_fail − μ_succ)
```

중심화 안 함 → μ 차이가 곧 신호

```
h′ = h Mᵀ ,   M = (1−β) I + β (I − r̂ r̂ᵀ) = I − β r̂ r̂ᵀ
```

한 방향만 축소, 나머지 1535차원 항등

## 한 줄 대비

| | 무엇을 fit | M의 고유값 | 방향 비중 |
|---|---|---|---|
| conceptor | 2차 모멘트 모양 (평균 제거 후) | 전부 ≈1−β | 3.1% |
| 평균차이 | 두 클래스 평균 차이 | 1−β 하나, 1이 1535개 | 100% |

핵심은 첫 줄입니다. conceptor는 X − μ_c 로 시작하기 때문에 μ_succ − μ_fail 이 fit 이전에 사라지고,
평균차이 방향은 바로 그 사라진 항을 신호로 씁니다. 서로 정확히 여집합이에요.

그리고 두 방법 모두 h′ = h Mᵀ 라는 같은 곱셈형 연산자 틀에 들어가므로
(build_steering_matrix 의 (1−β)I + βC 에 C = I − r̂ r̂ᵀ 를 넣으면 두 번째 식이 그대로 나옴),
serve 경로는 수정 없이 재사용됩니다.
