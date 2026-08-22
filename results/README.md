# 보존된 체크포인트 안내

`human-controller` 연구에서 판단 근거로 쓰는 체크포인트만 Git LFS로 보존한다.
서버(`server1:~/human-controller/ProtoMotions/results/`)에는 130개 이상의 run이
있지만, 여기에는 아래 6개만 있다. 각 디렉토리는 `last.ckpt`(스크립트 호환용) 와
동일 내용의 `epoch_N_final.ckpt`(불변 근거) 를 함께 둔다. 상세 sha256·크기는
`../../docs/EXECUTION_ARTIFACTS.md` 참조.

## 블록 용어

- **④ Motion Driver**: 속도/방향 명령을 받아 관절 target을 내는 보행 정책.
- **⑥ Posture Stabilizer**: 관절 target에 작은 residual(δ)을 더해 균형을 잡는
  정책. `HC_POSTURE_GAIN`(0~1)으로 평가 시 기여량을 줄여 볼 수 있다.
- **Stage 3**: ⑥을 **frozen**으로 두고 그 위에서 ④를 학습한 합성 정책. ⑥과 ④가
  **별도 ckpt**이므로 실행 시 둘 다 지정해야 한다:

  ```bash
  export HC_POSTURE_FOUNDATION_CKPT=results/<⑥ dir>/last.ckpt   # frozen ⑥
  # ④ = 아래 Stage 3 디렉토리의 last.ckpt (checkpoint 인자로 로드)
  export HC_POSTURE_GAIN=1.0                                    # 0.75 / 0.5 / 0.0 으로 sweep
  ```

## 디렉토리 요약

| 디렉토리 | 블록 | 한 줄 설명 | 핵심 수치 | 근거 문서 |
|---|---|---|---|---|
| `p3_locomotion_omni_uniform_speed_s80_task70_foot05_v2lateral` | ④ **V11** | ⑥ 없이 단독 학습한 omnidirectional 보행 기준선. 이후 모든 실험의 비교 기준 | speed_mae **0.110** | `docs/2026-07-20-current-locomotion-status.md` |
| `p6_posture_stand_smoke_r2` | ⑥ smoke | V11 위에 ⑥ 배선이 돌아가는지만 확인한 smoke. **효능 검증 아님** | — | `docs/2026-07-24-posture-stabilizer-training-structure-and-smoke-result.md` |
| `p6_F_seed1_25m` | ⑥ **F** (정지 bank) | 정지 자세 256개 bank에서 anchor-free로 학습한 ⑥. 서기·push 회복은 가장 좋았으나 보행 중에는 상수 편향으로 퇴화 | clean 236/256, push 230/256 | `docs/2026-08-01_anchor_free_final_results.md` |
| `s3_A_scratch4_on_frozen6` | ④ Stage 3 **A** | frozen ⑥ F 위에서 ④ from scratch. **gain을 0.75로만 낮춰도 기울며 휘청, 0.50에서 낙상** — ④가 ⑥에 균형을 위임한 사례 | gain 1.0: speed_mae 0.227, upright 0.978 → 0.75: 0.397 / 0.922 → 0.50: 1.084 / **0.247** | `docs/2026-08-01_stage3_frozen_posture_walking_results.md` |
| `p6_W_locobank_25m` | ⑥ **W** (보행 bank) | 보행 자세 256개 bank에서 재학습한 ⑥. 개루프에서 δ가 기울기와 양의 상관(+0.57) | corr(\|δ\|, tilt) +0.574 | `docs/2026-08-02_posture6_locomotion_bank_results.md` |
| `s3_W_scratch4_on_p6_W_locobank_25m` | ④ Stage 3 **W** | frozen ⑥ W 위에서 ④ from scratch. **현재 최고 성능 합성 정책**. gain 0.75에서는 서 있으되 속도 추종만 잃고 0.50에서 낙상 | gain 1.0: speed_mae **0.112**, 보행 중 tilt 0.065 (⑥ 없는 대조군 0.173) | `docs/2026-08-02_posture6_locomotion_bank_results.md` |

## 세트로 써야 하는 조합

| 목적 | ⑥ ckpt | ④ ckpt |
|---|---|---|
| 기준선 (⑥ 없음) | — | `p3_locomotion_omni_…_v2lateral/last.ckpt` |
| 최고 성능 합성 | `p6_W_locobank_25m/last.ckpt` | `s3_W_scratch4_on_p6_W_locobank_25m/last.ckpt` |
| gain↓ 시 휘청거리는 합성 (의존성 시연용) | `p6_F_seed1_25m/last.ckpt` | `s3_A_scratch4_on_frozen6/last.ckpt` |

## 각 디렉토리 내용물

- `last.ckpt`, `epoch_N_final.ckpt` — 동일 내용. Lightning 체크포인트(`["model"]` 에 state_dict).
- `resolved_configs*.yaml/.pt` — 학습/추론 시 실제 적용된 전체 config.
- `experiment_config.py`, `config.yaml` — 실험 정의 원본.
- `lightning_logs/version_0/hc_metrics.jsonl` — epoch별 학습 지표 (`"epoch"` 키로 최종 epoch 확인).
- `evaluations/*.json` — 평가 결과. `"checkpoint"` 필드에 어떤 ckpt를 평가했는지,
  `speed_mae_mps` 등에 수치가 있다. (Stage 3 디렉토리에만 있음.)
- `env_*.pt.ckpt` — env 보조 상태(수 KB).

## 의도적으로 제외한 것

- Stage 3 run의 `epoch_1000.ckpt`, `score_based.ckpt` (각 180MB) — LFS 용량.
  평가에 쓰인 것은 전부 `last.ckpt` 다.
- `s3_B_warm4_on_frozen6` (V11 웜스타트 arm), `s3_C_scratch4_alone` (⑥ 없는 대조군),
  `p6_X_mixbank_25m`, `p6_D/K1` 등 — 서버에만 있음.
- 모든 `last.ckpt`는 25M-frame 예산을 완주한 run의 것이다. 50M run에서는
  33M→50M 사이 붕괴 사례가 있으므로 다른 run의 `last.ckpt`를 검사 없이 쓰지 말 것.
