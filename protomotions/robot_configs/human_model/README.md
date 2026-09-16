# Human model — 공통 코드와 버전별 자산

> 분류: **내부 구현·실험 기록**. 2026-09-16 사용자 승인으로 폴더를 분리하고 v2의 teacher 연결 전 plant를 구현했다. v1의 자산·기본 profile은 보존한다.

## 버전 선택과 소유 경로

| 경로 | 역할·상태 |
|---|---|
| `common/` | 공통 계산·연결·metric·검증/보정 도구. v1/v2 force 계산과 IsaacLab 연결. v1 opt-in strength/activation/fatigue 후보를 v2에서 자동 실행하지 않음 |
| `human_model_v1/` | 현재 모델. `model_config.py`, `joint_map.json`, `profiles/`, `assets/`, SHA/provenance `manifest.json` |
| `human_model_v2/` | 59좌표 plant: MyoLeg 기반 하지 고정 축, ankle/subtalar 분리. 인체자료 보정 수동 profile 채택; 독립 생체 인증 아님 |
| `registry.py` | `human_model_v1`와 기존 `healthy_adult_v1` 이름 연결, v2 자산·profile 선택 |
| `tests/common/`, `tests/human_model_v1/`, `tests/human_model_v2/` | 공통 지표/호환성, v1 물리·연결, v2 좌표·자산/ROM·에너지·질량 보존 검사 |

기존 root Python 모듈은 호환 import/CLI 진입점이며 구현 복사본이 아니다. 기존 세 JSON과 기존 human-model USDA 경로는 canonical 자원으로 연결한 상대 symlink다. 기존 checkpoint의 import와 명령을 보존하기 위해 이 경로는 의도적으로 유지한다. 공용 legacy SMPL MJCF는 기존 자산 위치를 유지하고 v1 `assets/smpl_humanoid.xml`에서 참조한다. Linux symlink 지원 checkout을 기준으로 한다.

`PROTOMOTIONS_HUMAN_MODEL=human_model_v1`와 `healthy_adult_v1`은 같은 기본 물성을 선택한다. v2는 아래 전용 factory로 선택한다. 기존 69좌표 설정에 이름만 v2로 붙이면 좌표 불일치로 거부한다. 기본 선택은 기존 `healthy_adult_v1`이다. `profiles/population_profile_candidate.json`는 이전 실험의 보정 후보를 SHA 그대로 보존한 자료이며, v1 runtime 기본값으로 자동 채택하지 않는다. v2는 별도 profile에 명시적으로 반영하고 다시 검증했다. 과거 assay 실행에서는 명시적인 `candidate_profile_path`로 선택했다.

`python -m protomotions.robot_configs.human_model.validation` 및 `.calibration`은 유지되며 canonical 실행 경로는 `.common.validation`, `.common.calibration`이다. v1 재현 기준은 manifest의 **분리 전 baseline commit**과 자원 SHA다. 이후 공통 코드가 바뀌면 그 commit 또는 해당 실행의 코드 SHA로 재현해야 한다. 과거 원시 trace/영상은 상위 `output/`에 남고 이 패키지에 복사하지 않는다.

## v2 실행 계약 — teacher 연결 전

```python
from protomotions.robot_configs.human_model.human_model_v2.model_config import robot_config
robot = robot_config()  # IsaacLab, 59 scalar coordinates; TORQUE input
```

- 하지 한쪽7개: Hip_y/x/z(굴신→내외전→회전 순서), Knee_y, Ankle_y(부위 Talus), Subtalar_x(부위 Ankle/calcaneus), Toe_y. 상체45좌표 유지. 실제 부위26개와 수치 프레임34개. x/y/z 접미사는 좌표 식별자이며 새 하지 축이 직교축이라는 뜻이 아니다.
- `assets/myoleg_joint_reference.json`에 MyoLeg 원본 commit/SHA, 축·offset을 보존한다. 무릎 연동 운동·근육·건은 구현하지 않는다. 중립 SMPL 형상과 anthropometry는 유지한다. Talus는 기존 발 질량10%를 나누고 중립 총 COM/관성을 보존한 가정이다.
- `profiles/healthy_adult_v2.json`: v1 population 후보의 knee/passive toe/wrist 보정을 재사용한 v2 profile. 하지 힘 법칙은 새 고정 축으로 옮긴 대용값이며 subtalar strength는 v1 ankle inversion/eversion proxy다. v1 기본 profile은 변하지 않는다.
- v2의 `active_strength_model`은 Anderson2007 Eq9의 hip/knee/ankle 굴신6방향을 양쪽에 기본 적용한다. 각 관절 자신의 q/qd에 따라 능동 상한을 계산한다. `active_nm`은 해당 축의 실제 동적 상한이 아니라 기존 정적 참조값이며, 현재 상한은 `strength_caps(q, qd)` / simulator `human_negative_caps`, `human_positive_caps`로 읽는다. 나머지 축은 정적 proxy를 유지한다. 문헌 범위 밖은 경계값 유지와 flag이며 생체 검증 범위가 아니다. v1 optional `strength`와 별개로 v2 profile에 명시되어 있으며 기본 v1 물성은 바뀌지 않는다.
- 입력은 각 환경의 59차원 요청 토크와 현재 q/qd, 출력은 능동/수동/합산 토크. IsaacLab이 중력·contact·ROM을 해석하고 상태를 반환한다. 공개 상태 순서는 `joint_map.json`과 config metadata를 따른다. HumanJointModel 자체에는 q_ref 제어기가 포함되지 않는다.
- USD는 실행 자산이고 MJCF는 metadata/별도 FK·관성 교차검사 자산이다. v2 MuJoCo/IsaacGym runtime은 이 단계에서 검증하지 않으며 비IsaacLab 동역학 경로를 거부한다. v1 optional force candidates도 새 축에 자동 적용하지 않는다.
- Native 시험은 공통 `validation --scenario suspended_gait/population_load/v2_suite --population-config ...`와 v2 `validation.py`를 사용한다. 설정·결과·소스 bundle: workspace `output/260916/e02_human_model_v2/`. 단관절 ROM은 비검사 축 clamp, 발 하중은 외부 harness/비발 자세 clamp를 명시한다. 자력 서기/보행·retargeter·teacher 연결/학습은 포함하지 않는다.
- 모델 자산/힘 profile뿐 아니라 armature·solver·dt·접촉·외부 시험 지지 조건도 결과 해석에 필요하다. 지원발 시험의 TGS external-forces-every-iteration 설정은 다른 시험과 구분한다. 상세 수치·실패 기록은 상위 `docs_ghlee/archive/experiments/human_model_joint_design/README.md`와 manifest를 따른다.

## v1 모델 정의와 과거 검증 범위

이 디렉터리는 SMPL 강체 모델의 **관절 RoM, 방향별 능동 토크 한계, 수동 탄성·감쇠**를 관리한다. 기본 프로필은 `healthy_adult_v1.json`이다. 서로 다른 연구의 건강한 성인 측정 평균을 조합했으며, 근거가 없는 축은 가정 또는 미구현으로 명시했다. 하나의 모집단에서 측정한 전신 평균 데이터는 아니다.

기존 SMPL의 실제24부위,69개 hinge DOF 이름·순서,형상 및 원본 MJCF의 질량·관성을 보존한다. MJCF 총질량은 약63.314kg이다. IsaacLab은 XYZ 직렬 hinge 연결을 위해 보이지 않는 수치용 프레임46개(각1e-6kg, 등방관성1e-9kg·m²)를 추가하며 공개 body state는 실제24부위만 반환한다. 이 미세한 추가 질량은 해부학적 조직이 아니다. 따라서 **평균 성인의 체형·질량까지 새로 만든 모델은 아니다.** 토크를 체중에 비례해 임의로 보정하지 않았다.

## 파일과 실행

| 파일 | 역할 |
| --- | --- |
| `human_model_v1/profiles/healthy_adult_v1.json` | 69개 축의 값, 단위, 방향, 근거 ID, 가정; v1 기본 profile |
| `common/profile.py` | 프로필 로딩·검증 |
| `common/dynamics.py` | 방향별 능동 토크 제한과 별도의 수동 관절 힘 |
| `human_model_v1/assets/smpl_humanoid_healthy_adult_v1.usda` | **RoM·원본 부위 질량/COM/관성을 저장한 편집 원본; 기본 실행은 직렬 hinge 캐시로 변환** |
| `human_model_v1/assets/builder.py` | USDA의 RoM 일치 검사; IsaacLab 직렬 hinge 및 MuJoCo/IsaacGym MJCF 캐시 생성 |
| `common/integration.py` | 새 설정 및 이전 체크포인트에 프로필 적용 |
| `tests/human_model_v1/test_human_model.py` | 에너지·힘·축 방향·자산·실제 MuJoCo 경로 검증 |

이 패치를 적용하면 `SmplRobotConfig`의 기본 프로필이 자동으로 활성화된다. `resolved_configs_inference.pt`에 저장된 과거 설정도 시뮬레이터 생성 시 다시 적용한다. 정책 네트워크의 가중치나 입출력 순서를 변환하지 않는다. 기존 PD 이득은 능동 명령 계산에 유지한다. 생체 토크 제한을 적용하므로 과거 정책의 균형·보행 성능은 재평가해야 한다.

```bash
# ProtoMotions 디렉터리에서 기존 실행 명령을 그대로 사용
PROTOMOTIONS_HUMAN_MODEL=healthy_adult_v1 python ...

# 이전 모델과 비교할 때만 사용 (프로그램 시작 전에 지정)
PROTOMOTIONS_HUMAN_MODEL=legacy python ...

# CPU 검증; 프로젝트 의존성과 MuJoCo가 설치된 Python
python -m pytest -q protomotions/robot_configs/human_model/tests
```

IsaacLab은 [smpl_humanoid_healthy_adult_v1.usda](human_model_v1/assets/smpl_humanoid_healthy_adult_v1.usda)를 편집 원본으로 읽고, 기본 `human_model_usd_joint_mode="serial"`에서 X→Y→Z 단일축 연결 자산을 캐시하여 실행한다. 기존 D6 표현은 `"d6"`로 명시할 때 진단용으로 실행할 수 있지만 강한 토크 시험의 ROM 검증에 실패했다. 69축의 `limit:rotX/Y/Z:physics:low/high`에 아래의 RoM을 **도(°) 단위로 저장**했다. 이 텍스트 파일은 일반 Git으로 관리하므로 이 파일을 위한 LFS 다운로드는 필요 없다. 기존 `smpl_humanoid.usda`는 legacy 모델용으로 유지한다.

원본 USDA와 RoM은 실행 중 덮어쓰지 않는다. 직렬 hinge 캐시는 원본·변환 코드·수치 프레임 질량의 내용 해시로 구분해 생성한다. RoM을 변경할 때는 이 USDA와 JSON의 `rom_deg`를 함께 수정한다. JSON은 행동 범위와 MJCF 백엔드에 사용하며, IsaacLab 시작 시 USDA와 일치하는지 검사한다. MuJoCo/IsaacGym은 같은 JSON에서 만든 MJCF를 `/tmp/protomotions-human-model-<uid>/<content-hash>/`에 캐시한다. MJCF 캐시 위치는 `PROTOMOTIONS_HUMAN_MODEL_CACHE`로 지정할 수 있다.

연결한 엔진은 MuJoCo, IsaacLab, IsaacGym이다. Genesis/Newton은 검증되지 않은 적용을 막기 위해 오류를 낸다. **CPU MuJoCo 및 GPU IsaacLab의 고정 골반·무중력 L0/L1 16초 시험을 검증했다(seed2410). IsaacGym 동역학 및 전신 보행은 이 검증에 포함되지 않는다.**

## 값 선택과 근거

각 JSON 항목의 `rom_deg`는 `[하한, 상한]`, `active_nm`은 `[음의 방향 최대 토크 크기, 양의 방향 최대 토크 크기]`다. 양쪽 모두 크기는 0 이상이다. 계산은 rad, rad/s, Nm, Nm/rad, Nms/rad를 사용한다. 임상 ROM의 평균 종말각을 대표 모델의 경계로 채택했으며, 모든 성인의 절대 해부학적 한계라는 의미는 아니다.

| ID / 원문 | 채택한 자료와 조건 |
| --- | --- |
| [CDC2011 / Soucie et al.](https://archive.cdc.gov/www_cdc_gov/ncbddd/jointrom/index.html) | 건강한 20–44세 여성·남성의 평균 ROM을 1:1 산술평균. 예: 무릎 굴곡 `(141.9+137.7)/2=139.8°`. 성별 표본 수로 가중한 전체 평균은 아니다. |
| [MORIN2023](https://doi.org/10.1186/s12891-023-06400-2) | 건강한 성인 30명, 남17/여13, 21–69세, 평균41.3세·71.8kg. Table 2의 intra-rater `Mean (S1+S3)/2` Nm. 고정 장치로 측정한 등척성 토크이며 모든 각도·속도에서의 최대값은 아니다. [저자 기관 PDF](https://constellation.uqac.ca/id/eprint/9209/1/s12891-023-06400-2.pdf). |
| [DANNESKIOLD2009](https://doi.org/10.1111/j.1748-1716.2009.02022.x) | 20–29세 여성18/남성10 그룹의 **등척성** 평균을 1:1 평균. 몸통·고관절 내전·어깨 내전·발목 저측굴곡에 사용. 표의 일부 `N` 표기 대신 결과 본문에서 단위가 `Nm`임을 확인했다. |
| [SILDER2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2020832/) | 건강한 성인20명, 남9/여11, 평균 연령 각각26.1/25.5세. Table 4 평균 계수와 Appendix A의 수동 모멘트. 5개 단관절 항과 대퇴직근·햄스트링 2관절 항을 사용. |
| [FORMICA2012](https://doi.org/10.1152/jn.01014.2011) | 건강한 성인10명, 남7/여3, 24–42세. Table 2 전체 평균 중 요측/척측 편위, 회내/회외 수동 강성. 손목 굴곡/신전 강성은 아래 연구와 중복 합산하지 않았다. |
| [NGUYEN2020](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2020.00023/full) | Table 2, 25–45세 여성37/남성41 그룹을 1:1 평균. 이완 상태 손목 ±10°, 3–12Hz 실험의 EL/VI를 사용. 강성 `(1.64+2.22)/2=1.93 Nm/rad`, 감쇠 `(1.89+2.56)/2=2.225 Nms/rad`. 전체 ROM·다른 주파수로의 확장은 가정이다. |
| [KOYKKA2025](https://doi.org/10.1088/1361-6579/adf488) | 활동적인 성인35명, 남20/여15, 30±7세. Table 2 비우세측 test/retest 평균: 회내8.78Nm, 회외9.78Nm를 양측에 적용. 좌우·우세손 차이는 생략했다. |
| [NIEWIADOMSKI2019](https://doi.org/10.1186/s12998-018-0223-x) | 현재 목 통증 없는97명, 평균28.3세; 과거 통증 이력자는 포함. 이상치 제외 후 총 ROM: 굴곡+신전127.4°, 측굴 합89.3°, 회전 합146.4°. 방향 대칭 및 Neck/Head 균등 분배는 가정이다. |

조회일: 2026-09-08. DOI와 원문 내 표 위치를 함께 기록했다. 질병, 노인, 선수 등 특화 모델은 별도 프로필로 추가한다.

### 주요 관절 적용값

이 표는 **임상 방향**으로 표시한다. JSON은 SMPL 축 부호로 변환한 값이다.

| 운동 | ROM(°) | 최대 능동 토크(Nm) | 근거·보완 |
| --- | ---: | ---: | --- |
| 고관절 굴곡 / 신전 | 132.1 / 17.75 | 134.9 / 219.5 | CDC / Morin |
| 고관절 외전 / 내전 | 45 / 30 | 126.6 / 170.5 | ROM 가정; Morin / Danneskiold |
| 고관절 내회전 / 외회전 | 45 / 45 | 73.8 / 54.8 | ROM 가정; Morin |
| 무릎 굴곡 / 과신전 방향 | 139.8 / 1.3 | 굴곡93.1 / 신전149.0 | CDC / Morin |
| 발목 배측 / 저측굴곡 | 13.25 / 58.35 | 23.7 / 118.3 | CDC; Morin / Danneskiold |
| 발목 내번 / 외번 | 30 / 15 | 20.6 / 20.6 | ROM 가정; 외번 측정값을 내번에도 대입 |
| 어깨 거상 | 0–180 | 외전80.8 / 내전65.8 | ROM 가정; Morin / Danneskiold |
| 어깨 굴곡 / 신전 | 170.4 / 60 | 70.1 / 70.1 | 굴곡 CDC·Morin; 신전 ROM 가정·토크 대용값 |
| 어깨 내 / 외회전 | 90 / 90 | 35.4 / 26.6 | ROM 가정; Morin |
| 팔꿈치 굴곡 / 과신전 방향 | 147.3 / 2.75 | 굴곡53.2 / 신전32.5 | CDC / Morin |
| 전완 회내 / 회외 | 79.45 / 87.8 | 8.78 / 9.78 | CDC / Köykkä |
| 손목 굴곡 / 신전 | 87.59 / 68.23 | 10.5 / 7.6 | Nguyen / Morin |
| 손목 요측 / 척측편위 | 20 / 30 | 7.6 / 10.5 | ROM 가정; 신전/굴곡 토크를 대용 |
| 몸통 굴곡 / 신전 | 총60 / 25 | 121.75 / 284 | ROM 가정; Danneskiold, 세 직렬 관절에 같은 토크 |

### 축과 좌표계

기준 자세는 upstream SMPL T-pose, 전방+x·좌측+y·상방+z다. 관절 자체의 x/y/z hinge 순서는 바꾸지 않는다.

* `Hip_y`: 음수 굴곡, `Knee_y`: 양수 굴곡, `Ankle_y`: 음수 배측굴곡. 전방 운동은 MuJoCo 순기구학으로 확인한다.
* `Shoulder_x`: 왼쪽 -90°, 오른쪽 +90°가 팔을 내린 자세다. T-pose 0°에 임상 거상 0°를 그대로 대입하지 않는다. 거상량은 `90 + side_sign*q_x`다.
* 팔꿈치는 왼쪽 `Elbow_z` 음수, 오른쪽 양수로 굴곡한다. x/y는 ±0.5°로 제한한다.
* 회내/회외는 `Wrist_y`에 배치한다. `Elbow_y`에 함께 배치하면 같은 운동 범위·힘이 두 번 생기고, hinge 순서상 팔꿈치 굴곡면까지 회전한다. 이 원위부 대용 관절은 실제 전완의 회전 관성 분포를 정확히 재현하지 않는다.
* 손목은 T-pose에서 손바닥 아래 방향을 가정한다. 왼쪽 `Wrist_x` 음수가 굴곡, `Wrist_z` 음수가 요측 편위, `Wrist_y` 양수가 회외다. 오른쪽은 부호를 반전한다.
* 3차원 임상 ROM을 독립 hinge의 직사각형 범위로 근사했다. 특히 어깨의 자세 의존적 가동 범위, 견갑상완 리듬, 회전 결합까지 보장하지 않는다.

### 수동 특성과 능동 구동의 구분

`tau_total = clip_directional(tau_active_requested) + tau_elastic(q) + tau_damping(qdot)`

PD 명령의 `Kp*(target-q)-Kd*qdot`는 능동 요청이다. 원래 XML의 stiffness 300–1000, damping 30–100을 인체 수동 특성으로 해석하지 않는다. USDA에는 RoM과 원본 MJCF의24부위 질량·COM·주관성·주축을 명시한다. 2026-09-13 조사에서 기존 USD의 부위별 밀도 누락으로45.28kg이던 native 질량을 원본63.31kg과 일치하도록 수정했다. IsaacLab의 actuator 설정이 실행 시 drive 강성·감쇠를 0으로 지정하고, 최대 토크 제한과 passive 항은 코드에서 적용한다. MuJoCo/IsaacGym용 파생 MJCF도 기존 spring/damping을 0으로 한다. 수동 항은 매 물리 step에서 한 번만 더한다. 액추에이터는 이 합을 받을 수 있는 충분한 한계를 사용하고, 생체 최대 능동 토크는 합산 **전**에 방향별로 제한한다. 수동 토크나 관절 제한의 반력은 근육 최대 토크와 다른 양이다.

Silder 하지는 `q_h=-Hip_y`, `q_k=Knee_y`, `q_a=-Ankle_y`로 변환한다. JSON의 각 항은 `E=scale*exp(a·q+offset)`, `tau=-∂E/∂q` 형식이다. hip flexor/extensor, knee flexor/extensor, plantarflexor와 RF/HAM 연결을 구현한다. 논문 α(°)는 rad로 변환했다. **비복근의 2관절 항은 부호·각도 규약 교차 검증을 마치지 못해 이번 버전에서 제외했다.** 평균 계수를 비선형식에 넣은 결과가 개인별 수동 토크 곡선의 산술평균과 같다는 가정도 하지 않는다.

Formica 편위 강성은 요측1.710/척측1.245Nm/rad, 회내0.240/회외0.186Nm/rad이다. Nguyen의 손목 굴신 강성·감쇠와 함께 원점 기준으로 적용한다. 손목 외의 점성 감쇠는 선택할 수 있는 측정 근거가 없어 0으로 둔다. **0은 조직 저항이 실제로 없다는 뜻이 아니라 해당 항의 미구현 표시다.** 어깨·팔꿈치·목·몸통 등에는 이번 버전에서 측정 기반 중앙 수동 강성을 넣지 않았다. 엔진의 ROM 제한 반력은 별도로 존재한다.

극단 자세에서 지수 폭주를 막기 위해 각 지수 에너지를 지수값200 이후 접선으로 연장한다. 손목 선형 스프링 토크와 점성 토크는 각각30Nm에서 포화한다. 이 값들은 **수치적 외삽 가드이며 생체 측정값이 아니다.** 결합 토크를 관절마다 독립 clip하는 대신 에너지를 연장하여 보존성을 유지하고, 점성 포화도 `tau_d*qdot <= 0`을 유지한다.

### 측정되지 않은 모델 구조의 가정

* 무릎 x/z, 발목 z, 발가락 x/z, 팔꿈치 x/y, 말단 Hand x/y/z는 ±0.5°, 능동 토크0으로 둔다. 차원 호환성을 위한 거의 고정된 축이다. 실제 무릎의 3차원 부수 운동이나 손가락 모델을 구현한 것은 아니다.
* Toe_y는 -60..30°, 음/양 토크5/10Nm의 단일 MTP 대용값이다.
* 좌우 Thorax는 견갑대 대용으로 각 축 ±15°, 양방향20Nm이다. 실제 한 관절의 평균값이 아니다.
* Torso/Spine/Chest에 총 몸통 ROM을 1/3씩 분배한다. 측굴 총±35°, 축회전 총±45°, 각 축 토크60/40Nm는 가정이다. 직렬 관절 토크를 1/3로 나누지 않는다.
* Neck/Head에는 총 목 ROM을 방향 대칭으로 나누고 다시 1/2씩 분배한다. 각 관절 굴곡20/신전30Nm, 측굴20Nm, 축회전10Nm는 가정이다.
* 상수 토크 한계만 제공한다. 각도·각속도·피로·활성화에 따른 근력 변화, 인대 손상, 반사, 개인별 근육 길이·모멘트암은 모델링하지 않는다.

JSON의 `ASSUMPTION`, `PROXY`, `UNMEASURED`를 측정 평균과 구분해서 검토해야 한다. 서로 다른 연구의 두 측정 방향을 조합한 축도 `note`에 각각의 출처를 기록한다. 향후 개선은 이러한 가정을 측정값으로 교체하는 것부터 진행할 수 있다.

## 패치 적용 범위

이 패치는 ProtoMotions 내부에 적용한다. 연결 지점은 `robot_configs/smpl.py`, simulator의 `base_simulator/simulator.py`, `isaaclab/simulator.py`, `isaaclab/utils/scene.py`, `mujoco/simulator.py`의 설정·로딩·힘 적용·body 순서 처리다. 모델 패키지와 실제 USDA를 추가하며 `data/assets/usd/.gitattributes`는 이 새 USDA 한 파일만 일반 텍스트로 관리한다. 원래 XML/USDA는 legacy 비교용으로 유지한다.

패치에는 상위 human-controller 저장소의 scripts, controller 구현, README, 설치 설정이나 ProtoMotions의 다른 기능 변경을 포함하지 않는다. 파일 경로와 변경 내용은 함께 배포하는 manifest에서 확인한다. 기존 상위 `human_model`의 bootstrap이 있는 브랜치에서는 중복 모델 연결을 건너뛰어 힘을 한 번만 계산한다.

```bash
# 적용할 controller 브랜치의 ProtoMotions에서 실행한다.
# human-model.patch를 먼저 준비한 뒤, 검사 성공 시에만 적용한다.
git apply --stat /tmp/human-model.patch
git apply --check /tmp/human-model.patch
git apply /tmp/human-model.patch
```

모델의 파라미터·수식과 69축 RoM은 `healthy_adult_v1`을 유지한다. 파일 추가와 연결 수정이 함께 적용되어야 하며, 별도의 모델 실행 명령은 필요 없다. 평소 사용하던 Python 환경이 이 ProtoMotions 체크아웃을 참조해야 한다. 적용 후 ProtoMotions의 변경을 커밋하고 상위 저장소에는 그 커밋을 기록한다. 다른 브랜치의 ProtoMotions 버전으로 교체하지 않는다.

현재 확인한 9개 controller 브랜치에 같은 패치를 적용·검사한다. 이후 연결부를 수정한 버전에서는 먼저 `git apply --check`를 실행하고, 실패하면 적용하지 않는다. 같은 패치를 두 번 적용하지 않는다. 되돌릴 때도 먼저 `git apply -R --check`로 확인한다.

## 라이선스와 검증 범위

### 2026-09-13 정량 검증 중인 선택 기능

기본 `healthy_adult_v1` profile은 유지한다. `PROTOMOTIONS_HUMAN_MODEL_FEATURES`의 쉼표 구분 값으로 `strength`(각도·속도별 능동 상한), `activation`(방향별 유효 활성화 지연), `passive_fit`(수동 에너지 진폭 적합), `strength_coupling`(무릎 각도에 따른 발목 저측굴곡 상한 보정)을 선택한다. `strength_coupling`은 `strength`가 필요하다. 모두 기본 비활성·미채택 candidate이며 새로운 구현 버전이 아니다.

활성화 상태는 물리 substep마다 갱신하고 환경 reset과 replay에서 복구한다. 근력 결합은 두 연구의 등척성 상대 토크로 식별한 가설이며 실제 GAS 근력을 재현한 모델이 아니다. 동적 적용의 분리 가능성은 미검증 가정으로 domain flag에 표시한다. 수동 적합 및 근력 후보 모두 외부 screening 기준 미달이 남아 있다.

`validation_contract.json`은 고정 평가 조건과 목표, `metrics.py`는 계산, `calibration.py`는 출처와 연구 분리를 보존하는 비교·적합, `validation.py`는 MuJoCo/Isaac Lab 계측과 `--summarize` 결과 목록을 제공한다. 소유 실험 `output/260913/e01_human_model_validation/`(상위 저장소 기준), 최신 판단은 `docs_ghlee/archive/experiments/environment_and_baselines/README.md`에 누적한다. Lab 물리 추적은 확보했으나 종료 대기 문제로 정상 종료와 해당 실행의 reset 검증은 미완료다.

ProtoMotions의 Apache-2.0 라이선스·기존 헤더를 유지하고 변경한 파일에는 변경 표시를 추가했다. 이 패키지는 자체 파라미터와 변환 코드를 제공하며 SMPL 원본 파라메트릭 모델이나 새 형상 자산을 배포하지 않는다. 기존 자산의 이용 조건은 그대로 적용된다. [ProtoMotions LICENSE](../../../LICENSE.md), [SMPL Model license](https://smpl.is.tue.mpg.de/modellicense.html), [SMPL-Body license](https://smpl.is.tue.mpg.de/license.html).

검증은 단위·방향별 상한, 수동 에너지의 음의 기울기, 감쇠의 소산, 극단 입력 유한성, 기존 형상·질량·관절 순서 보존, MJCF 제한, 저장된 USDA의 69축 RoM, USDA 원본 검사 및 직렬 hinge 파생 경로, 이전 체크포인트 설정 적용, MuJoCo substep 실행을 포함한다. 안정적인 전신 보행이나 생체실험과의 일치를 입증하는 검증은 아니다.

수동 임피던스 계측: `HumanJointModel.passive_impedance(q, qd)`는 `delta_tau=-K@delta_q-B*delta_qd`의 전체 결합 강성 K와 대각 감쇠 B를 반환한다. 수치 포화 이후의 접선과 비미분 경계 표시를 포함한다. 엔진 제한/접촉 반력, 제어기 PD, 능동 공동수축은 포함하지 않는다. 미구현 조직 항의 0을 실측 0으로 해석하지 않는다. 기존 토크 적용 법칙을 변경하지 않는 계측이며 candidate 검증용이다.

손목 감쇠 정의 후보: `wrist_damping`은 Nguyen2020 Methods/Figure2의 Hz 회귀 정의에 따라 VI를 `2*pi`로 나눠 양쪽 Wrist_x의 속도 감쇠계수로 사용한다(2.225→0.3541197484 Nms/rad). 논문의 표는 VI를 Nms/rad로 표기하지만 Figure2의 독립변수는 Hz다. 이 후보는 정의 변환 가설이며 비선형 주파수 응답·독립 생체 타당성까지 해결하지 않는다. 기본 비활성 candidate, ±10도·3~12Hz의 이완 손목 측정 조건 밖은 미검증이다. 기본 profile은 보존한다.

근력 출처 집단 선택: `HumanJointModel(..., features=("strength",), strength_cohort="male" 또는 "female")`로 Anderson 원문의 해당 곡선을 선택한다. 기본 `equal_sex`는 기존 남녀 곡선 평균을 유지한다. 선택에 맞춰 backend 능동 상한도 계산한다. 다른 관절 파라미터·형상·체중·관성은 변경하지 않으므로 전신 개인화가 아니다. calibration의 `--match-strength-cohort`는 명시적 단일 성별 표본 수만 사용하고 혼합/미상은 추정하지 않는다. candidate 평가용이며 기본 시뮬레이터 설정은 그대로다.

근력 참조 체격: 선택 인수 `strength_reference_size=(mass_kg,height_m)`는 Anderson의 체중×신장 토크 정규화에만 적용한다. 기본은 출처 집단의 원래 체격이다. 관절 ROM·수동 저항·몸체 질량·관성에는 적용하지 않으므로 전체 plant 개인화로 간주하지 않는다. calibration의 `--match-strength-size`는 명시적 단일 집단 cm/kg 평균만 읽고 범위·혼합·미상 값은 추정하지 않는다.

방향별 근력 보정: `active_strength_scale={"L_Ankle_y":[1.0,1.3]}`은 해당 DOF의 음/양 방향 능동 상한에 각각 적용한다. 기본은 모든 방향1이며 수동 저항·ROM·반대쪽 관절은 변경하지 않는다. 각도/속도 의존 상한, activation 출력과 backend ceiling에 함께 반영한다.0은 해당 능동 방향 비활성화이고 음수/비유한값/미등록 DOF는 거부한다. 보정 자료와 평가 자료의 분리 책임은 평가 절차에 있으며 실험 배율을 기본 프로필로 자동 채택하지 않는다. 현재는 클래스 API 및 SmplRobotConfig의 명시적 설정으로 선택하는 candidate 기능이며, 설정이 없으면 기본1을 사용한다.

시뮬레이터 연결: `SmplRobotConfig(human_model_parameters={"features":["strength"], "strength_cohort":"male", "strength_reference_size":[79,1.8], "active_strength_scale":{"L_Ankle_y":[1,1.75]}})`처럼 명시한다. 위 수치는 API 예시이며 채택된 대표 인체 파라미터가 아니다. metadata와 backend 힘 생성기가 같은 설정을 사용한다. 기존 체크포인트에 필드가 없으면 빈 설정을 사용한다. 기존 `PROTOMOTIONS_HUMAN_MODEL_FEATURES`가 명시돼 있으면 features에 우선한다. 미등록 설정 키는 거부하며 실험 산출물을 자동으로 읽지 않는다.

수동 수치 제한 진단: `passive_guard_flags(q,qd)`는 지수 에너지 성분별·선형 spring DOF별·감쇠 DOF별 포화를 반환한다. 계측 호출에만 계산하며 일반 힘 적용 루프에 추가 연산을 넣지 않는다. 비활성 에너지 성분은 힘 포화로 세지 않는다. 이 표시는 수치 보호장치의 사용 여부이며 생체 ROM/조직 한계 판정이 아니다.

근력 외삽 계측: `strength_caps(q, qd, directional_domain=True)`의 세 번째 반환값은 `[..., DOF, 2]`이며 마지막 축은 음/양 토크 방향이다. 기본 호출은 기존대로 두 방향의 OR를 반환한다. 해당 기능에서 확인하는 출처 영역의 초과 여부이며, false가 미구현 특성의 생체 타당성을 뜻하지 않는다. 무릎 의존 strength_coupling의 동적 가정 표시는 PF 방향에만 적용한다. 토크 수치·기본 후보 설정은 동일하다.

공동수축 입력 후보: torques(requested, q, qd, dt=..., coactivation=...)에서 activation 후보 또는 fatigue에 명시적으로 대응한 축에서만 [..., DOF] 형상의 0–1 값을 받는다. 포화된 순토크 요구를 먼저 배분하고 남은 음/양 방향 capacity의 최솟값에 coactivation을 곱한 동일 Nm를 양쪽 effort에 더한다. 같은 정규화 activation을 더하는 방식과 달리 비대칭 방향 근력에서도 고정 자세·정상상태 순토크를 유지한다. 기존 방향별 activation 상태에 rise/fall을 적용하므로 입력 제거·반전에는 과도응답이 있고 partial reset은 같은 상태를 초기화한다. 생체 근육 force/EMG가 아닌 joint-level 배분 가정이며 active K/B 법칙은 미구현이다. 입력 생략 시 기존 동작 유지; simulator apply 입력은 확장하지 않았고 API candidate만 추가했다. 새 버전·기본 물성 채택 없음.

피로 계산부 후보: dynamics.fatigue_compartment_step은 Frey-Law2012의 휴지/활성/피로3상태 ODE를 SSP RK2로 계산한다. target은0–1의 정규화 노력이며 F/R/L은초당 계수다. 양의 상태 보존을 위해 큰 dt를 나누고 부동소수점 질량 drift만 정규화한다. candidate_parameters.fatigue에 원문Table1 지역별F/R·L10을 보존했다. HumanJointModel의 선택 기능 fatigue로 토크·가용 상한·reset/replay에 연결했다. fatigue_regions={"R_Knee_y":"Knee"}처럼 지역 대응을 명시해야 하며 activation과 동시 사용은 거부한다. 음/양 방향의 독립 피로 저장소는 모델링 가정이다. strength_caps는 비피로 기준 상한, 물리 step 진단 상한은 피로 반영 값이다. 미대응 축은 기존 직접 토크 경로를 유지하며 기본값 변경은 없다. Hand/Grip을 손목이나 원문에 없는 부위에 자동 대응하지 않는다. 원문 검증은 지속 등척성 ET로 한정되며 force-time/간헐 수축/회복의 독립 검증은 남아 있다.

피로 원문 곡선 재현 감사는 기존 calibration 명령의 `--audit-fatigue-endurance --output <기존 실험 루트>`로 실행한다. scratch/frey_law2010_endurance_bioc.json을 요구하며 Knee10–90%9강도를 평가한다. 현재 RMSE25.9254s로 원문6.7s 재현 미충족; 해당 곡선은 피로 계수의 적합 자료이므로 독립 생체 검증으로 세지 않는다.

휴식 회복 후보: fatigue_rest_multiplier(기본1)는 fatigue 기능에서 target==0인 방향의 F→R 회복만 배가한다. 비기본값은 명시적으로 선택하며 Looft2018/2020의15/30은 검증 후보이지 전체관절 기본값이 아니다. simulator 설정과 replay 조건에 포함한다. 어깨 간헐 실측 참조를 확보했으나 MVC probe 배치·후기 표본 감소 조건 대응과 우리 실행 오차 평가는 남아 있다.
