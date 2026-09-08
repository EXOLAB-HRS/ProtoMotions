# 건강한 성인 관절 모델 — healthy_adult_v1

이 디렉터리는 SMPL 강체 모델의 **관절 RoM, 방향별 능동 토크 한계, 수동 탄성·감쇠**를 관리한다. 기본 프로필은 `healthy_adult_v1.json`이다. 서로 다른 연구의 건강한 성인 측정 평균을 조합했으며, 근거가 없는 축은 가정 또는 미구현으로 명시했다. 하나의 모집단에서 측정한 전신 평균 데이터는 아니다.

기존 SMPL의 24개 강체, 69개 hinge DOF와 순서, 형상, 관성, 질량은 유지한다. MJCF 총질량은 약 63.314 kg이다. 따라서 **평균 성인의 체형·질량까지 새로 만든 모델은 아니다.** 토크를 체중에 비례해 임의로 보정하지 않았다.

## 파일과 실행

| 파일 | 역할 |
| --- | --- |
| `healthy_adult_v1.json` | 69개 축의 값, 단위, 방향, 근거 ID, 가정; 향후 모델을 추가할 위치 |
| `profile.py` | 프로필 로딩·검증 |
| `dynamics.py` | 방향별 능동 토크 제한과 별도의 수동 관절 힘 |
| `../../data/assets/usd/smpl_humanoid_healthy_adult_v1.usda` | **RoM을 직접 저장한 실제 모델 파일; IsaacLab에서 그대로 로딩** |
| `assets.py` | USDA의 RoM 일치 검사; MuJoCo/IsaacGym용 MJCF 캐시 생성 |
| `integration.py` | 새 설정 및 이전 체크포인트에 프로필 적용 |
| `tests/test_human_model.py` | 에너지·힘·축 방향·자산·실제 MuJoCo 경로 검증 |

이 브랜치의 코드를 적용하면 `SmplRobotConfig`의 기본 프로필이 자동으로 활성화된다. `resolved_configs_inference.pt`에 저장된 과거 설정도 시뮬레이터 생성 시 다시 적용한다. 정책 네트워크의 가중치나 입출력 순서를 변환하지 않는다. 기존 PD 이득은 능동 명령 계산에 유지한다. 생체 토크 제한을 적용하므로 과거 정책의 균형·보행 성능은 재평가해야 한다.

```bash
# ProtoMotions 디렉터리에서 기존 실행 명령을 그대로 사용
PROTOMOTIONS_HUMAN_MODEL=healthy_adult_v1 python ...

# 이전 모델과 비교할 때만 사용 (프로그램 시작 전에 지정)
PROTOMOTIONS_HUMAN_MODEL=legacy python ...

# CPU 검증; 프로젝트 의존성과 MuJoCo가 설치된 Python
python -m pytest -q protomotions/robot_configs/human_model/tests
```

IsaacLab은 [smpl_humanoid_healthy_adult_v1.usda](../../data/assets/usd/smpl_humanoid_healthy_adult_v1.usda)를 직접 읽는다. 69축의 `limit:rotX/Y/Z:physics:low/high`에 아래의 RoM을 **도(°) 단위로 저장**했다. 이 약 115 KB의 텍스트 파일은 일반 Git으로 관리하므로 이 파일을 위한 LFS 다운로드는 필요 없다. 기존 `smpl_humanoid.usda`는 legacy 모델용으로 유지한다.

실행 중 USDA를 생성하거나 RoM을 덮어쓰지 않는다. RoM을 변경할 때는 이 USDA와 JSON의 `rom_deg`를 함께 수정한다. JSON은 행동 범위와 MJCF 백엔드에 사용하며, IsaacLab 시작 시 USDA와 일치하는지 검사한다. MuJoCo/IsaacGym은 같은 JSON에서 만든 MJCF를 `/tmp/protomotions-human-model-<uid>/<content-hash>/`에 캐시한다. MJCF 캐시 위치는 `PROTOMOTIONS_HUMAN_MODEL_CACHE`로 지정할 수 있다.

연결한 엔진은 MuJoCo, IsaacLab, IsaacGym이다. Genesis/Newton은 검증되지 않은 적용을 막기 위해 오류를 낸다. **CPU MuJoCo 실행을 검증했으며 GPU IsaacLab/IsaacGym 동역학 검증은 남아 있다.**

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

PD 명령의 `Kp*(target-q)-Kd*qdot`는 능동 요청이다. 원래 XML의 stiffness 300–1000, damping 30–100을 인체 수동 특성으로 해석하지 않는다. USDA에서 바꾼 물리 속성은 **RoM뿐**이다. IsaacLab의 actuator 설정이 실행 시 drive 강성·감쇠를 0으로 지정하고, 최대 토크 제한과 passive 항은 코드에서 적용한다. MuJoCo/IsaacGym용 파생 MJCF도 기존 spring/damping을 0으로 한다. 수동 항은 매 물리 step에서 한 번만 더한다. 액추에이터는 이 합을 받을 수 있는 충분한 한계를 사용하고, 생체 최대 능동 토크는 합산 **전**에 방향별로 제한한다. 수동 토크나 관절 제한의 반력은 근육 최대 토크와 다른 양이다.

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

## 변경 범위와 병합

정책·controller·학습 데이터 파일은 수정하지 않는다. 기존 연결부 수정은 `robot_configs/smpl.py`와 모델 로딩·힘 적용을 위한 simulator 3곳(base, IsaacLab scene, MuJoCo)다. 모델 패키지와 이 연결부가 함께 있어야 수동 특성이 실제 시뮬레이션에 작용한다. 별도의 건강한 성인 USDA를 추가하고, `.gitattributes`에서 그 파일만 일반 텍스트로 관리한다. 원래 XML/USDA는 legacy 비교용으로 유지한다.

상위 저장소와 ProtoMotions는 **별도 Git 저장소**다. 일반 `git merge`로 두 저장소의 변경을 병합한 뒤 상위 gitlink를 기록한다. 별도 Python 병합 도구는 사용하지 않는다. 명령 순서는 [human_model 브랜치 README](https://github.com/EXOLAB-HRS/human-controller/blob/human_model/README.md#다른-브랜치에-병합)에 있다. 대상 README와 다른 기능을 유지하며 다른 브랜치 ref는 변경하지 않는다.

기존 파일을 이 브랜치에서 삭제하면 삭제도 merge된다. 따라서 Git에는 공통 기준의 기존 파일을 남기고, 변경분만 모델에 한정한다. 별도 모델 보기 폴더나 sparse checkout은 작업 화면을 줄이는 수단이며 삭제 commit과 다르다.

## 라이선스와 검증 범위

ProtoMotions의 Apache-2.0 라이선스·기존 헤더를 유지하고 변경한 파일에는 변경 표시를 추가했다. 이 패키지는 자체 파라미터와 변환 코드를 제공하며 SMPL 원본 파라메트릭 모델이나 새 형상 자산을 배포하지 않는다. 기존 자산의 이용 조건은 그대로 적용된다. [ProtoMotions LICENSE](../../../LICENSE.md), [SMPL Model license](https://smpl.is.tue.mpg.de/modellicense.html), [SMPL-Body license](https://smpl.is.tue.mpg.de/license.html).

검증은 단위·방향별 상한, 수동 에너지의 음의 기울기, 감쇠의 소산, 극단 입력 유한성, 기존 형상·질량·관절 순서 보존, MJCF 제한, 저장된 USDA의 69축 RoM, USDA 직접 로딩 경로, 이전 체크포인트 설정 적용, MuJoCo substep 실행을 포함한다. 안정적인 전신 보행이나 생체실험과의 일치를 입증하는 검증은 아니다.
