# Drone 直連任務環境總覽（中文）

本文件整理 `Drone/source/Drone/Drone/tasks/direct/drone` 目前各系列環境設定，方便快速查閱每個環境在做什麼、用了哪些關鍵參數。

## 1. 檔案與責任分工

- `drone_env_basic_cfg.py`：Basic 系列設定（場景、相機、獎勵、課程）
- `drone_env_advanced_cfg.py`：Advanced 系列設定（多障礙等級）
- `drone_env_target_touch_cfg.py`：Target Touch 舊系列基底設定
- `drone_env_target_touch_vehicle_cfg.py`：Vehicle 新系列 Stage0~Stage5 設定
- `drone_env_target_touch_moving_cfg.py`：Moving 系列基底設定
- `drone_env_target_touch_moving_fast_cfg.py`：Moving Fast 設定
- `drone_env_target_touch_moving_ladder_cfg.py`：Moving 速度階梯（Faster/VeryFast/UltraFast）
- `drone_env_target_touch.py`：Touch 核心環境邏輯（觀測、獎勵、終止、重置）
- `drone_env_target_touch_moving.py`：Moving 目標動態邏輯（其餘沿用 touch）
- `__init__.py`：Gym 環境註冊與 env id 對應

## 2. 系列與環境 ID 對應

### 2.1 Basic

- `Drone-Direct-Basic-v0`
- `Drone-Direct-Basic-Test-v0`

對應設定：`drone_env_basic_cfg.py:DroneEnvCfg`

### 2.2 Advanced

- `Drone-Direct-Advanced-v0`
- `Drone-Direct-Advanced-Test-v0`
- `Drone-Direct-Advanced-Level{0..5}-Test-v0`

對應設定：`drone_env_advanced_cfg.py:DroneTrainEnvCfg / DroneTrainLevel{0..5}EnvCfg`

### 2.3 Target Touch（舊系列）

- `Drone-Direct-Target-Touch-v0`
- `Drone-Direct-Target-Touch-Test-v0`

對應設定：`drone_env_target_touch_cfg.py:DroneTargetTouchEnvCfg`

### 2.4 Moving（舊系列延伸）

- `Drone-Direct-Target-Touch-Moving-v0`
- `Drone-Direct-Target-Touch-Moving-Test-v0`
- `Drone-Direct-Target-Touch-Moving-Fast-v0`
- `Drone-Direct-Target-Touch-Moving-Fast-Test-v0`

對應設定：
- `drone_env_target_touch_moving_cfg.py:DroneTargetTouchMovingEnvCfg`
- `drone_env_target_touch_moving_fast_cfg.py:DroneTargetTouchMovingFastEnvCfg`

### 2.5 Vehicle（新系列）

- `Drone-Direct-Target-Touch-Vehicle-Stage0-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage1-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage2-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage3-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage4-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage5-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Faster-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-VeryFast-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-UltraFast-v0` / `-Test-v0`

對應設定：
- Stage0~5：`drone_env_target_touch_vehicle_cfg.py`
- Faster 以上：`drone_env_target_touch_moving_ladder_cfg.py`

## 3. 動作空間與觀測空間

## 3.1 動作空間（Touch/Moving/Vehicle）

- 維度：4
- 結構：`[thrust_cmd, moment_x, moment_y, moment_z]`
- 映射邏輯（`drone_env_target_touch.py::_pre_physics_step`）：
  - 推力：由 `thrust_to_weight`、機體重量、`thrust_cmd` 映射成 z 推力
  - 力矩：`moment_scale * actions[:, 1:]`

## 3.2 觀測空間（Touch/Moving/Vehicle）

- 12 維（基礎）：
  - `root_lin_vel_b(3), root_ang_vel_b(3), projected_gravity_b(3), desired_pos_b(3)`
- 25 維（擴展）：
  - `root_pos_rel(3), root_lin_vel_w(3), root_ang_vel_w(3), rot_mat(9), desired_pos_rel(3), last_action(4)`

`Vehicle` 系列預設使用 25 維（`use_extended_observation=True`）。

## 4. Touch 核心 reward 組成（`drone_env_target_touch.py::_get_rewards`）

每步總獎勵由多項相加（`timeout_penalty` 只記錄，不計入總和）：

- `lin_vel`：線速度懲罰
- `ang_vel`：角速度懲罰
- `distance_to_goal`：`(1 - tanh(distance / distance_to_goal_tanh_scale)) * distance_to_goal_reward_scale`
- `approach_reward`：朝目標前進速度正獎勵
- `tcmd_penalty`：控制命令懲罰（角速度命令 + 動作變化量）
- `time_penalty`：每步固定扣分
- `distance_penalty`：距離懲罰（可選僅在未接近時生效）
- `near_touch_hover_penalty`：近目標低速懸停但不碰觸懲罰
- `touch_bonus`：碰觸一次性獎勵
- `touch_early_bonus`：早碰觸額外獎勵
- `death_penalty` / `far_away_penalty` / `failure_penalty`：終止事件懲罰
- `tilt_forward_reward`：前進時傾角軟式 shaping（可關）

## 5. 重置與終止規則（Touch/Vehicle）

終止條件（`_get_dones`）：

- `died`：高度過低或接地
- `far_away`：超過 `far_away_termination_distance`
- `touched`：距離小於 touch threshold（若 `terminate_on_touch=True`）
- `time_out`：步數達回合上限

重置流程（`_reset_idx_impl`）：

- 重置機體姿態/速度/動作緩衝
- 重生無人機位置（`spawn_xy_*`, `spawn_z_*`）
- 重生目標位置：
  - 若設定 `target_spawn_distance_min/max`：以無人機為圓心在距離區間取樣
  - 否則使用預設平面區域取樣
- 更新統計：
  - `Episode_Reward/*`、`Episode_RewardRaw/*`
  - `Episode_Termination/*`
  - `Metrics/approaching_rate / touched_rate / far_away_rate / died_rate`

## 6. 各系列重點設定差異

### 6.1 Touch（舊系列）

- 支援 12/25 維觀測切換
- 支援內部 curriculum（spawn / target distance）
- 主要用於靜態目標碰觸訓練

### 6.2 Moving

- reward 沿用 touch
- 只改目標動態：目標每步移動、可限制轉向與禁止瞬間反向

### 6.3 Vehicle（Stage0~5）

- 固定為 25 維觀測
- 以「外部階段」方式切分難度（每個 stage 一個獨立 env）
- 目前 Stage0~Stage5 已統一採用 Stage1 參數模板
- 各 stage **唯一差異**：目標重生距離區間（`target_spawn_distance_min/max`）

目前距離區間：

- Stage0：15 ~ 25 m
- Stage1：30 ~ 60 m
- Stage2：20 ~ 40 m
- Stage3：20 ~ 40 m
- Stage4：20 ~ 40 m
- Stage5：25 ~ 50 m

## 7. 測試環境輸出

- `DroneTargetTouchTestEnv` / `DroneTargetTouchMovingTestEnv` 會在 reset 時輸出：
  - 成功率
  - 平均步數
  - 死亡原因縮寫（如 `dh/dg/dt/fa/to/fn`）
  - 重生距離摘要（`rd=mean(min,max)`）
  - reward 摘要（Touch test）

---

若要查單一參數如何影響邏輯，優先看：

- 設定值在哪裡定義：`*_cfg.py`
- 參數怎麼被使用：`drone_env_target_touch.py` 與 `drone_env_target_touch_moving.py`
