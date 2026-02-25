# Drone 任務環境總覽（外層 `source` 版本）

本文件以目前外層程式碼為準：
- `source/Drone/Drone/tasks/direct/drone/__init__.py`
- `source/Drone/Drone/tasks/direct/drone/*.py`

重點整理：
1. 有哪些 Gym 環境（ID）。
2. 每個環境在做什麼。
3. 動作空間、觀察空間。
4. 獎勵函數組成。
5. 重生/終止規則與測試輸出。

---

## 1) 環境清單（註冊 ID）

### A. Basic 系列
- `Drone-Direct-Basic-v0`
- `Drone-Direct-Basic-Test-v0`

### B. Advanced 系列
- `Drone-Direct-Advanced-v0`
- `Drone-Direct-Advanced-Test-v0`
- `Drone-Direct-Advanced-Level0-Test-v0` ~ `Drone-Direct-Advanced-Level5-Test-v0`

### C. Target Touch（舊系列）
- `Drone-Direct-Target-Touch-v0`
- `Drone-Direct-Target-Touch-Test-v0`

### D. Target Touch Moving（舊系列移動目標）
- `Drone-Direct-Target-Touch-Moving-v0`
- `Drone-Direct-Target-Touch-Moving-Test-v0`
- `Drone-Direct-Target-Touch-Moving-Fast-v0`
- `Drone-Direct-Target-Touch-Moving-Fast-Test-v0`

### E. Vehicle Touch Stage（新系列）
- `Drone-Direct-Target-Touch-Vehicle-Stage0-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage0-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage1-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage1-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage2-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage2-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage3-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage3-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage4-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage4-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage5-v0`
- `Drone-Direct-Target-Touch-Vehicle-Stage5-Test-v0`

### F. Vehicle Moving（車輛風格移動目標）
- `Drone-Direct-Target-Touch-Vehicle-Faster-v0`
- `Drone-Direct-Target-Touch-Vehicle-Faster-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-VeryFast-v0`
- `Drone-Direct-Target-Touch-Vehicle-VeryFast-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-UltraFast-v0`
- `Drone-Direct-Target-Touch-Vehicle-UltraFast-Test-v0`

---

## 2) 動作空間（主要任務共通）

Target Touch / Moving / Vehicle 系列為 4 維動作：
- `a[0]`：總推力命令
- `a[1:4]`：三軸力矩命令

映射方式（`drone_env_target_touch.py`）：
- `actions` 先 clamp 到 `[-1, 1]`
- 推力：`thrust_z = thrust_to_weight * weight * (a0 + 1) / 2`
- 力矩：`moment_xyz = moment_scale * a[1:4]`

目前設定（Touch/Vehicle 基礎）：
- `thrust_to_weight = 4`
- `moment_scale = 0.05`

Basic/Advanced 也是 4 維控制，但 Basic 系列 `thrust_to_weight=2.5`。

---

## 3) 觀察空間

## 3.1 Basic（`observation_space=30`）
由下列組成：
- `root_lin_vel_b` (3)
- `root_ang_vel_b` (3)
- `projected_gravity_b` (3)
- `desired_pos_b` (3)
- `goal_dist` (1)
- `goal_dir` (3)
- `last_action` (4)
- 深度特徵（前後相機，各 5 維）共 10 維

合計：30。

## 3.2 Advanced（`observation_space=40`）
和 Basic 類似，但深度改四方向相機（前後左右），每個 5 維，共 20 維。

## 3.3 Target Touch（舊系列）
支援兩種觀察：
- 12 維（預設）：
  - `root_lin_vel_b(3) + root_ang_vel_b(3) + projected_gravity_b(3) + desired_pos_b(3)`
- 25 維（擴展）：
  - `root_pos + root_lin_vel_w + root_ang_vel_w + rot_mat(9) + desired_pos + last_action`
  - 目前實作中擴展版位置已用相對環境原點（`*_rel`）

## 3.4 Vehicle Stage0/Stage1
目前在 cfg 中明確使用 25 維擴展觀察。

---

## 4) 獎勵函數組成

## 4.1 Basic 獎勵（`drone_env_basic.py`）
主要組成：
- `r_progress = 4.0 * clamp(prev_dist - dist, -1, 1)`
- `r_time = -0.5 * dt`
- `r_hover`：接近目標時，低速懸停給分
- `r_ctrl = -0.05 * ||action||^2`

總和：`reward = r_progress + r_time + r_hover + r_ctrl`

## 4.2 Advanced 獎勵（`drone_env_advanced.py`）
在 Basic 基礎上更強化平穩與姿態：
- `r_progress`
- `r_time`
- `r_lin_vel`（線速度懲罰）
- `r_ang_vel`（角速度懲罰）
- `r_tilt`（傾斜懲罰）
- `r_ctrl`（控制成本）
- `r_hover`（近目標穩定懸停）

## 4.3 Target Touch / Moving / Vehicle（共用 `drone_env_target_touch.py`）

目前為「舊版 touch reward 組成」：
- `lin_vel`：線速度平方懲罰（可縮放）
- `ang_vel`：角速度平方懲罰（可縮放）
- `distance_to_goal`：`1 - tanh(distance / distance_to_goal_tanh_scale)`
- `touch_bonus`：碰觸成功一次性獎勵
- `touch_early_bonus`：越早碰觸額外加分（未設則為 0）
- `approach_reward`：朝目標方向速度正向獎勵（只取 `approach_speed>0`）
- `tcmd_penalty`：
  - `r_tcmd = λ4 * ||a_omega,t|| + λ5 * ||a_t-a_{t-1}||^2`
  - 實際以負號加入總獎勵（懲罰）
- `time_penalty`：每步固定扣分
- `near_touch_hover_penalty`：近目標但不碰且低速時懲罰（未設則可為 0）
- `distance_penalty`：`-distance_penalty_scale * distance * dt`
- `death_penalty`：死亡懲罰
- `tilt_forward_reward`：傾角軟引導（可關閉）
- `far_away_penalty`：太遠懲罰
- `failure_penalty`：timeout 且無觸碰、非 died/非 far_away 時懲罰

總獎勵為上述分量總和。

### 目前 Vehicle Base 參數（關鍵）
（見 `drone_env_target_touch_vehicle_cfg.py`）
- `lin_vel_reward_scale = 0.0`
- `ang_vel_reward_scale = 0.0`
- `distance_to_goal_reward_scale = 10.0`
- `distance_to_goal_tanh_scale` 預設沿 base（0.8），Stage1 明確也是 0.8
- `approach_reward_scale = 0.1`
- `tcmd_lambda_4 = 1e-3`
- `tcmd_lambda_5 = 1e-4`
- `touch_bonus_reward = 100.0`
- `time_penalty_scale = 0.15`
- `distance_penalty_scale = 0.1`
- `distance_penalty_only_when_not_approaching = True`
- `death_penalty = 100.0`
- `tilt_forward_reward_scale = 0.0`

---

## 5) 終止條件（Target Touch 系列）

`drone_env_target_touch.py::_get_dones`：
- `time_out`：episode 步數到上限
- `died`：
  - 高度低於 `died_height_threshold` 或
  - 接地判定（若啟用 ground contact）
- `far_away`：距離大於 `far_away_termination_distance`
- `touched`：距離小於觸碰門檻（`touch_radius + body_sphere + margin`）

`terminated = died | far_away | touched(可選)`。

---

## 6) 重生規則（Target Touch 系列）

`_reset_idx_impl` 主要流程：
1. 重置無人機狀態。
2. 依 `spawn_xy_min/max`, `spawn_z_min/max` 取樣無人機重生點。
3. 若設定 `target_spawn_distance_min/max`，目標以「無人機為圓心」取樣：
   - 先取 `theta ~ U(0, 2π)`
   - 再取半徑 `r ~ U(min, max)`
   - `target_xy = drone_xy + r * [cos(theta), sin(theta)]`
4. 目標高度 `z` 在指定範圍取樣。

---

## 7) Moving/Vehicle-Faster 系列的目標移動規則

`drone_env_target_touch_moving.py::_update_moving_targets`：
- 每步重算「目標遠離無人機」方向。
- 方向 z 分量乘 `moving_target_vertical_dir_scale`。
- 可選限制：
  - `moving_target_no_instant_reverse`
  - `moving_target_turn_rate_limit`
- 速度：`moving_target_speed`。
- 可加 z 正弦波：
  - `moving_target_z_wave_amplitude`
  - `moving_target_z_wave_period_s`
- 目標 z 超界會反射回 `moving_target_z_min/max`。

### 目前速度分級（重要）
- Moving: `1.0`
- Moving-Fast: `2.0`
- Vehicle-Faster: `5.0`
- Vehicle-VeryFast: `10.0`
- Vehicle-UltraFast: `15.0`

---

## 8) Vehicle Stage 距離配置（目前）

`drone_env_target_touch_vehicle_cfg.py`：
- Stage0: `target_spawn_distance = [6, 10]`, `far_away=15`
- Stage1: `target_spawn_distance = [15, 25]`, `far_away=30`
- Stage2: `target_spawn_distance = [20, 40]`, `far_away=60`
- Stage3: `target_spawn_distance = [20, 40]`, `far_away=60`
- Stage4: `target_spawn_distance = [20, 40]`, `far_away=60`
- Stage5: `target_spawn_distance = [25, 50]`, `far_away=70`

備註：Stage3 的 docstring 文字目前寫 `[15, 30]`，但實際數值是 `[20, 40]`。

---

## 9) Test 環境輸出與 TensorBoard 記錄

## 9.1 Test console 輸出（Touch/Moving）
- 成功數、成功率、平均成功步數
- 重生距離摘要 `rd=...`
- 重置原因摘要 `rsn=...`
- Touch test 目前也會印 `reward=mean(min,max)`

## 9.2 TensorBoard（Touch 系列）
- `Episode_Reward/*`：每回合分量平均後再除以 episode 秒數
- `Episode_RewardRaw/*`：每回合分量原始累積值
- `Episode_Termination/*`
- `Metrics/final_distance_to_goal`
- 事件率：
  - `Metrics/approaching_rate`
  - `Metrics/touched_rate`
  - `Metrics/far_away_rate`
  - `Metrics/died_rate`

---

## 10) 你常改的關鍵檔案（索引）

- 環境註冊：
  - `source/Drone/Drone/tasks/direct/drone/__init__.py`
- Touch 核心邏輯：
  - `source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py`
  - `source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_cfg.py`
- Vehicle stage 設定：
  - `source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_cfg.py`
- Moving 目標動態：
  - `source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_moving.py`
  - `source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_moving_cfg.py`
  - `source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_moving_fast_cfg.py`
  - `source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_moving_ladder_cfg.py`
- Agent 訓練參數：
  - `source/Drone/Drone/tasks/direct/drone/agents/*.yaml`

