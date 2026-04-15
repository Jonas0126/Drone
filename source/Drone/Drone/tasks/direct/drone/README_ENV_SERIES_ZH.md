# Drone 直連任務環境總覽（中文）

本文件整理 `Drone/source/Drone/Drone/tasks/direct/drone` 目前各系列環境設定，方便快速查閱每個環境在做什麼、用了哪些關鍵參數。

## 1. 檔案與責任分工

目前已改成「系列內聚」結構：

- `target_touch/`
  - `env.py`：Target Touch 主環境
  - `base_cfg.py`：Touch 共同基底設定
  - `cfg.py`：相容性 facade，保留舊 import 路徑
  - `reset_ops.py` / `observations.py` / `rewards.py` / `terminations.py`
  - `scene_ops.py` / `debug_vis.py`
- `target_touch_vehicle/`
  - `env.py`：Vehicle 主環境
  - `base_cfg.py`：Vehicle 共同基底設定
  - `stage_cfgs.py`：Vehicle Stage0~Stage5 設定
  - `curriculum.py`：Vehicle 專用課程與 reset hook
- `target_touch_vehicle_moving/`
  - `env.py`：Vehicle-Moving 主環境
  - `base_cfg.py`：Vehicle-Moving 共同基底設定
  - `stage_cfgs.py`：Vehicle-Moving Stage0~Stage5 / Test 設定
  - `demo_cfgs.py`：台北 demo 設定
  - `moving_target.py`：目標移動主流程
  - `motion_sampling.py`：速度 / heading / turn 段長取樣
  - `obstacle_ops.py`：場景障礙物避讓 / pushout
  - `reset_ops.py`：moving reset 與 distance curriculum
  - `test_mode.py`：測試模式 facade
  - `test_visuals.py`：trail / debug 可視化
  - `test_stats.py`：測試 done 與統計列印

舊路徑仍保留為 facade，相容現有匯入與 Gym 註冊：

- `drone_env_target_touch.py`
- `drone_env_target_touch_cfg.py`
- `drone_env_target_touch_vehicle.py`
- `drone_env_target_touch_vehicle_cfg.py`
- `drone_env_target_touch_vehicle_moving.py`
- `drone_env_target_touch_vehicle_moving_cfg.py`

其他尚未重構成系列目錄的檔案：

- `drone_env_basic_cfg.py`：Basic 系列設定（場景、相機、獎勵、課程）
- `drone_env_advanced_cfg.py`：Advanced 系列設定（多障礙等級）
- `drone_env_target_touch_moving_cfg.py`：Moving 系列基底設定
- `drone_env_target_touch_moving_fast_cfg.py`：Moving Fast 設定
- `drone_env_target_touch_moving_ladder_cfg.py`：Moving 速度階梯（Faster/VeryFast/UltraFast）
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

對應設定：
- Stage0~2：`drone_env_target_touch_vehicle_cfg.py`

### 2.6 Vehicle-Moving（新增系列，目標會移動）

- `Drone-Direct-Target-Touch-Vehicle-Moving-Pre-v0` / `-Test-v0`（Pre-Moving：靜態目標，對齊 Vehicle Stage0）
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage0-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage1-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage2-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage3-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage4-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage4-Taipei-Demo-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage5-v0` / `-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage5-Taipei-Demo-Test-v0`

對應設定：
- Stage0~5 參數：`drone_env_target_touch_vehicle_moving_cfg.py`
- 環境入口：`drone_env_target_touch_vehicle_moving.py`

說明：
- `Vehicle-Moving` 是新增系列，不覆蓋原本 `Vehicle`。
- `Vehicle-Moving` 使用 moving target 邏輯，目標每個 step 都會移動。
- `Pre` 階段使用 `Vehicle Stage0` 同款環境設定（靜態目標），但 agent 模型維度使用 `Vehicle-Moving` 的 `[192, 96]`。
- `Vehicle-Moving` 的 `-Test-v0`：
  - Stage0/Stage1/Stage2 使用 `episode_length_s=120.0`（2 分鐘）
  - Stage3/Stage4/Stage5 使用 `episode_length_s=180.0`（與訓練一致）
- `Stage4-Taipei-Demo-Test`：
  - 沿用 Stage4-Test 的 reward / 目標速度 / 測試可視化
  - 額外載入 `assets/Taipei_demo_001.usd` 城鎮背景
  - 預設只開 1 個 env，並把無人機與目標抬高到城鎮上空飛行
- `Stage5-Taipei-Demo-Test`：
  - 沿用 Stage5-Test 的 reward / 目標移動規則 / 測試可視化
  - 額外載入 `assets/Taipei_demo_001.usd` 城鎮背景
  - 預設只開 1 個 env，並把無人機與目標抬高到城鎮上空飛行

## 3. 動作空間與觀測空間

## 3.1 動作空間（Touch/Moving/Vehicle/Vehicle-Moving）

- 維度：4
- 結構：`[thrust_cmd, moment_x, moment_y, moment_z]`
- 映射邏輯（`drone_env_target_touch.py::_pre_physics_step`）：
  - 推力：由 `thrust_to_weight`、機體重量、`thrust_cmd` 映射成 z 推力
  - 力矩：`moment_scale * actions[:, 1:]`

## 3.2 觀測空間（Touch/Moving/Vehicle/Vehicle-Moving）

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

## 5. 重置與終止規則（Touch/Vehicle/Vehicle-Moving）

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
- 目前 Stage0~Stage2 已統一採用 Stage1 參數模板
- 各 stage 主要差異：目標重生距離區間（`target_spawn_distance_min/max`）
- 例外：Stage0 目前額外啟用姿態控制強化
  - `ang_vel_reward_scale = -0.002`
  - `tilt_forward_reward_scale = 0.05`
  - `tcmd_lambda_4 = 1e-2`
  - `tcmd_lambda_5 = 1e-3`

目前距離區間：

- Stage0：10 ~ 40 m（已固定，不啟用 Stage0 內部 target-distance 課程）
- Stage1：20 ~ 50 m（對齊 2026-03-03_15-33-30 配置：`approach_reward_scale=0.1`、`time_penalty_scale=0.20`、`distance_to_goal_tanh_scale=3.2`、`death_penalty=300`、`failure_penalty=300`、`tcmd_lambda_4=5e-3`、`tcmd_lambda_5=5e-4`）
- Stage2：20 ~ 50 m（速度/通過率優化：`approach_reward_scale=0.1`、`time_penalty_scale=0.20`、`distance_to_goal_tanh_scale=28`、`death_penalty=300`、`failure_penalty=300`、`tcmd_lambda_4=5e-3`、`tcmd_lambda_5=5e-4`）
- Stage3（Moving）：目標速度 `moving_target_speed=7.0`

### 6.4 Vehicle-Moving（Stage0~5）

- reward 與終止邏輯沿用 `drone_env_target_touch.py`（不另寫第二套 reward）
- 目標動態沿用 `drone_env_target_touch_moving.py`：每步更新目標位置
- 新系列專用入口：`drone_env_target_touch_vehicle_moving.py`
- 新系列專用參數檔：`drone_env_target_touch_vehicle_moving_cfg.py`

目前目標移動參數（Base）：

- `moving_target_speed=1.0`（作為未覆寫 stage 的預設值）
- `moving_target_vertical_dir_scale=0.6`
- `moving_target_turn_rate_limit=0.0`
- `moving_target_no_instant_reverse=False`
- `moving_target_z_wave_amplitude=0.2`
- `moving_target_z_wave_period_s=8.0`
- `moving_target_z_min=0.5`, `moving_target_z_max=5.0`

目前距離區間：

- Stage0：5 ~ 15 m
- Stage1：10 ~ 20 m
- Stage2：15 ~ 25 m
- Stage3：20 ~ 60 m
- Stage4：20 ~ 60 m（沿用 Stage3）
- Stage5：20 ~ 80 m（Stage5 擴展長距離）

目前目標速度：

- Stage0：`moving_target_speed=3.0`
- Stage1：`moving_target_speed=3.0`
- Stage2：`moving_target_speed=5.0`
- Stage3：`moving_target_speed=3.0`
- Stage4：`moving_target_speed=6.0`
- Stage5：`moving_target_speed=3.0~12.0`（區間抽樣）

Stage4 台北展示環境（`Stage4-Taipei-Demo-Test`）：

- 場景背景：`assets/Taipei_demo_001.usd`
- 地面/碰撞地形：仍是程式建立的 `plane`（`TerrainImporterCfg(terrain_type="plane")`），不是來自某個 terrain USD
- 預設場景數：`num_envs=1`
- 城市資產會先平移 `(-58800.16739, -117400.47675, 0)`，把 101 周邊拉回世界原點附近，避免大世界座標造成 PhysX / 渲染精度問題
- 主要展示錨點：固定使用平移後的原點 `(0, 0)`
- 無人機會生成在指定矩形區域內：原始 `Taipei_demo_001` 座標約 `x=58059~58660`、`y=117575~118393`
- 由於城市資產已平移到原點附近，目前世界座標對應約為 `x=-741.16739~-140.16739`、`y=174.52325~992.52325`
- 雖然畫面上會飛到較大範圍，但 extended observation 的 `x/y` 會改用「相對本回合重生點」的局部座標，避免 Stage4 policy 直接看到大尺度絕對位置
- 若目標抽樣落到安全圈內，會被往外推，避免飛到 101 核心區域內
- 邏輯高度仍沿用 Stage4：無人機 `1~5 m`、目標 `0.5~5 m`
- 畫面顯示高度：整體額外上抬 `55 m`，讓飛行看起來在城市上空，同時保持建物細節可見
- 目標高度夾限（邏輯值）：`0.5~5 m`
- 觀測中的 `x/y` 也會扣掉 101 錨點，避免城市大座標直接進模型
- 地圖內建燈光 graph 在 referenced 子場景時不完整，因此展示版改為手動補上與 `Taipei_demo_001.usd` 一致的雙燈配置：
- `DistantLight`：`/Environment/defaultLight`，`intensity=1000`、`exposure=1`、`angle=1.0`、`normalize=True`、`color=(1,1,1)`、旋轉 `(45, 0, 90)`
- `DomeLight`：`/Environment/DomeLight`，`intensity=500`、`exposure=0`、`color=(1,1,1)`、`texture=SubUSDs/textures/Sky_horiz_6-2048.jpg`、`texture_format=latlong`、旋轉 `(-270, 0, 270)`
- 啟用建物 bbox 重生清理：目標重生後若落在建物 bbox 內，會立刻被推出 bbox 外，避免一出生就在建物內
- 額外開啟無人機半透明亮色 outline 外殼，並把移動軌跡球放大到 `0.065m`，讓遠景錄影時更容易辨識本體與路徑
- 其餘 reward、終止與移動速度維持 Stage4-Test，不另外改 policy 設定

Stage5 台北展示環境（`Stage5-Taipei-Demo-Test`）：

- 場景背景、地面/碰撞地形、城市平移、燈光與視覺高度偏移都和 `Stage4-Taipei-Demo-Test` 相同
- 主要展示錨點：固定使用平移後的原點 `(0, 0)`
- 無人機會生成在指定矩形區域內：原始 `Taipei_demo_001` 座標約 `x=58059~58660`、`y=117575~118393`
- 由於城市資產已平移到原點附近，目前世界座標對應約為 `x=-741.16739~-140.16739`、`y=174.52325~992.52325`
- 畫面顯示高度：整體額外上抬 `100 m`
- extended observation 的 `x/y` 仍使用「相對本回合重生點」的局部座標，避免台北展示場景的大尺度絕對位置直接進模型
- reward、終止與目標移動規則改為完全沿用 Stage5：
  - 長距離目標距離範圍 `20~80m`
  - 基準速度 `3~12m/s`
  - `road_like` 車輛式平滑轉彎
  - Stage5 的中等版穩姿 reward 係數
- 同樣啟用建物 bbox 重生清理：目標重生後若落在建物 bbox 內，會立刻被推出 bbox 外，避免一出生就在建物內
- 額外開啟無人機半透明亮色 outline 外殼，並把移動軌跡球放大到 `0.065m`，讓遠景錄影時更容易辨識本體與路徑

Stage5 目標移動規則（車輛式）：

- 改用 `road_like`：多數時間直行，偶爾進入連續平滑轉彎段
- 每 4~8 秒決策一次（直行或轉彎），轉彎段角速度約 30~50 度/秒
- 若抽到轉彎，轉彎段持續時間約 1.5~3 秒（短於直行段）
- 直行/轉彎機率約 60% / 40%
- 每回合先固定抽樣一個基準速度（`3~12 m/s`）
- 直行使用基準速度，轉彎時降到基準速度的約 70%
- 關閉 Z 軸波浪擾動，主要在平面移動
- 為減少螺旋式前進，Stage5 重新啟用少量姿態穩定項：
  - `lin_vel_reward_scale = -0.002`
  - `ang_vel_reward_scale = -0.006`
  - `tcmd_lambda_4/5 = 4e-3 / 4e-4`
  - `tilt_forward_reward_scale = 0.03`

Stage5-Test 測試覆寫參數（透過 `play.py`）：

- `--test_fixed_distance <m>`：固定目標重生距離
- `--test_fixed_speed <m/s>`：固定 moving target 速度
- `--test_num_episodes <N>`：跑滿指定 episode 數後自動停止

目前時間懲罰設定：

- Stage0（含 Base）：`time_penalty_scale=0.15`
- Stage1：`time_penalty_scale=0.20`
- Stage2：`time_penalty_scale=0.20`
- Stage3：`time_penalty_scale=0.20`
- Stage4：`time_penalty_scale=0.20`
- Stage5：`time_penalty_scale=0.20`

目前關鍵 reward 係數（Vehicle-Moving）：

- `distance_to_goal_reward_scale=14.0`（提高距離 shaping 權重）
- `distance_to_goal_tanh_scale`：
  - Stage0：`2.4`
  - Stage1：`3.2`
  - Stage2：`3.2`（已對齊 Stage1）
  - Stage3：`3.2`
- `distance_penalty_scale=0.0`（已關閉，改用 progress 項）
- `progress_reward_scale=1.0`（每步距離縮短量獎勵）
- `touch_early_bonus_scale=200.0`（越早 touch 額外獎勵越高）
- `approach_reward_scale`：
  - Stage0：`0.1`
  - Stage1：`0.05`
  - Stage2：`0.05`
  - Stage3：`0.05`
- `tcmd_lambda_4/5`：
  - Stage1：`2e-3 / 2e-4`
  - Stage2：`2e-3 / 2e-4`（reward 係數已對齊 Stage1）
  - Stage3：`2e-3 / 2e-4`

Stage1（訓練）額外設定：

- `episode_length_s=120.0`（2 分鐘）
- `far_away_termination_distance=100.0`
- `tcmd_lambda_4=2e-3`, `tcmd_lambda_5=2e-4`（控制命令懲罰下修）

## 7. 測試環境輸出

- `DroneTargetTouchTestEnv` / `DroneTargetTouchMovingTestEnv` 會在 reset 時輸出：
  - 成功率
  - 平均步數
  - 死亡原因縮寫（如 `dh/dg/dt/fa/to/fn`）
  - 起始距離摘要（`start_dist=mean(min,max)`；重置後的目標-無人機距離）
  - 重生距離摘要（`rd=mean(min,max)`）
  - reward 摘要（Touch test）

---

若要查單一參數如何影響邏輯，優先看：

- 設定值在哪裡定義：`*_cfg.py`
- 參數怎麼被使用：`drone_env_target_touch.py` 與 `drone_env_target_touch_moving.py`
