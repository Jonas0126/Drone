# 環境建立步驟（example env 詳細範例說明）

這份文件是「教學版」範例說明，目標是讓第一次看的人也能知道：
1. 這一步在做什麼
2. 為什麼這一步重要
3. 這個範例實際怎麼做
4. 你要改參數時該從哪裡下手

本文件完全對照以下檔案：
- `drone_env_target_touch_vehicle_cfg.py`
- `drone_env_target_touch.py`
- `skrl_ppo_target_touch_vehicle_example_env_cfg.yaml`
- `drone_tasks_example_env_registration.py`

---

## Step 1. 設定機器人模型（模型、初始狀態、物理基礎）

### 這一步在做什麼
先決定「你要控制誰」以及「它一開始長什麼樣子」。

### 為什麼重要
模型如果沒有清楚定義，後面所有 action/reward 都會失真。尤其是飛行任務，質量與重力決定了推力尺度，這是能不能穩定飛的基礎。

### 這個範例實際做了什麼
- 機器人來源：`DRONE_CFG`，掛載到每個平行環境下的 `/World/envs/env_.*/Robot`。
  - 位置：`drone_env_target_touch_vehicle_cfg.py:77`
- 重生範圍：
  - XY 範圍 `[-5, 5]`
  - Z 範圍 `[1, 5]`
  - 位置：`drone_env_target_touch_vehicle_cfg.py:176-177,142-144`
- 在環境初始化時計算機體物理常數：
  - `_robot_mass`（質量）`drone_env_target_touch.py:85`
  - `_gravity_magnitude`（重力大小）`drone_env_target_touch.py:87`
  - `_robot_weight = mass * g`（重量）`drone_env_target_touch.py:89`

### 本步關鍵函式
- `DroneTargetTouchEnv.__init__`：建立動作/力矩緩衝，並計算質量/重力/重量。
  - 位置：`drone_env_target_touch.py:26`
- `_reset_idx_impl`：真正把重生位置與狀態寫回模擬器。
  - 位置：`drone_env_target_touch.py:426`

---

## Step 2. 設定動作空間（模型輸出定義）

### 這一步在做什麼
定義 policy 每一步會吐出什麼命令，總共有幾維。

### 為什麼重要
如果動作語意不明確，模型可能學到無效操作。動作範圍若不限制，數值容易爆掉。

### 這個範例實際做了什麼
- `action_space = 4`
  - 第 1 維：推力命令
  - 第 2~4 維：x/y/z 三軸力矩命令
  - 位置：`drone_env_target_touch_vehicle_cfg.py:32`
- 在進入物理前，先做 `[-1, 1]` 裁切：
  - `self._actions = actions.clone().clamp(-1.0, 1.0)`
  - 位置：`drone_env_target_touch.py:226`

### 本步關鍵函式
- `_pre_physics_step`：接收 action 並先標準化。
  - 位置：`drone_env_target_touch.py:220`

---

## Step 3. 設定 action 轉換與施力方式

### 這一步在做什麼
把神經網路輸出的標準化數字，轉成物理引擎可執行的推力和力矩。

### 為什麼重要
同一組 action，在不同轉換倍率下會是完全不同的控制強度。這步決定了可控性與訓練穩定度。

### 這個範例實際做了什麼
- 轉換倍率：
  - `thrust_to_weight = 4`（推力倍率）`drone_env_target_touch_vehicle_cfg.py:78`
  - `moment_scale = 0.05`（力矩倍率）`drone_env_target_touch_vehicle_cfg.py:79`
- 推力公式：
  - `thrust_z = thrust_to_weight * robot_weight * (a0 + 1) / 2`
  - 位置：`drone_env_target_touch.py:227`
- 力矩公式：
  - `moment_xyz = moment_scale * action[1:4]`
  - 位置：`drone_env_target_touch.py:228`
- 寫入模擬器：
  - `set_external_force_and_torque(...)`
  - 位置：`drone_env_target_touch.py:232`

### 本步關鍵函式
- `_pre_physics_step`：負責算出 `_thrust` 與 `_moment`。
- `_apply_action`：把 `_thrust` / `_moment` 套到機體。

---

## Step 4. 設定感測來源與感測相關參數

### 這一步在做什麼
決定環境用哪些感測資訊，並設定接觸判定相關閾值。

### 為什麼重要
感測資料是模型的「輸入世界」。資料不穩定或不一致，模型會學得很慢，甚至學不到。

### 這個範例實際做了什麼
- 使用狀態型觀測，不使用影像感測器。
- 接地判定參數：
  - `terminate_on_ground_contact = True` `drone_env_target_touch_vehicle_cfg.py:109`
  - `ground_contact_height_threshold = 0.2` `drone_env_target_touch_vehicle_cfg.py:110`
  - `ground_contact_body_margin = 0.10` `drone_env_target_touch_vehicle_cfg.py:111`
- 機體包覆球（touch/ground 代理）：
  - `drone_body_sphere_enabled = True` `drone_env_target_touch_vehicle_cfg.py:126`
  - `drone_body_sphere_radius = 0.2` `drone_env_target_touch_vehicle_cfg.py:127`
  - `drone_body_sphere_margin = 0.23` `drone_env_target_touch_vehicle_cfg.py:128`

### 本步關鍵函式
- `_resolve_drone_body_sphere_radius`：決定包覆球半徑。
  - 位置：`drone_env_target_touch.py:93`
- `_get_ground_contact_mask`：根據 root/body 高度判定接地。
  - 位置：`drone_env_target_touch.py:137`
- `_get_touch_threshold`：計算觸碰門檻。
  - 位置：`drone_env_target_touch.py:129`

---

## Step 5. 設定場景（地形、平行環境數、間距）

### 這一步在做什麼
建立訓練場景，決定一次並行訓練多少個環境。

### 為什麼重要
平行環境數直接影響資料吞吐量和訓練速度；間距與地形影響穩定性與碰撞干擾。

### 這個範例實際做了什麼
- 地形：`terrain_type = "plane"` `drone_env_target_touch_vehicle_cfg.py:59`
- 地板 prim：`/World/ground` `drone_env_target_touch_vehicle_cfg.py:58`
- 平行環境：
  - `num_envs = 8192`
  - `env_spacing = 8`
  - `replicate_physics=True`
  - `clone_in_fabric=True`
  - 位置：`drone_env_target_touch_vehicle_cfg.py:73`

### 本步關鍵函式
- `_setup_scene`：建立 robot、terrain、複製環境、加光源。
  - 位置：`drone_env_target_touch.py:203`

---

## Step 6. 設定地板材質與接觸物理

### 這一步在做什麼
設定摩擦、彈性、材質 combine 規則。

### 為什麼重要
地板材質會直接改變滑動、落地、反彈行為，對飛行與觸碰任務都很敏感。

### 這個範例實際做了什麼
- 全域材質：
  - `static_friction = 1.0`
  - `dynamic_friction = 1.0`
  - `restitution = 0.0`
  - `friction_combine_mode = multiply`
  - `restitution_combine_mode = multiply`
  - 位置：`drone_env_target_touch_vehicle_cfg.py:49-55`
- 地形材質：同樣設置
  - 位置：`drone_env_target_touch_vehicle_cfg.py:61-67`

---

## Step 7. 設定觀察空間（observation）

### 這一步在做什麼
把原始狀態資料整理成固定長度向量給 policy。

### 為什麼重要
輸入內容決定模型能不能判斷目前狀態；維度與語意要一致，模型才可穩定學習。

### 這個範例實際做了什麼
- 觀測模式：extended（25 維）
  - `observation_space = 25` `drone_env_target_touch_vehicle_cfg.py:35`
  - `use_extended_observation = True` `drone_env_target_touch_vehicle_cfg.py:37`
- 觀測組成（25 維）：
  - `root_pos_rel`（3）`drone_env_target_touch.py:238`
  - `root_lin_vel_w`（3）`drone_env_target_touch.py:243`
  - `root_ang_vel_w`（3）`drone_env_target_touch.py:244`
  - `rot_mat_wb`（9）`drone_env_target_touch.py:237`
  - `desired_pos_rel`（3）`drone_env_target_touch.py:239`
  - `self._actions`（4）`drone_env_target_touch.py:247`
- 回傳格式：`{"policy": obs}` `drone_env_target_touch.py:251`

### 本步關鍵函式
- `_get_observations`：組 observation。
  - 位置：`drone_env_target_touch.py:234`

---

## Step 8. 設定時間步（dt、decimation、episode 長度）

### 這一步在做什麼
決定模擬與控制的時間解析度，以及一回合有多長。

### 為什麼重要
時間步太粗會不穩，太細會很慢；episode 太短學不到，太長效率差。

### 這個範例實際做了什麼
- `sim.dt = 1/100` `drone_env_target_touch_vehicle_cfg.py:47`
- `decimation = 2` `drone_env_target_touch_vehicle_cfg.py:30`
- `episode_length_s = 30.0` `drone_env_target_touch_vehicle_cfg.py:28`
- `render_interval = decimation` `drone_env_target_touch_vehicle_cfg.py:48`

### 實際含意
- 物理更新約 100Hz
- 控制更新約 50Hz
- 每回合最長 30 秒

---

## Step 9. 設定獎勵函數（reward）

### 這一步在做什麼
定義策略該追求什麼、避免什麼。

### 為什麼重要
reward 是訓練方向盤。設計不當會學到奇怪策略（例如一直懸停但不碰目標）。

### 這個範例實際做了什麼
- reward 主函式：`_get_rewards` `drone_env_target_touch.py:269`
- example env 覆寫的重要權重：
  - `approach_reward_scale = 0.2` `cfg.py:180`
  - `distance_penalty_scale = 0.2` `cfg.py:181`
  - `tcmd_lambda_4 = 1e-2` `cfg.py:183`
  - `tcmd_lambda_5 = 1e-3` `cfg.py:184`
  - `ang_vel_reward_scale = -0.002` `cfg.py:186`
  - `tilt_forward_reward_scale = 0.05` `cfg.py:187`
  - `touch_bonus_reward = 200.0` `cfg.py:179`
  - `death_penalty = 200.0` `cfg.py:178`
  - `distance_to_goal_tanh_scale = 1.6` `cfg.py:195`
- base 仍生效：
  - `distance_to_goal_reward_scale = 10.0` `cfg.py:87`
  - `time_penalty_scale = 0.15` `cfg.py:97`
  - `distance_penalty_only_when_not_approaching = True` `cfg.py:101`

### `_get_rewards` 內你會看到的重點邏輯
- 距離 shaping：`1 - tanh(distance / scale)` `drone_env_target_touch.py:291-293`
- 朝目標前進獎勵：`drone_env_target_touch.py:299-302`
- 控制平滑懲罰 `tcmd`：`drone_env_target_touch.py:302-308`
- 事件懲罰（died/far/time_out/failure）：`drone_env_target_touch.py:342-350`
- 傾角引導：`drone_env_target_touch.py:351-378`
- reward 組件彙整：`drone_env_target_touch.py:380-401`

---

## Step 10. 設定 episode 終止機制（dones）

### 這一步在做什麼
定義何時結束回合（成功、失敗、超時）。

### 為什麼重要
done 條件就是任務邊界。邊界不清楚，策略會學到錯誤行為。

### 這個範例實際做了什麼
- 主要函式：`_get_dones` `drone_env_target_touch.py:409`
- 參數：
  - `terminate_on_touch = True` `cfg.py:130`
  - `died_height_threshold = 0.3` `cfg.py:108`
  - `far_away_termination_distance = 80.0` `cfg.py:194`
  - `terminate_on_ground_contact = True` `cfg.py:109`
- 邏輯：
  - `time_out`：`episode_length_buf >= max_episode_length - 1` `drone_env_target_touch.py:415`
  - `died`：高度過低或接地 `drone_env_target_touch.py:417-418`
  - `far_away`：距離過遠 `drone_env_target_touch.py:420`
  - `touched`：距離小於 touch threshold `drone_env_target_touch.py:421`
  - `terminated = died | far_away | touched` `drone_env_target_touch.py:423`

---

## Step 11. 設定 reset（終止後重置與隨機化）

### 這一步在做什麼
每回合結束後，重設機體、目標與統計，準備下一回合。

### 為什麼重要
reset 的分佈會決定訓練資料分佈。分佈太窄容易 overfit，太亂容易學不動。

### 這個範例實際做了什麼
- 核心函式：`_reset_idx_impl` `drone_env_target_touch.py:426`
- 訓練入口：`_reset_idx` `drone_env_target_touch.py:647`
- 測試入口：`DroneTargetTouchTestEnv._reset_idx` `drone_env_target_touch.py:677`

`_reset_idx_impl` 主要流程：
1. 寫回回合統計
- `Episode_Reward/*`, `Episode_RewardRaw/*` `drone_env_target_touch.py:457-461`
- `Episode_Termination/*`, `Metrics/*` `drone_env_target_touch.py:465-478`

2. 清空暫存
- `self._actions = 0`, `self._prev_actions = 0` `drone_env_target_touch.py:488-489`

3. 重生機體
- XY 抽樣 `uniform_(spawn_xy_min, spawn_xy_max)` `drone_env_target_touch.py:532-536`
- Z 抽樣 `uniform_(spawn_z_min, spawn_z_max)` `drone_env_target_touch.py:537-541`
- 寫回 root pose / vel / joint state `drone_env_target_touch.py:639-641`

4. 重生目標
- 若有距離限制，從機體位置為圓心在距離區間內抽樣
- 關鍵程式：`drone_env_target_touch.py:547-633`
- example env 設定：`target_spawn_distance_min=30.0`, `max=60.0` `cfg.py:192-193`

---

## Step 12. 設定訓練演算法與模型（skrl）

### 這一步在做什麼
設定學習器本身：演算法、網路、超參數與訓練長度。

### 為什麼重要
同一個環境，不同訓練器設定會有完全不同的收斂速度與穩定度。

### 這個範例實際做了什麼
檔案：`skrl_ppo_target_touch_vehicle_example_env_cfg.yaml`

1. 模型結構
- `policy`：GaussianMixin，MLP `[128, 128]`，`elu` `yaml:6-18`
- `value`：DeterministicMixin，MLP `[128, 128]`，`elu` `yaml:19-27`

2. PPO 參數
- `rollouts = 8` `yaml:41`
- `learning_epochs = 4` `yaml:42`
- `mini_batches = 16` `yaml:43`
- `discount_factor = 0.99` `yaml:44`
- `lambda = 0.95` `yaml:45`
- `learning_rate = 5.0e-04` `yaml:46`
- `grad_norm_clip = 1.0` `yaml:56`
- `ratio_clip = 0.2` `yaml:57`
- `value_clip = 0.2` `yaml:58`

3. 訓練長度與輸出
- `trainer.timesteps = 10000000` `yaml:76`
- `experiment.directory = "drone_target_touch_vehicle_example_env"` `yaml:66`
- `write_interval = 2000` `yaml:68`
- `checkpoint_interval = 2000` `yaml:69`

---

## Step 13. 註冊環境（最後一步）

### 這一步在做什麼
把你定義好的 env/cfg/agent_cfg 綁成可呼叫的環境 ID。

### 為什麼重要
不註冊就無法用 `--task <env_id>` 直接啟動訓練。

### 這個範例實際做了什麼
檔案：`drone_tasks_example_env_registration.py`

1. 訓練環境
- `id = "Drone-Direct-Target-Touch-Vehicle-Example-Env-v0"` `registration.py:9`
- `entry_point = DroneTargetTouchEnv` `registration.py:10`
- `env_cfg_entry_point = DroneTargetTouchVehicleExampleEnvCfg` `registration.py:13`
- `skrl_cfg_entry_point = skrl_ppo_target_touch_vehicle_example_env_cfg.yaml` `registration.py:14`

2. 測試環境
- `id = "Drone-Direct-Target-Touch-Vehicle-Example-Env-Test-v0"` `registration.py:20`
- `entry_point = DroneTargetTouchTestEnv` `registration.py:21`
- cfg 對應與訓練環境一致 `registration.py:24-25`

---

## 表格版索引（快速查找）

| Step | Function / 模組 | 參數或邏輯 | 值 | 位置 |
| --- | --- | --- | --- | --- |
| 1 模型 | `DroneTargetTouchVehicleExampleEnvCfg` | robot prim path | `/World/envs/env_.*/Robot` | `drone_env_target_touch_vehicle_cfg.py:77` |
| 1 初始狀態 | `_reset_idx_impl` | spawn XY / Z | `[-5,5] / [1,5]` | `drone_env_target_touch_vehicle_cfg.py:176-177,142-144` / `drone_env_target_touch.py:532-544` |
| 2 動作空間 | cfg + `_pre_physics_step` | `action_space`, clamp | `4`, `[-1,1]` | `drone_env_target_touch_vehicle_cfg.py:32` / `drone_env_target_touch.py:226` |
| 3 動作轉換 | `_pre_physics_step` / `_apply_action` | thrust / moment mapping | `tw=4`, `ms=0.05` | `drone_env_target_touch_vehicle_cfg.py:78-79` / `drone_env_target_touch.py:227-232` |
| 4 感測設定 | `_resolve_drone_body_sphere_radius`, `_get_ground_contact_mask` | ground/contact thresholds | `0.2`, `0.10` | `drone_env_target_touch_vehicle_cfg.py:110-111` / `drone_env_target_touch.py:137-183` |
| 5 場景 | `_setup_scene` | `num_envs`, `env_spacing`, terrain | `8192`, `8`, `plane` | `drone_env_target_touch_vehicle_cfg.py:57-74` / `drone_env_target_touch.py:203-218` |
| 6 地板材質 | cfg | friction / restitution | `1.0/1.0/0.0` | `drone_env_target_touch_vehicle_cfg.py:49-67` |
| 7 觀測空間 | `_get_observations` | observation mode / dim | `extended`, `25` | `drone_env_target_touch_vehicle_cfg.py:35-37` / `drone_env_target_touch.py:234-251` |
| 8 時間參數 | cfg | `dt`, `decimation`, `episode_length_s` | `1/100`, `2`, `30.0` | `drone_env_target_touch_vehicle_cfg.py:28-30,47` |
| 9 獎勵 | `_get_rewards` + cfg | approach/distance/tcmd/tilt | `0.2/0.2/1e-2,1e-3/0.05` | `drone_env_target_touch_vehicle_cfg.py:180-187` / `drone_env_target_touch.py:269-401` |
| 10 終止 | `_get_dones` | touch / died / far / timeout | `on`, `0.3`, `80.0`, `max_len` | `drone_env_target_touch_vehicle_cfg.py:108,130,194` / `drone_env_target_touch.py:409-424` |
| 11 重置 | `_reset_idx_impl` | target respawn distance | `[30,60]` | `drone_env_target_touch_vehicle_cfg.py:192-193` / `drone_env_target_touch.py:547-633` |
| 12 訓練 | skrl YAML | PPO / MLP / lr / timesteps | `PPO`, `[128,128]`, `5e-4`, `1e7` | `skrl_ppo_target_touch_vehicle_example_env_cfg.yaml:40-46,16-17,25-26,76` |
| 13 註冊 | `gym.register` | train/test env IDs | `...Example-Env-v0`, `...Test-v0` | `drone_tasks_example_env_registration.py:8-27` |

---

## 建議閱讀順序（給第一次接手的人）
- 先看 Step 2 + Step 3：知道模型輸出怎麼變控制命令。
- 再看 Step 7 + Step 9：知道模型看什麼、被鼓勵做什麼。
- 接著看 Step 10 + Step 11：知道什麼時候結束、怎麼重開。
- 最後看 Step 12 + Step 13：知道怎麼訓練、怎麼啟動。
