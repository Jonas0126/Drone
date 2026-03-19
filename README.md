# Drone 專案總覽

這份文件是給第一次接手這個專案的人看的。目的不是列出所有細節，而是讓你先快速理解：

- 這個專案在做什麼
- 目前主要在訓練哪一類任務
- 專案裡哪些檔案是「設定」、哪些是「邏輯」、哪些是「歷史紀錄」
- 想查環境、獎勵、模型、訓練指令時，該去哪裡找

---

## 1. 這個專案在做什麼

這個專案是在 Isaac Lab / skrl 上訓練四軸無人機（drone）的強化學習任務。  
目前主要目標是讓無人機在 3D 空間中追逐並碰觸一個移動目標，逐步從較簡單條件訓練到較困難條件。

目前你最常會碰到的是這條主線：

1. `Vehicle-Moving Pre`
2. `Vehicle-Moving Stage0`
3. `Vehicle-Moving Stage1`
4. `Vehicle-Moving Stage2`
5. `Vehicle-Moving Stage3`

這條主線的設計方向是：

- 目標物會移動
- 不同 stage 會調整目標重生距離、目標速度、回合長度、終止條件
- 用前一階段訓練好的 checkpoint 繼續 fine-tune 下一階段

簡單說，這不是一個單一環境，而是一組循序漸進的訓練環境。

---

## 2. 這個 repo 的結構怎麼看

這個專案有兩層：

- 外層：`/home/jonas/Drone`
- 內層：`/home/jonas/Drone/Drone`

實務上可以這樣理解：

- 外層 repo：偏說明、工作紀錄、操作輔助
- 內層 repo：真正的訓練程式碼與環境實作

你平常要看環境邏輯、reward、終止條件、訓練腳本，幾乎都在內層 repo。

最重要的路徑如下：

- 內層主程式碼：
  [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone)
- 內層訓練與播放腳本：
  [`/home/jonas/Drone/Drone/scripts/skrl`](/home/jonas/Drone/Drone/scripts/skrl)
- 歷史訓練紀錄與 checkpoint：
  [`/home/jonas/Drone/Drone/logs/skrl`](/home/jonas/Drone/Drone/logs/skrl)
- 已整理的模型紀錄：
  [`/home/jonas/Drone/Drone/docs/trained_models.md`](/home/jonas/Drone/Drone/docs/trained_models.md)

---

## 3. 目前主要在用哪些環境

雖然專案裡有很多任務系列，但目前主要在看的，是 `Vehicle-Moving` 這組：

- `Drone-Direct-Target-Touch-Vehicle-Moving-Pre-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Pre-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage0-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage0-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage1-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage1-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage2-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage2-Test-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage3-v0`
- `Drone-Direct-Target-Touch-Vehicle-Moving-Stage3-Test-v0`

這些環境的註冊位置在：
- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/__init__.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/__init__.py)

如果你想知道某個 task ID 對應哪個 cfg class、哪個 Python 類別，先看這個檔案。

---

## 4. Vehicle-Moving 系列在做什麼

這系列任務的核心目標是：

- 無人機從某個起始位置出生
- 目標物會在空中持續移動
- policy 要控制無人機追上目標並碰觸它
- 若撞地、掉太低、太遠、或某些 stage 設定下傾角過大，就會提早終止

和靜態目標環境相比，`Vehicle-Moving` 多了這些難點：

- 目標本身在動
- 追擊過程中要持續修正方向
- 距離越遠，reward 訊號越稀疏
- 太激進時容易翻車或撞地

因此訓練策略通常是：

1. 先用較短距離、較低速度學會追擊
2. 再逐步增加重生距離
3. 最後再把目標速度拉高

---

## 5. 動作空間與觀察空間

### 5.1 動作空間

這系列主要是 4 維動作：

- `a[0]`：總推力命令
- `a[1]`：roll 力矩
- `a[2]`：pitch 力矩
- `a[3]`：yaw 力矩

實作位置：
- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py)

目前常見基礎設定：

- `thrust_to_weight = 4`
- `moment_scale = 0.05`

### 5.2 觀察空間

`Vehicle-Moving` 目前主要使用 25 維擴展觀察。

觀察大致包含：

- 無人機位置
- 無人機線速度
- 無人機角速度
- 姿態旋轉矩陣
- 目標位置
- 上一步動作

對應設定檔：
- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving_cfg.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving_cfg.py)

對應組裝邏輯：
- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py)

---

## 6. Vehicle-Moving 系列的環境設定怎麼看

最重要的設定檔是：

- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving_cfg.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving_cfg.py)

這個檔案裡分成：

- `BaseEnvCfg`
- `Pre`
- `Stage0`
- `Stage1`
- `Stage2`
- `Stage3`
- 各自的 `TestEnvCfg`

你在這裡最常查的欄位有：

- `episode_length_s`
- `target_spawn_distance_min`
- `target_spawn_distance_max`
- `moving_target_speed`
- `far_away_termination_distance`
- `enable_tilt_limit_termination`
- `max_tilt_deg`
- `distance_to_goal_tanh_scale`
- `distance_to_goal_reward_scale`
- `approach_reward_scale`
- `progress_reward_scale`

### 6.1 目前 Stage3（依目前程式碼）

目前內層程式碼中的 `Stage3EnvCfg` 是一個「長距離、慢速過渡版」：

- `episode_length_s = 180.0`
- `target_spawn_distance_min/max = 20.0 / 100.0`
- `moving_target_speed = 3.0`
- `far_away_termination_distance = 130.0`
- `enable_tilt_limit_termination = true`
- `max_tilt_deg = 55.0`
- `distance_to_goal_tanh_scale = 12.0`

目前 `Stage3TestEnvCfg` 和 `Stage3EnvCfg` 保持一致，不再另外覆寫。

---

## 7. 目標物是怎麼移動的

`Vehicle-Moving` 的 moving target 更新邏輯在：

- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving.py)

核心機制：

- 每步重算「目標相對於無人機」的方向
- 目標會朝某個方向持續移動
- z 軸可加正弦波
- 高度超出上下限會被夾回範圍內

常見參數：

- `moving_target_speed`
- `moving_target_vertical_dir_scale`
- `moving_target_turn_rate_limit`
- `moving_target_no_instant_reverse`
- `moving_target_z_wave_amplitude`
- `moving_target_z_wave_period_s`
- `moving_target_z_min/max`

---

## 8. Reward 函數怎麼看

`Vehicle-Moving` 沒有自己獨立寫一套 reward，而是共用：

- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py)

最重要的函式是：

- `_get_rewards`
- `_get_dones`
- `_reset_idx_impl`

### 8.1 目前主要 reward 項

對 `Vehicle-Moving` 來說，最重要的是這幾項：

1. `distance_to_goal`
用途：
- 距離目標越近，reward 越高

公式：
- `distance_to_goal_mapped = 1 - tanh(distance / distance_to_goal_tanh_scale)`
- 再乘 `distance_to_goal_reward_scale * step_dt`

2. `approach_reward`
用途：
- 朝目標方向飛行時給正獎勵

公式：
- `approach_speed = dot(v_body, goal_dir_body)`
- `reward = approach_reward_scale * max(approach_speed, 0) * step_dt`

3. `progress_reward`
用途：
- 這一步比上一步更接近目標，就給正獎勵
- 如果變遠，這項就變成負值

公式：
- `progress = prev_distance - current_distance`
- `reward = progress_reward_scale * progress`

4. `touch_bonus`
用途：
- 碰到目標時給一次性大獎勵

5. `touch_early_bonus`
用途：
- 越早碰到，額外加分越多

6. `time_penalty`
用途：
- 每步固定扣分，逼策略不要拖太久

7. `death_penalty`
用途：
- 撞地、掉太低、或某些 stage 下傾角超限時扣分

8. `far_away_penalty`
用途：
- 離目標過遠時扣分

9. `failure_penalty`
用途：
- timeout 但沒成功碰到，也不是 died/far-away 時扣分

### 8.2 目前很多項是關閉的

以下這些欄位雖然程式有算，但目前常常設為 0：

- `speed_to_goal_reward_scale`
- `distance_penalty_scale`
- `near_touch_push_reward_scale`
- `follow_behind_penalty_scale`
- `tilt_forward_reward_scale`
- `lin_vel_reward_scale`
- `ang_vel_reward_scale`

所以目前訓練主要還是靠：

- `distance_to_goal`
- `approach_reward`
- `progress_reward`
- `touch_bonus`
- 各種終止/失敗懲罰

---

## 9. 終止條件怎麼看

終止條件也是看：

- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py)

主要幾種終止：

- `time_out`：超過回合長度
- `died_by_height`：高度太低
- `ground_contact`：接地
- `died_by_tilt`：傾角超限（只有啟用時才生效）
- `far_away`：距離超過 `far_away_termination_distance`
- `touched`：成功碰到目標

目前 `Stage3` 訓練環境中，傾角限制是啟用的；測試環境也與訓練環境一致。

---

## 10. 目前有哪些已訓練模型可用

看：

- [`/home/jonas/Drone/Drone/docs/trained_models.md`](/home/jonas/Drone/Drone/docs/trained_models.md)

這份檔案紀錄：

- 目前有哪些 `best_agent.pt`
- 每個模型的訓練階段
- 它是從哪個 checkpoint fine-tune 來的
- 當時實際訓練用的環境設定
- 哪些模型可用、哪些模型失敗

如果你想知道：

- 「Stage2 是從哪個模型接著訓的？」
- 「失敗的 Stage3 當時用的是 20~50 還是 20~100？」
- 「某個模型訓練時目標速度是多少？」

先看這個檔案最快。

---

## 11. 訓練、測試、播放要看哪些腳本

最常用的是：

- 訓練：
  [`/home/jonas/Drone/Drone/scripts/skrl/train.py`](/home/jonas/Drone/Drone/scripts/skrl/train.py)
- 播放 / 看 checkpoint：
  [`/home/jonas/Drone/Drone/scripts/skrl/play.py`](/home/jonas/Drone/Drone/scripts/skrl/play.py)
- 多 stage shell workflow：
  [`/home/jonas/Drone/Drone/scripts/skrl/train_target_touch_vehicle_stages.sh`](/home/jonas/Drone/Drone/scripts/skrl/train_target_touch_vehicle_stages.sh)

如果你只是想手動開某一階段訓練，通常直接用 `train.py` 就好。

---

## 12. tmux 自動接續訓練

如果你現在用 tmux 跑長時間訓練，專案裡有一支小工具：

- [`/home/jonas/Drone/Drone/scripts/tmux_run_after_pane_idle.sh`](/home/jonas/Drone/Drone/scripts/tmux_run_after_pane_idle.sh)

作用是：

- 監看某個 tmux pane
- 等目前 `python` 結束
- 自動送出下一條訓練指令

這支腳本本身不綁定任何 task。  
它會執行你啟動它時傳進去的那條指令。

---

## 13. 想查某個問題時，應該去哪裡

### 13.1 想知道有哪些環境 ID

看：
- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/__init__.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/__init__.py)

### 13.2 想知道某個 stage 的設定

看：
- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving_cfg.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving_cfg.py)

### 13.3 想知道 reward / done / reset 怎麼算

看：
- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py)

### 13.4 想知道 moving target 怎麼動

看：
- [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving.py)

### 13.5 想知道歷史模型與訓練鏈

看：
- [`/home/jonas/Drone/Drone/docs/trained_models.md`](/home/jonas/Drone/Drone/docs/trained_models.md)

### 13.6 想知道某個舊模型當時用什麼設定訓的

看對應 run 底下的：
- `logs/skrl/<run>/params/env.yaml`

這很重要：

- 想知道「現在重新開訓會用什麼」：看目前 `cfg.py`
- 想知道「某個舊模型當年實際用什麼訓」：看該 run 的 `params/env.yaml`

這兩者不要混在一起看。

---

## 14. 建議第一次接手時的閱讀順序

建議照這順序看：

1. [`/home/jonas/Drone/README.md`](/home/jonas/Drone/README.md)
2. [`/home/jonas/Drone/Drone/docs/trained_models.md`](/home/jonas/Drone/Drone/docs/trained_models.md)
3. [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/__init__.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/__init__.py)
4. [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving_cfg.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving_cfg.py)
5. [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py)
6. [`/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving.py`](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving.py)
7. [`/home/jonas/Drone/Drone/scripts/skrl/train.py`](/home/jonas/Drone/Drone/scripts/skrl/train.py)

照這個順序看，通常就能把：

- 任務是什麼
- 環境怎麼配
- reward 怎麼算
- checkpoint 怎麼接

整體串起來。
