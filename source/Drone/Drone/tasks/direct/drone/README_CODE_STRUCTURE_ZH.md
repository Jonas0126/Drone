# Drone 直連任務程式結構（中文）

本文件說明 `Drone/source/Drone/Drone/tasks/direct/drone` 在模組化重構後的程式結構，重點是讓研究用途下能快速定位：

- 哪個系列的程式在哪裡
- 哪些檔案是系列內主邏輯
- 哪些舊路徑只是相容用 facade

## 1. 重構原則

- 每個系列各自成組，不把所有系列硬塞進同一套共用模組。
- 優先保持行為一致，不改 reward、observation、reset 分佈、termination threshold。
- 舊的匯入路徑盡量保留，避免現有註冊與腳本一次全部改掉。

## 2. 目前系列結構

### 2.1 `target_touch/`

位置：
- [target_touch](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch)

責任：
- [env.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/env.py)
  - 主環境 class
  - 物理 helper
  - reset / reward / obs / dones 的整合入口
- [base_cfg.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/base_cfg.py)
  - Target Touch 共同基底設定
- [cfg.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/cfg.py)
  - facade，維持舊匯入路徑
- [reset_ops.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/reset_ops.py)
  - reset 流程切分
- [observations.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/observations.py)
  - observation 組裝
- [rewards.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/rewards.py)
  - reward 計算
- [terminations.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/terminations.py)
  - terminated / timeout 判定
- [scene_ops.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/scene_ops.py)
  - scene 建立
  - anchor / obstacle cache
  - demo scene spawn helper
- [debug_vis.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/debug_vis.py)
  - 基礎 debug marker

### 2.2 `target_touch_vehicle/`

位置：
- [target_touch_vehicle](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle)

責任：
- [env.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle/env.py)
  - Vehicle 系列主 env
- [base_cfg.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle/base_cfg.py)
  - Vehicle 系列共同基底設定
- [stage_cfgs.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle/stage_cfgs.py)
  - Stage0~Stage5 設定
- [cfg.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle/cfg.py)
  - facade，維持舊匯入路徑
- [curriculum.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle/curriculum.py)
  - vehicle distance curriculum
  - vehicle-specific reset / pre-physics hook

### 2.3 `target_touch_vehicle_moving/`

位置：
- [target_touch_vehicle_moving](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving)

責任：
- [env.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/env.py)
  - Vehicle-Moving 系列主 env
- [base_cfg.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/base_cfg.py)
  - Vehicle-Moving 共同基底設定
- [stage_cfgs.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/stage_cfgs.py)
  - Stage0~Stage5 與 test cfg
- [demo_cfgs.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/demo_cfgs.py)
  - 台北 demo cfg
- [cfg.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/cfg.py)
  - facade，維持舊匯入路徑
- [moving_target.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/moving_target.py)
  - moving target 狀態
  - moving path update 主流程
- [motion_sampling.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/motion_sampling.py)
  - moving speed / heading / turn 段長取樣
- [obstacle_ops.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/obstacle_ops.py)
  - obstacle avoidance / pushout
- [reset_ops.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/reset_ops.py)
  - moving 系列 reset 與 distance curriculum
- [test_mode.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/test_mode.py)
  - facade，保留舊測試模式匯入路徑
- [test_visuals.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/test_visuals.py)
  - test/debug trail
  - test visual markers
- [test_stats.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/test_stats.py)
  - test done 判定
  - test reset 統計列印

## 3. 舊路徑相容層

以下檔案仍保留，但現在主要是 facade：

- [drone_env_target_touch.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py)
- [drone_env_target_touch_cfg.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_cfg.py)
- [drone_env_target_touch_vehicle.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle.py)
- [drone_env_target_touch_vehicle_cfg.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_cfg.py)
- [drone_env_target_touch_vehicle_moving.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving.py)
- [drone_env_target_touch_vehicle_moving_cfg.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_moving_cfg.py)

用途：
- 保持既有 Gym 註冊、腳本、匯入路徑先不壞
- 逐步把真正邏輯移入系列目錄

## 4. 目前閱讀建議

若要看：

- Touch 任務主 reward / reset
  - 先看 [target_touch/env.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/env.py)
  - 再看 [target_touch/rewards.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/rewards.py)
  - 與 [target_touch/reset_ops.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/reset_ops.py)

- Vehicle distance curriculum
  - 看 [target_touch_vehicle/curriculum.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle/curriculum.py)

- Vehicle-Moving 目標移動規則
  - 看 [target_touch_vehicle_moving/moving_target.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/moving_target.py)
  - 取樣細節在 [target_touch_vehicle_moving/motion_sampling.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/motion_sampling.py)
  - 場景避障在 [target_touch_vehicle_moving/obstacle_ops.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/obstacle_ops.py)

- Vehicle-Moving 台北 demo 設定
  - 看 [target_touch_vehicle_moving/demo_cfgs.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/demo_cfgs.py)

- Vehicle-Moving 測試統計與 trail
  - 看 [target_touch_vehicle_moving/test_stats.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/test_stats.py)
  - 與 [target_touch_vehicle_moving/test_visuals.py](/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/test_visuals.py)

## 5. 後續可再整理的點

- `target_touch` 目前只有共同基底設定，因此拆成 `base_cfg.py + cfg.py facade`
- `basic/advanced/moving` 舊系列尚未同步用相同模組風格整理
- facade 仍存在，等整體穩定後可再決定是否進一步收斂
