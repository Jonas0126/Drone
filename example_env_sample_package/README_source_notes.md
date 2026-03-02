# example env 範例檔案包

此資料夾為「Vehicle example env」精簡複本，僅保留範例環境直接相關內容。

## 檔案清單
- `drone_env_target_touch.py`：Touch 核心環境邏輯（action/observation/reward/done/reset）
- `drone_env_target_touch_vehicle_cfg.py`：僅保留 Example Env 設定
- `skrl_ppo_target_touch_vehicle_example_env_cfg.yaml`：example env 的 skrl PPO 訓練設定
- `drone_tasks_example_env_registration.py`：example env 任務註冊範例

## 原始來源
- `/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch.py`
- `/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_cfg.py`
- `/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/agents/skrl_ppo_target_touch_vehicle_stage0_cfg.yaml`
- `/home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/__init__.py`（已抽出 example env 註冊區塊）
