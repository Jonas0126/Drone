# Vehicle Stage1 參數快照

更新時間：2026-02-25
檔案來源：`Drone/source/Drone/Drone/tasks/direct/drone/drone_env_target_touch_vehicle_cfg.py`

## 1) 調整前（原本）

- `distance_to_goal_tanh_scale = 0.8`
- `distance_penalty_scale = 0.1`
- `approach_reward_scale = 0.1`
- `touch_bonus_reward = 100.0`
- `death_penalty = 100.0`

## 2) 目前值（調整後）

- `distance_to_goal_tanh_scale = 8.0`
- `distance_penalty_scale = 0.08`
- `approach_reward_scale = 0.2`
- `touch_bonus_reward = 120.0`
- `death_penalty = 120.0`

## 3) 備註

- 本快照只記錄你這次指定要調整的 Stage1 核心 reward 參數。
- 其他 Stage1 參數（重生範圍、far_away 等）請以 cfg 原檔為準。
