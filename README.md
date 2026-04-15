# Drone — RL Drone Target Tracking

用 **Isaac Lab + skrl** 訓練四軸無人機追逐並碰觸移動目標的強化學習專案。

## 概覽

- 演算法：PPO
- 模擬器：Isaac Lab（基於 Isaac Sim / PhysX）
- 平行環境：最多 8192 個（GPU）
- 任務：無人機在 3D 空間中追逐移動目標並碰觸它

訓練採課程學習，從短距離慢速逐步進階到長距離高速：

| Stage | 目標距離 | 目標速度 |
|-------|---------|---------|
| Stage0 | 5–15 m | 1.0 m/s |
| Stage1 | 10–20 m | 3.0 m/s |
| Stage2 | 15–25 m | 5.0 m/s |
| Stage3 | 20–100 m | 3.0 m/s |
| Stage4 | 20–60 m | 6.0 m/s |
| Stage5 | 20–80 m | 3–12 m/s |

## 安裝

1. 安裝 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
2. Clone 此 repo
3. 安裝套件：
```bash
python -m pip install -e source/Drone
```

## 訓練

```bash
python scripts/skrl/train.py --task Drone-Direct-Target-Touch-Vehicle-Moving-Stage1-v0 --num_envs 8192
```

從 checkpoint 接續訓練：
```bash
python scripts/skrl/train.py --task <TASK_ID> --num_envs 8192 --checkpoint <path/to/best_agent.pt>
```

## 播放 / 測試

```bash
python scripts/skrl/play.py --task Drone-Direct-Target-Touch-Vehicle-Moving-Stage1-Test-v0 --checkpoint <path/to/best_agent.pt>
```

## 專案結構

```
source/Drone/Drone/tasks/direct/drone/
├── target_touch/                  # reward / obs / reset / done 核心邏輯
├── target_touch_vehicle/          # 靜止目標 + 車輛系列
└── target_touch_vehicle_moving/   # 移動目標系列（主線）
scripts/skrl/
├── train.py                       # 訓練入口
└── play.py                        # 播放 / 測試入口
example_env_sample_package/        # 範例環境精簡複本（供參考）
```

## 相關文件

- `env_setup_steps.md` — 新建環境的完整步驟指南
- `example_env_sample_package/` — 最小可執行範例環境
- `source/.../README_ENV_SERIES_ZH.md` — 各系列環境詳細說明
- `source/.../README_CODE_STRUCTURE_ZH.md` — 程式模組結構說明
- `CLAUDE.md` — AI 協作規則與專案脈絡（給 Claude 看）
