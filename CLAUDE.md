# CLAUDE.md — Drone 專案 AI 協作規則

這份文件是給 Claude（AI）看的，說明這個專案的結構、規則、以及你應該如何協作。

---

## 1. 專案是什麼

這個專案是在 **Isaac Lab / skrl** 上訓練四軸無人機（quadcopter）的強化學習任務。
核心目標是讓無人機學會在 3D 空間中追逐並碰觸一個移動目標（Vehicle-Moving 系列）。

訓練用 PPO 演算法，支援最多 8192 個平行環境，在 GPU 上跑。

---

## 2. 兩層 Repo 結構（最重要）

```
/home/jonas/Drone/          ← 外層 repo（你的 git，版控備份用）
/home/jonas/Drone/Drone/    ← 內層 repo（clone 別人的，訓練實驗用）
```

### 規則

- **內層** `/home/jonas/Drone/Drone/`：
  - clone 自別人的 upstream repo
  - 只用來跑訓練、做實驗
  - **絕對不做任何 git commit / push**
  - 程式碼改動都先在這裡做

- **外層** `/home/jonas/Drone/`：
  - 你自己的 repo，remote 是 `github.com/Jonas0126/Drone`
  - 所有 git commit / push 只在這裡操作
  - 只有內層改動**確認穩定後**，才手動同步到外層

### 同步方向

```
內層（實驗） → 確認穩定 → 外層（版控）
```

同步指令（供參考）：
```bash
rsync -av --exclude="*.pyc" --exclude="__pycache__" \
  /home/jonas/Drone/Drone/source/ /home/jonas/Drone/source/

rsync -av --exclude="*.pyc" --exclude="__pycache__" \
  /home/jonas/Drone/Drone/scripts/ /home/jonas/Drone/scripts/
```

---

## 3. 外層 git 追蹤的範圍

外層 repo 只追蹤以下目錄與檔案：

| 路徑 | 說明 |
|------|------|
| `source/` | 環境程式碼（從內層同步過來） |
| `scripts/` | 訓練 / 播放腳本（從內層同步過來） |
| `example_env_sample_package/` | 範例環境精簡複本，供參考 |
| `*.md`（根目錄） | 專案說明文件 |
| `.gitignore` | git 忽略規則 |
| `CLAUDE.md` | 本檔案 |

**不追蹤的東西：**
- `Drone/`（內層 repo）
- `outputs/`（實驗輸出）
- `.agents/`、`.codex/`（AI 工具設定）
- `__pycache__/`、`*.pyc`
- 大型 USD assets 檔

---

## 4. 所有文件都在外層管理

這個專案的文件（`.md` 檔）以**外層**為主。

內層的 `README_CODE_STRUCTURE_ZH.md`、`REFACTOR_HANDOFF_ZH.md` 等 md 是由 rsync 同步過來的，不是獨立維護的。

如果你要新增或修改文件，應該：
1. 在外層寫好
2. 若同樣的文件也存在於內層，同步過去；若只是外層的說明文件，外層維護即可

---

## 5. 訓練任務主線

目前主要訓練的是 **Vehicle-Moving** 系列，難度遞增：

| Stage | 目標距離 | 目標速度 | 備注 |
|-------|---------|---------|------|
| Pre | 5~15m | 靜態 | 熱身 |
| Stage0 | 5~15m | 1.0 m/s | 入門 |
| Stage1 | 10~20m | 3.0 m/s | 穩定起點 |
| Stage2 | 15~25m | 5.0 m/s | 激進 |
| Stage3 | 20~100m | 3.0 m/s | 長距離，有傾角限制 |
| Stage4 | 20~60m | 6.0 m/s | — |
| Stage5 | 20~80m | 3~12 m/s | road_like 移動，速度抽樣 |

訓練鏈（fine-tune 方向）：Pre → Stage1 → Stage2 → Stage3 → ...

---

## 6. 關鍵檔案速查

### 想改環境設定
```
內層: /home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/stage_cfgs.py
```

### 想看 reward / done / reset 邏輯
```
內層: /home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch/
  - rewards.py
  - terminations.py
  - reset_ops.py
```

### 想看 moving target 更新邏輯
```
內層: /home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/target_touch_vehicle_moving/moving_target.py
```

### 想跑訓練
```
內層: python /home/jonas/Drone/Drone/scripts/skrl/train.py --task <TASK_ID> --num_envs 8192
```

### 想看 / 播放 checkpoint
```
內層: python /home/jonas/Drone/Drone/scripts/skrl/play.py --task <TASK_ID> --checkpoint <path>
```

### 已訓練模型紀錄
```
內層: /home/jonas/Drone/Drone/docs/trained_models.md
```

### 環境 ID 對應表
```
內層: /home/jonas/Drone/Drone/source/Drone/Drone/tasks/direct/drone/__init__.py
```

---

## 7. 程式碼架構概覽

```
source/Drone/Drone/tasks/direct/drone/
├── target_touch/               # 靜止目標基底（reward/obs/reset/done 核心邏輯都在這）
├── target_touch_vehicle/       # 有車輛的版本
└── target_touch_vehicle_moving/ # 移動目標版（主線）
    ├── env.py                  # 環境主體
    ├── base_cfg.py             # 基礎參數
    ├── stage_cfgs.py           # Stage0~Stage5 參數
    ├── demo_cfgs.py            # 台北展示場景
    ├── moving_target.py        # 目標移動主流程
    ├── motion_sampling.py      # 速度/heading/turn 取樣
    ├── obstacle_ops.py         # 障礙物避讓
    ├── reset_ops.py            # reset 流程
    ├── test_visuals.py         # debug 視覺化
    └── test_stats.py           # 測試統計
```

舊路徑 facade（保留相容性，不是主邏輯）：
- `drone_env_target_touch.py`
- `drone_env_target_touch_vehicle.py`
- `drone_env_target_touch_vehicle_moving.py`
- 對應 `_cfg.py` 檔

---

## 8. 對 Claude 的行為要求

1. **改程式碼永遠先動內層**，不主動動外層 source / scripts
2. **同步外層要等使用者明確確認**，不自行決定同步時機
3. **git 操作只在外層**，內層絕對不碰 git
4. **commit 前先確認** `.gitignore` 沒有把不該追蹤的東西加進去
5. 文件有異動時，同步更新這份 `CLAUDE.md` 或相關 md
