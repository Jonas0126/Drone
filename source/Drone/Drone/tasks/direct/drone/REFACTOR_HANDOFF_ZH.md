# Isaac Lab + skrl 環境模組化重構紀錄

## 文件目的

本文件用來記錄目前 Isaac Lab + skrl 強化學習專案的環境重構進度、目標、限制、已完成內容、未完成內容，以及下一階段預計如何繼續。

下次若要繼續重構，請先以本文件為依據，再接續實作。

---

## 一、重構目標

本次重構的核心目標如下：

- 把過大的檔案與方法拆小，讓每個檔案與函式盡量只負責單一職責。
- 優先重構 Isaac Lab 環境實作與 cfg，不是主訓練入口檔。
- 保持目前訓練流程、環境語意、reward、observation、reset 分佈、termination threshold、action semantics 不變或盡量不變。
- 保持 tensor 的 shape、dtype、device placement 一致。
- 保持 Isaac Lab / skrl 的整合點正常。
- 讓每個系列各自成組，不要把所有系列都塞進同一套共用模組。
- 盡量降低跨系列耦合，避免讓研究者之後需要跨很多檔案才能理解一個系列。

---

## 二、使用者明確要求

目前已確認的要求如下：

- 重點不是 `main.py`，`main.py` 只是拿來比喻「希望主流程變薄」。
- 真正重構對象是環境實作與 config 結構。
- 每個系列要各自有自己的檔案與責任切分。
- 不希望最後變成所有系列共用一大坨 helper。
- 希望盡量不要依賴繼承，尤其不希望最終仍維持很深的 env inheritance chain。
- 但重構方式要保守，不能大幅改變原本行為。
- 在使用者確認前，不同步外層 repo，只修改內層 repo。
- 若改到環境設定，要同步更新 inline comments 與 README。

---

## 三、目前採用的重構策略

目前重構採用的是「分階段、保守式」策略，而不是一次性大改。

### 第一階段目標

先完成：

- 檔案責任拆分
- 系列內聚
- 舊路徑 facade 保留
- 不破壞現有 import / gym 註冊 / 執行方式

### 第二階段目標

之後再處理：

- 拔除 `target_touch_vehicle` 對 `target_touch` 的繼承
- 拔除 `target_touch_vehicle_moving` 對 `target_touch_vehicle` 的繼承
- 讓每個系列真正成為「系列內自有 env 與邏輯」

也就是說：

- 目前已先完成檔案結構模組化
- 但尚未完全去繼承化

---

## 四、目前已完成的內容

### 4.1 已建立的系列目錄

已在內層 repo 建立以下系列化目錄：

- `target_touch/`
- `target_touch_vehicle/`
- `target_touch_vehicle_moving/`

位置基準：

- `Drone/source/Drone/Drone/tasks/direct/drone/`

### 4.2 `target_touch` 已拆出的模組

`target_touch/` 目前包含：

- `env.py`
- `base_cfg.py`
- `cfg.py`
- `reset_ops.py`
- `observations.py`
- `rewards.py`
- `terminations.py`
- `scene_ops.py`
- `debug_vis.py`

說明：

- `env.py` 保留主環境 class 與整合入口
- `base_cfg.py` 放 `target_touch` 基底設定
- `cfg.py` 保留為 facade，維持舊匯入路徑
- `reset_ops.py` 拆出 reset 流程
- `observations.py` 拆出 observation 建構
- `rewards.py` 拆出 reward 計算
- `terminations.py` 拆出 done / timeout 判定
- `scene_ops.py` 拆出 scene / anchor / obstacle / demo scene 相關處理
- `debug_vis.py` 拆出 debug 可視化

### 4.3 `target_touch_vehicle` 已拆出的模組

`target_touch_vehicle/` 目前包含：

- `env.py`
- `curriculum.py`
- `base_cfg.py`
- `stage_cfgs.py`
- `cfg.py`

說明：

- `env.py` 保留 vehicle 系列主 env
- `curriculum.py` 放 vehicle distance curriculum 與 vehicle-specific reset / pre-physics hook
- `base_cfg.py` 放 vehicle 基底設定
- `stage_cfgs.py` 放 Stage0 ~ Stage5 設定
- `cfg.py` 保留為 facade

### 4.4 `target_touch_vehicle_moving` 已拆出的模組

`target_touch_vehicle_moving/` 目前包含：

- `env.py`
- `base_cfg.py`
- `stage_cfgs.py`
- `demo_cfgs.py`
- `cfg.py`
- `moving_target.py`
- `motion_sampling.py`
- `obstacle_ops.py`
- `reset_ops.py`
- `test_mode.py`
- `test_visuals.py`
- `test_stats.py`

說明：

- `env.py` 保留 moving 系列主 env
- `base_cfg.py` 放 moving 系列共同基底設定
- `stage_cfgs.py` 放 Stage0 ~ Stage5 / Test 設定
- `demo_cfgs.py` 放台北 demo 設定
- `cfg.py` 保留 facade
- `moving_target.py` 現在只保留 moving target 主流程與入口
- `motion_sampling.py` 拆出速度 / heading / turn 段長取樣
- `obstacle_ops.py` 拆出 obstacle avoidance / pushout
- `reset_ops.py` 拆出 moving reset 與 distance curriculum
- `test_mode.py` 保留為 facade
- `test_visuals.py` 拆出 trail / marker / debug callback
- `test_stats.py` 拆出 test done / reset 統計列印

### 4.5 舊路徑 facade 已保留

以下舊檔仍保留，目的是避免現有匯入與 gym 註冊立刻壞掉：

- `drone_env_target_touch.py`
- `drone_env_target_touch_cfg.py`
- `drone_env_target_touch_vehicle.py`
- `drone_env_target_touch_vehicle_cfg.py`
- `drone_env_target_touch_vehicle_moving.py`
- `drone_env_target_touch_vehicle_moving_cfg.py`

目前這些檔案主要作為 facade / compatibility layer。

### 4.6 已修正的重構過程問題

在重構過程中，已修正過以下真實問題：

- moving 系列初始化時漏掉 `_episode_min_distance` 等 buffer 初始化
- helper 中使用 dynamic `super()` 導致 test subclass reset / pre-physics 調用錯父層
- `target_touch` reset helper 中父層 reset 調用方式不安全，已改成顯式 base callback
- cfg 結構已對齊為 `base_cfg.py + cfg.py facade` 形式

### 4.7 文件更新

已新增或更新：

- `README_CODE_STRUCTURE_ZH.md`
- `README_ENV_SERIES_ZH.md`

用途：

- 記錄目前模組化後的系列結構
- 說明各檔案責任
- 說明 facade 舊路徑仍保留

### 4.8 驗證狀況

已完成的驗證：

- 內層新增與修改過的 Python 檔已通過 `py_compile`
- 使用者已實際跑過 `Stage4` / `Stage5` 測試環境，確認能跑

尚未完成的驗證：

- 尚未建立完整自動 smoke test
- 尚未有系統化地自動驗證 init / reset / step / obs-reward-done 結構

---

## 五、目前尚未完成的部分

以下屬於「還沒完成」：

- `target_touch_vehicle` 仍然繼承 `target_touch`
- `target_touch_vehicle_moving` 仍然繼承 `target_touch_vehicle`
- 也就是 class hierarchy 仍未完全去繼承化
- `target_touch/reset_ops.py` 仍然偏大
- `target_touch/scene_ops.py` 仍然偏大
- 尚未建立最小自動 smoke test
- 尚未同步到外層 repo

---

## 六、目前結構上的現況判讀

目前的狀態可以概括成：

### 已完成

- 檔案層級已經系列化
- 各系列責任比過去清楚
- 大檔已明顯縮小
- 使用者閱讀單一系列的成本已下降

### 尚未完成

- class 關係仍偏向舊有繼承鏈
- 系列之間仍不是完全獨立
- 還沒有真正做到「vehicle_moving 自己完整擁有 env 主邏輯」

---

## 七、下一階段重構目標

下一階段的主要目標是：

### 第一優先

讓 `target_touch_vehicle_moving` 脫離 `target_touch_vehicle`

目標狀態：

- `DroneTargetTouchVehicleMovingEnv` 不再繼承 `DroneTargetTouchVehicleEnv`
- `vehicle_moving` 自己擁有完整 env 主流程
- 只保留極薄 helper，不再依賴父類核心行為

### 第二優先

讓 `target_touch_vehicle` 脫離 `target_touch`

目標狀態：

- `DroneTargetTouchVehicleEnv` 不再繼承 `DroneTargetTouchEnv`
- `vehicle` 自己擁有完整 env 主流程
- 降低跨系列依賴

### 第三優先

繼續細拆 remaining 大檔

包括：

- `target_touch/reset_ops.py`
- `target_touch/scene_ops.py`

### 第四優先

補 smoke test

至少要自動驗證：

- env initialize
- reset
- step
- observations / rewards / dones 結構

---

## 八、下一階段預計如何做

若下次要繼續，建議按照以下順序做：

### Step 1

先分析 `target_touch_vehicle_moving/env.py` 目前仍依賴 `target_touch_vehicle.env` 的行為有哪些。

重點盤點：

- curriculum
- reset hook
- pre-physics hook
- base buffer / state 初始化
- reward / obs / done 是否仍經由父層取得

### Step 2

將 `vehicle` 提供給 `moving` 的實際邏輯搬回 `target_touch_vehicle_moving/` 系列內。

預計會新增或擴充的模組可能包括：

- `env.py`
- `reset_ops.py`
- `curriculum.py`（若需要）
- `buffers.py` 或 `state.py`（若需要）
- `hooks.py`（若需要，但要避免抽象過度）

### Step 3

讓 `DroneTargetTouchVehicleMovingEnv` 改為不繼承 `DroneTargetTouchVehicleEnv`。

但必須保證：

- public API 不變
- cfg key 不變
- obs/reward/done 行為不變
- tensor shape/device 不變

### Step 4

完成 moving 系列脫鉤後，再回頭處理 `target_touch_vehicle` 對 `target_touch` 的依賴。

---

## 九、重構時必須持續遵守的限制

每次繼續重構前，都要再次確認以下限制：

- 不可隨意改 config key
- 不可默默改 reward 邏輯
- 不可默默改 observation 內容
- 不可默默改 reset 分佈
- 不可默默改 termination threshold
- 不可默默改 action semantics
- 必須保持 tensor device / dtype / batch 維度一致
- 必須保持 Isaac Lab / skrl integration 正常
- 優先採取保守、逐步、低風險重構
- 不要一次性重寫整個專案
- 不要引入過多抽象層
- 保持研究用途下容易讀、容易改

---

## 十、外層同步規則

目前 source of truth 是內層 repo：

- `/home/jonas/Drone/Drone`

在使用者未明確確認前：

- 不要修改外層：
  - `/home/jonas/Drone/source`
  - `/home/jonas/Drone/scripts`

只有當使用者確認這輪重構有效後，才可將內層同步到外層。

---

## 十一、下次續做時建議的起手式

下次若要根據本文件繼續，建議先做：

1. 重新閱讀本文件
2. 確認目前 source of truth 仍是內層 repo
3. 先從 `target_touch_vehicle_moving/env.py` 著手
4. 盤點它仍透過父類取得的邏輯
5. 設計 moving 系列自有 env 所需的最小拆移方案
6. 先保守搬移，不碰外部 API
7. 完成後再驗證 Stage4 / Stage5 測試環境是否仍能跑

---

## 十二、目前一句話總結

目前已完成「系列化拆檔與責任模組化」，但尚未完成「去繼承化」；下一階段的核心工作是讓 `target_touch_vehicle_moving` 先成為真正自有的獨立系列。
