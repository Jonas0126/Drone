## Step 1. 設定機器人模型（無人機、機器手臂等等）

這一步是在定義「你要控制的實體是誰」。

### 1.1 設定模型 USD 檔位置
- 指定機器人 USD/USDA/USDC 的路徑。
- 確認模型內有正確的 articulation 結構（link、joint、root）。
- 確認 prim path 命名規則一致，避免後續 view / regex 抓不到。

### 1.2 設定初始狀態
- 設定初始位置與姿態（base pose）。
- 設定初始線速度、角速度。
- 有關節時設定初始 joint position / velocity。
- 明確定義 reset 時是否固定初始值，或加隨機擾動。

### 1.3 設定物理規則與限制
- 是否受重力影響。
- 最大速度（線速度、角速度）與可能的速度裁切策略。
- 關節限制（上下界、關節阻尼、驅動模式、最大力矩）。
- 接觸相關參數（碰撞殼、接觸偏移等）是否需要調整。

重點：這一步先把「物理上可動、可控」打通，後面 action/reward 才有意義。

---

## Step 2. 設定動作空間（模型輸出什麼）與控制介面

這一步是在定義 policy 輸出的語意。

- 決定 action space 維度與範圍（例如 `[-1, 1]`）。
- 定義每一維 action 對應哪個控制量：
  - 推力/力矩
  - 目標速度
  - 目標角速度
  - 關節目標位置/速度/力矩
- 決定控制型態是絕對命令（absolute）還是增量命令（delta）。
- 決定是否需要 action smoothing / low-pass filter。

重點：action 的語意要穩定、可學習，避免同時混太多控制概念。

---

## Step 3. 設定 action 如何轉換成力，並應用到機器

這一步是把神經網路輸出真的送進模擬器。

- 在 `_pre_physics_step` 或等價流程接收 action。
- 先做前處理：clamp、縮放、單位轉換。
- 將 action 轉成可施加的控制量（force/torque/joint target）。
- 在 `_apply_action`（或同等方法）實際套用到 robot。
- 確認控制頻率：每幾個 sim step 更新一次 action（decimation）。

建議檢查：
- action 全 0 時系統行為是否合理。
- action 飽和時是否爆震或數值發散。

---

## Step 4. 設定感測器（用哪些感測器、參數如何設）

這一步是在定義 observation 的資料來源。

- 決定使用哪些感測器：
  - proprioception（位置、速度、姿態）
  - contact sensor
  - ray caster / lidar
  - camera（RGB/Depth/Segmentation）
- 設定更新頻率、延遲、噪聲模型。
- 設定量測座標系（world frame / body frame）。
- 若有多感測器，確認時間同步策略。

重點：先用最少但關鍵的感測器，讓任務可學，再逐步加複雜度。

---

## Step 5. 設定要載入的環境（地形、平行環境數、間距）

這一步是 scene 複製與大規模訓練的基礎。

- 選擇地形/場景資產（平地、障礙、隨機地形）。
- 設定 `num_envs`（平行環境數）。
- 設定 `env_spacing`（環境間距）避免互相碰撞。
- 規劃 task 中其他物件（目標點、障礙物）在每個 env 的配置方式。

建議：先小規模（例如 64~256 env）驗證正確，再擴到大規模。

---

## Step 6. 設定地板與接觸材料

這一步影響穩定性與真實度。

- 設定地板是否可碰撞。
- 設定靜摩擦、動摩擦、彈性/恢復係數。
- 設定接觸模型相關參數（若引擎可調）。
- 確認地板法向與尺度正確，避免「看似平地其實有偏差」。

重點：地板材料與機器人材料配對，會直接影響滑移、起飛、落地表現。

---

## Step 7. 設定觀察空間（輸入給模型的資訊）

這一步是在定義 policy 看得到什麼。

- 明確列出 observation 向量的每一欄位。
- 只放與任務有關且可泛化的資訊，避免資訊洩漏。
- 決定是否加入歷史資訊（stack frames）或狀態估計量。
- 做正規化/尺度對齊（例如速度、距離量級一致）。
- 需要時加入噪聲，提升 sim-to-real 魯棒性。

建議：
- 為 observation 建立維度檢查與欄位註解。
- 在 log 中抽樣印出統計值（mean/std/min/max）找異常。

---

## Step 8. 設定模擬時間步與 episode 時間設計

這一步是在定義訓練的時間解析度。

- 設定 `sim dt`（物理模擬步長）。
- 設定 action decimation（幾個 sim step 才收一次 action）。
- 推導 control dt：`control_dt = sim_dt * decimation`。
- 設定 episode 長度（秒或 step）。

關鍵原則：
- 太大的 `sim dt` 可能不穩定。
- 太小的 `sim dt` 計算成本高。
- episode 要長到能學到完整行為，但不能長到浪費收斂效率。

---

## Step 9. 設定獎勵函數（獎懲機制）

這一步是任務成功與否的核心。

- 拆成多個可解釋 reward term：
  - 目標接近/追蹤
  - 姿態穩定
  - 能耗或控制平滑
  - 安全懲罰（碰撞、越界）
- 設定每個 term 的權重，避免某項完全主導。
- 檢查 reward 尺度，避免數值差距過大。
- 區分 dense reward 與 sparse bonus（達成目標時額外加分）。

建議：
- 每個 reward term 分開記錄 log。
- 先保證「最小可學」，再追求精緻 shaping。

---

## Step 10. 設定 episode 中止機制（`get_dones`）

這一步定義什麼情況算失敗或回合結束。

- 時間到（timeout）。
- 失敗條件（墜機、翻覆、碰撞、越界、關節超限）。
- 成功條件（觸碰目標、達成姿態、停留一定時間）。
- 分清楚 `terminated`（任務終止）與 `truncated`（時間截斷）。

重點：done 條件要跟任務目標一致，不然 agent 會學到錯誤策略。

---

## Step 11. 設定中止後如何重置（`reset`/`_reset_idx`）

這一步決定資料分佈與泛化能力。

- 重置機器人狀態：位置、姿態、速度、關節。
- 重置任務物件：目標點、障礙、路徑參數。
- 加入隨機化（domain randomization）：
  - 初始位姿
  - 質量/摩擦/阻尼
  - 感測噪聲
  - 外力擾動
- 確保重置只作用於需要重置的 env index（避免全域覆蓋）。

建議：
- 重置流程要可重現（可設定 seed）。
- 記錄每次 reset 的隨機參數範圍，方便除錯。

---

## Step 12. 設定訓練演算法與模型（skrl config）

這一步定義學習器本身。

- 選擇演算法：PPO / SAC / TD3 等。
- 設定模型結構：MLP / CNN / RNN / Transformer。
- 設定關鍵超參數：
  - learning rate
  - batch size
  - rollout length
  - gamma / lambda
  - entropy coefficient
  - clip range（PPO）
- 設定訓練資源：裝置（CPU/GPU）、混合精度、checkpoint 頻率。
- 設定評估模式與儲存最佳模型策略。

重點：先用保守超參數得到可收斂基線，再做系統化調參。

---

## 實作對照（建議檔案分工）

通常可分成三塊：

1. `env config`：放靜態設定（模型、場景、時間步、材料、感測器參數）。
2. `env`：放動態邏輯（action 套用、observation、reward、dones、reset）。
3. `agent skrl config`：放演算法與網路超參數。

此外別忘了：
- task registry / train 入口註冊（讓 `--task` 能找到）。
- 最小驗證流程（先跑 reset + step + 維度檢查，再開正式訓練）。

---

## 建議開發順序（避免一次太複雜）

1. 先完成可穩定 step 的最小環境（不追求漂亮 reward）。
2. 再補 observation 與 reward，確保短時間內有學習訊號。
3. 最後再加隨機化、複雜感測器、進階模型。

這樣通常能最快找到 bug，並降低「訓練不收斂但不知道卡在哪」的風險。
