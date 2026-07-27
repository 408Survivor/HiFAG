# HiFAG 项目状态备忘录（给 Kimi 的下一次会话）

> 最后更新：2026-07-27

## 协作工作流（用户定，必须遵守）

- **所有实验由用户亲自执行**：Kimi 只给出完整命令，不在后台/前台代为运行训练或评估。
- **实验编号跟踪**：`experiments/INDEX.md` 是权威索引，维护"下一个可用 exp 编号"和
  exp_N ↔ (配置, seed, 指标) 对照表。每次用户跑完实验，Kimi 根据输出更新 INDEX.md
  并递增编号。给命令时若涉及 exp_N（如 test.py 的 checkpoint 路径），以 INDEX.md 为准。
- 运行中断会留下残缺 exp_N 目录，重跑前先删除（见 INDEX.md 编号规则）。

## Git / GitHub

- 远程仓库：`git@github.com:408Survivor/HiFAG.git`（main 分支，2026-07-27 首次推送）。
- 本机 DNS 把 github.com 解析到不可达的 20.205.x.x；已用 `~/.ssh/config` 把
  `github.com` 直连真实 IP `140.82.112.3:22`（免 sudo 绕过），SSH key 为
  `~/.ssh/id_ed25519`。git push/pull 直接可用；若失效，重测一个 140.82.x.x 可用 IP 替换。
- `gh` CLI 已装但未登录（网页 API 域名不通，暂不依赖它）。
- 提交署名（repo 本地配置）：408Survivor / 408Survivor@users.noreply.github.com。

## 项目定位

HiFAG（Hierarchical Facial-Audio Graph Network）：在 AFGNN 的 68-landmark 细粒度面部图之上增加 **9-region 粗粒度面部图**，层级化面部建模用于 D-Vlog 抑郁检测。

- 工作目录：`/home/ltq/DepressionCode/DepGNN/HiFAG`
- AFGNN 目录：`/home/ltq/DepressionCode/DepGNN/AFGNN`（**只读复用，不修改**）
- SFAF 目录：`/home/ltq/DepressionCode/DepGNN/SFAF`（上一轮探索，已归档；其教训见下）
- 环境：`conda activate DVlog`
- 数据：`/data/ltq/DVlog/processed_official_features/{train,valid,test}_{visual,labels,acoustic}.npy`
  - visual: (N, 596, 136) 归一化 landmark；labels 二分类（pos≈58%）；train/valid/test = 647/102/212

## 核心设计（详见 DESIGN.md）

- 粗节点 = **手工描述子（方案 A，零参数）**：每区域每帧 10 维 = 质心(2) + 面积(1) + 散布度(1) + 平均速度(2) + 速率(1) + 运动方差(1) + 对称性(2)。左右眉(1,2)、左右眼(5,6) 为对称对。
- 区域划分复用 AFGNN `LANDMARK_GROUPS_68`（9 区域，定义已复制进 `region_features.py`）。
- 粗图：9 节点 × T=32 帧；空间边起步用解剖学邻接，时间边同区域相邻帧。
- 细粒度分支 = 复用 AFGNN FaceGNN；音频 = 复用 AFGNN AudioGNN；融合 concat 起步。
- 训练配方沿用 SFAF 最优：focal α=0.25 γ=2.0 + AUC 早停 + dropout 0.5。
- 层级交互（fine↔coarse 消息传递）是**第二阶段**，第一阶段不做。

## 已完成

1. 项目骨架 + `DESIGN.md` / `README.md` / `PROGRESS.md`。
2. `src/hifag/paths.py`：sys.path 单点管理（HiFAG/src 优先，AFGNN/src 追加）。
3. `src/hifag/data/region_features.py`：描述子计算（`compute_region_features`, (T,68,2)→(T,9,10)）。
4. `src/scripts/sanity_region_features.py` + 运行完成，报告在 `experiments/results/sanity_region_features.json`。
5. 工程骨架（2026-07-27 完成，冒烟测试 9 项全绿 + 真实数据前向验证）：
   - `src/hifag/utils/experiment.py`（exp_N 自动管理，移植自 SFAF）
   - `src/hifag/utils/builders.py`、`src/train.py`、`src/test.py`
   - `src/hifag/data/region_graph.py`：`HiFAGFaceDataset` 在 AFGNN 细图 Data 上附加 `coarse_x`（从 `data.x[:, :2]` 还原采样后坐标计算，增广一致）；train 集描述子标准化（类同 audio norm stats）
   - `src/hifag/models/region_gnn.py`：RegionGNN（复用 AFGNN `WeightedGATConv` + AttentionalAggregation；边在模型内按 batch 构建，同 AudioGNN 模式）
   - `src/hifag/models/hifag.py`：主模型（use_fine/use_coarse/use_audio + concat + MLP；粗细两路入口均断言特征维度）
   - `tests/test_build_model.py`（合成数据，含特征契约断言测试）
   - `experiments/configs/hifag_a2_coarse_only.yaml` / `hifag_a3_fine_coarse.yaml`

## Sanity check 结论（关键，假设已验证）

- ✅ 粗描述子**有强信号**：180 个聚合特征中 113 个 p<0.05。
- 最强：`std(inner_mouth.spread)` 单变量 AUC=0.692，d=-0.686，p=1.5e-27。
- 模式：top 特征全是**时序 std**，方向一致为抑郁组更低 = 面部动态范围收窄。
- 对称性特征（眉/眼 sym_* 的 std）|d|≈0.5，验证了"细图难学、粗图能抓"的假设。
- 9 个区域全部非冗余。注意区分：cx/cy 的 std = 头部运动，spread/area 的 std = 表情形变（论文写作需分开讨论）。

## 关键实现决策（2026-07-27 定下，改动前先想清楚）

- **粗图边不进 Data**：边对所有样本相同，RegionGNN 把 base edge_index 存为 buffer，forward 时按图数偏移（AudioGNN 同款模式），避免 PyG 自定义 `__inc__` 的坑。
- **粗节点排布 frame-major**：node = t*9 + region_id，与细图 node = t*68 + landmark 一致。
- **粗描述子从细图 x 还原坐标计算**（而非重新采样），保证与细图分支逐帧对齐、且吃到同样的增广。
- **空间边解剖学邻接**：轮廓-全区域、眉-眼、眼-鼻、鼻梁-鼻底、鼻-嘴、外唇-内唇（15 对无向）；`coarse_edge_mode: "full"` 可切全连接（A6）。
- **训练配方**：focal α=0.25 γ=2.0 + AUC 早停 + dropout 0.5 + plateau scheduler + grad_clip 1.0（已写进两个 yaml）。
- A2 参数 15.8K，A3 参数 76.6K（< AFGNN 基线，符合小数据预算原则）。

## 未落实（下一个会话的任务，按顺序）

1. 跑 A2：`python src/train.py --config experiments/configs/hifag_a2_coarse_only.yaml --seed 42`
2. 跑 A3：`python src/train.py --config experiments/configs/hifag_a3_fine_coarse.yaml --seed 42`
3. 评估：`python src/test.py --config <yaml> --checkpoint <exp_N>/best_seed42.pt --split test`
4. 多 seed（42~46）。对照基线：AFGNN `face_enhanced_focal` Test AUC **0.797 ± 0.011**（5 seeds，划分一致）。
5. 视结果做 A4~A7 消融（对称性/运动特征、边拓扑、音频）。A4/A5 需在 `compute_region_features` 加特征组开关（当前未实现）。
6. 第二阶段：层级交互（fine↔coarse 消息传递），A3 有增量才做。

## SFAF 教训（不要重犯）

1. 小数据（~800 样本）上架构复杂度是负收益：GRU+Transformer 堆叠（805K 参数）不如简单 GNN（112K）。
2. 特征维度必须断言，禁止静默截断（SFAF 曾静默丢弃 region one-hot，误导了一个实验结论）。
3. focal α 不要低于 0.25（α=0.1 曾导致模型坍缩全预测负例）。
4. F1 早停在模型接近全负例时会选中噪声 epoch；用 AUC 早停。
5. 失败实验也记入 PROGRESS.md。

## 常用命令

```bash
cd /home/ltq/DepressionCode/DepGNN/HiFAG
conda activate DVlog

# 信号 sanity check（已跑通）
python src/scripts/sanity_region_features.py \
    --data_dir /data/ltq/DVlog/processed_official_features --num_frames 32
```
