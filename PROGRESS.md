# HiFAG 项目进度

> 本文档记录 HiFAG 项目已完成事项与实验结论（含失败实验）。
> 创建：2026-07-27

---

## 已完成

- [x] 项目命名：HiFAG（Hierarchical Facial-Audio Graph）
- [x] 目录结构搭建
- [x] `DESIGN.md`：层级化设计（细 68-landmark + 粗 9-region），粗节点采用手工描述子（方案 A）
- [x] `src/hifag/paths.py`：sys.path 单点管理
- [x] `src/hifag/data/region_features.py`：9 区域 × 10 维手工描述子（几何/运动/对称性）
- [x] sanity check `src/scripts/sanity_region_features.py`：✅ **信号确认存在且较强**
- [x] `src/hifag/utils/experiment.py`：实验目录自动管理 exp_N（从 SFAF 移植）
- [x] `src/hifag/data/region_graph.py`：HiFAGFaceDataset（细图 Data 上附加 `coarse_x`）+ train 集描述子标准化 + loader 工厂
- [x] `src/hifag/models/region_gnn.py`：粗粒度分支 RegionGNN（复用 AFGNN WeightedGATConv + attention readout，解剖学/全连接边可切换）
- [x] `src/hifag/models/hifag.py`：主模型（use_fine/use_coarse/use_audio 开关 + concat + MLP，特征维度契约断言）
- [x] `src/hifag/utils/builders.py`、`src/train.py`、`src/test.py`
- [x] 冒烟测试 `tests/test_build_model.py`：9 项全绿（合成数据）
- [x] 真实数据端到端验证：A2（15.8K 参数）、A3（76.6K 参数）前向均通过
- [x] 配置 `experiments/configs/hifag_a2_coarse_only.yaml` / `hifag_a3_fine_coarse.yaml`

## 待办

### 1. 信号验证（先于建模）
- [x] `src/hifag/data/region_features.py`：9 区域手工描述子（几何/运动/对称性）
- [x] sanity check：描述子在抑郁 vs 非抑郁上的分布差异（结果见下）

### 2. 工程骨架
- [x] 全部完成（见"已完成"）

### 3. 模型
- [x] 全部完成（见"已完成"）；层级交互（fine↔coarse 消息传递）为第二阶段，未做

### 4. 实验（下一步，按顺序）
- [x] A2 粗图单独（`hifag_a2_coarse_only.yaml`）→ exp_1，test AUC 0.6728
- [x] A3 细+粗 concat（`hifag_a3_fine_coarse.yaml`，核心假设）→ exp_2，test AUC 0.6714（细分支零增量）
- [x] 多 seed（42~46）验证 → A2: 0.6599±0.0095 / A3: 0.6709±0.0045（见 exp_3~exp_10 记录）。
      对照基线：AFGNN 纯面部 `face_only_enhanced` 0.657（单 seed）；
      `face_enhanced_focal` 0.797 ± 0.011 含音频，非纯面部对照（见 exp_2 记录）
- [ ] A4~A7 消融（见 DESIGN.md 第 6 节）
  - [x] A7 加音频（`hifag_a7_full.yaml`）→ exp_11~exp_15，test AUC 0.7756±0.0085（见下）
- [ ] A1 细图单独（= AFGNN 基线，已有结果可引用，不必重跑）

---

## 实验记录

### Sanity check（2026-07-27，`experiments/results/sanity_region_features.json`）

961 样本（train+valid+test，pos=555/neg=406），T=32 帧采样，描述子时序 mean+std 聚合后做单变量检验（180 个特征）：

- ✅ **信号确认存在**：113/180 特征 p<0.05（未校正）。
- **最强特征**：`std(inner_mouth.spread)` AUC=0.692（翻转后），Cohen's d=-0.686，p=1.5e-27 —— 抑郁组内唇张开度的时序波动显著更低（表情活动度下降）。
- **模式高度一致**：top 特征几乎全是**时序 std**（而非 mean），方向全部为抑郁组更低 —— "面部动态范围收窄"是主线信号，与抑郁文献一致。
- **对称性特征有效**：眉/眼 `sym_centroid`/`sym_velocity` 的 std |d|≈0.5，p~1e-19 —— 左右不对称的波动携带信号。
- 每个区域都有 p<1e-15 的特征，粗图 9 个节点均非冗余。
- 单变量 AUC 上限约 0.69（inner_mouth.spread std），组合后应更高；这是粗图的"信号地板"，不是上限。
- 注意：`cx/cy` 的 std 反映头部运动（抑郁组头动更少），`spread/area` 的 std 反映表情形变——两类信号都有，建模时都可保留。

**结论：粗粒度层假设成立，进入建模阶段（A2 粗图单独 → A3 细+粗）。**

### exp_1 — A2 粗图单独（2026-07-27，seed 42）

- best valid AUC 0.7138（ep16），早停 ep31；**test AUC 0.6728**（acc 0.651 / F1 0.741）。
- 解读：粗图（15.8K 参数）单独有真实判别力，高于随机、接近单变量地板（0.69），
  但明显低于细图基线 0.797 —— 符合预期，粗图定位是"补充信号"而非替代。
- 注意：valid AUC 到 ep31 仍在缓升（0.7127），patience=15 可能略紧，多 seed 后再判断。
- 明细：`experiments/exp_1/`；索引：`experiments/INDEX.md`。

### exp_2 — A3 细+粗 concat（2026-07-28，seed 42）

- best valid AUC 0.7181（ep6），早停 ep21；**test AUC 0.6714**（acc 0.660 / F1 0.690）。
- 与 A2 粗图单独（0.6728）**持平** —— 加入细分支没有带来增量，粗细信号在 concat
  融合下高度冗余（或细分支在联合训练中被粗分支压制）。
- **基线对照修正**：AFGNN `face_enhanced_focal` 0.797 ± 0.011 含**音频**模态
  （`use_audio: true` + cross-attention），不是纯面部基线；AFGNN 纯面部
  `face_only_enhanced`（weighted_bce，单 seed）Test AUC **0.657**。
  A2/A3（无音频）均已略超纯面部基线，与 0.797 的差距来自音频。
- 观察：valid 上 recall ≈ 0.10（acc 0.47），模型在 valid 接近全预测负例但 AUC 仍有
  0.71 —— 排序信号在、阈值偏移；test 阈值 0.5 下 recall 0.65，不算坍缩。
- 明细：`experiments/exp_2/`；索引：`experiments/INDEX.md`。
- ⚠️ "细分支零增量"仅是 seed 42 单点结论，多 seed 后已修正（见下节）。

### exp_3~exp_10 — A2/A3 多 seed（2026-07-28，seeds 42~46）

| 配置 | test AUC (mean ± std) |
|------|----------------------|
| A2 粗图单独 | 0.6599 ± 0.0095 |
| A3 细+粗 | **0.6709 ± 0.0045** |

- 配对（同 seed）A3−A2 均值 **+0.011**，5 个 seed 中 4 个为正（配对 t≈2.27, p≈0.086，
  边缘显著）。**修正 exp_2 的单 seed 结论：细分支有小但真实存在的增量，且让训练更稳**
  （A3 std 0.0045 vs A2 0.0095；valid AUC A3 一致 0.73+ vs A2 0.70±）。
- A2 seed 46 训练不稳（best 停在 ep1），A3 无此现象。
- 两者均高于 AFGNN 纯面部基线 0.657（单 seed），但差距不大；
  音频仍是通往 0.797 的主要缺口。
- 明细：`experiments/exp_3/` ~ `experiments/exp_10/`；汇总表见 `experiments/INDEX.md`。

### exp_11~exp_15 — A7 细+粗+音频（2026-07-28，seeds 42~46）

| 配置 | test AUC (mean ± std) |
|------|----------------------|
| A7 细+粗+音频（concat） | **0.7756 ± 0.0085** |

- 逐 seed：0.7782 / 0.7880 / 0.7756 / 0.7651 / 0.7713；valid AUC 0.794~0.818。
- vs A3（无音频）0.6709：**+0.105**，音频模态是通往高性能的主要增量，符合预期。
- vs AFGNN `face_enhanced_focal` 基线 0.797 ± 0.011：**−0.021，未追平**。
  主要差异候选：融合方式（A7 为 concat，基线为 cross-attention）。
- ⚠️ 3/5 seeds 的 best valid 出现在 ep3~5（早停 ep18~20）——音频分支收敛极快，
  可能欠训练；后续可考虑复查 lr/scheduler，或直接上 cross-attention 融合。
- 决策指向：要追平 0.797，下一步优先实现 cross-attention 融合（对齐基线），
  而非继续调 concat 的超参。
- 明细：`experiments/exp_11/` ~ `experiments/exp_15/`；汇总表见 `experiments/INDEX.md`。

---

## 当前阻塞 / 风险

首要风险（粗描述子无判别信号）已排除。剩余风险见 `DESIGN.md` 第 8 节：粗图与细图信息冗余（A4/A5 消融定位）、800 样本过拟合（粗图参数预算 < 细图）。
