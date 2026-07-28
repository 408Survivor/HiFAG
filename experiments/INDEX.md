# HiFAG 实验索引

> 每次训练/评估完成后更新此表。exp_N 编号由 `src/hifag/utils/experiment.py`
> 运行时自动扫描分配（max+1），下表是人工维护的权威索引。
>
> **下一个可用编号：exp_31**

## 编号规则

- 一次 `train.py` 运行 = 一个 exp_N（含单 seed）；多 seed 就是连续多个 exp_N。
- exp_N 目录内容：`config.yaml`、`model_summary.txt`、`run_info.txt`、
  `best_seed{N}.pt`、`training_history_seed{N}.json`、`test_{split}_results.json`。
- 若运行中断/失败，exp_N 目录是残缺的：**重跑前先删除该目录**，避免编号错位。
- `outputs/` 下的 `<expid>_<时间戳>/` 是 AFGNN `run_context` 的附带日志
  （run.log + run.json + best.pt 软链），仅作日志参考，权威记录以 exp_N 为准。

## 实验登记表

| exp | 配置 | seed | 目的 | best valid AUC | test AUC | 状态/备注 |
|-----|------|------|------|----------------|----------|-----------|
| 1 | hifag_a2_coarse_only | 42 | 粗图单独判别力 | 0.7138 (ep16) | 0.6728 | ✅ 完成；早停 ep31，训练曲线后期仍缓升 |
| 2 | hifag_a3_fine_coarse | 42 | 细+粗 concat（核心假设） | 0.7181 (ep6) | 0.6714 | ✅ 完成；早停 ep21；与 A2 持平，细分支零增量 |
| 3 | hifag_a2_coarse_only | 43 | A2 多 seed | 0.7146 (ep47) | 0.6596 | ✅ 完成；早停 ep62 |
| 4 | hifag_a2_coarse_only | 44 | A2 多 seed | 0.7016 (ep22) | 0.6551 | ✅ 完成；早停 ep37 |
| 5 | hifag_a2_coarse_only | 45 | A2 多 seed | 0.7103 (ep49) | 0.6643 | ✅ 完成；早停 ep64 |
| 6 | hifag_a2_coarse_only | 46 | A2 多 seed | 0.6953 (ep1) | 0.6476 | ✅ 完成；早停 ep16；best 在 ep1，训练不稳 |
| 7 | hifag_a3_fine_coarse | 43 | A3 多 seed | 0.7329 (ep5) | 0.6775 | ✅ 完成；早停 ep20 |
| 8 | hifag_a3_fine_coarse | 44 | A3 多 seed | 0.7511 (ep26) | 0.6691 | ✅ 完成；早停 ep41 |
| 9 | hifag_a3_fine_coarse | 45 | A3 多 seed | 0.7333 (ep36) | 0.6652 | ✅ 完成；早停 ep51 |
| 10 | hifag_a3_fine_coarse | 46 | A3 多 seed | 0.7329 (ep21) | 0.6710 | ✅ 完成；早停 ep36 |
| 11 | hifag_a7_full | 42 | A7 加音频（对齐 0.797 基线） | 0.8179 (ep15) | 0.7782 | ✅ 完成；早停 ep30 |
| 12 | hifag_a7_full | 43 | A7 多 seed | 0.7938 (ep23) | 0.7880 | ✅ 完成；早停 ep38；本轮最佳 |
| 13 | hifag_a7_full | 44 | A7 多 seed | 0.7938 (ep3) | 0.7756 | ✅ 完成；早停 ep18；best 在 ep3 |
| 14 | hifag_a7_full | 45 | A7 多 seed | 0.8000 (ep4) | 0.7651 | ✅ 完成；早停 ep19；best 在 ep4 |
| 15 | hifag_a7_full | 46 | A7 多 seed | 0.8129 (ep5) | 0.7713 | ✅ 完成；早停 ep20；best 在 ep5 |
| 16 | hifag_a8_full_xattn | 42 | A8 cross-attn 融合 | 0.8207 (ep41) | 0.7938 | ✅ 完成；早停 ep56 |
| 17 | hifag_a8_full_xattn | 43 | A8 多 seed | 0.7969 (ep17) | 0.7934 | ✅ 完成；早停 ep32 |
| 18 | hifag_a8_full_xattn | 44 | A8 多 seed | 0.7996 (ep16) | 0.7906 | ✅ 完成；早停 ep31 |
| 19 | hifag_a8_full_xattn | 45 | A8 多 seed | 0.8086 (ep2) | 0.7769 | ✅ 完成；早停 ep17 |
| 20 | hifag_a8_full_xattn | 46 | A8 多 seed | 0.8066 (ep13) | 0.7766 | ✅ 完成；早停 ep28 |
| 21 | hifag_a9_fine_audio_xattn | 42 | A9 去粗分支（关键消融） | 0.8074 (ep33) | 0.7783 | ✅ 完成；早停 ep48 |
| 22 | hifag_a9_fine_audio_xattn | 43 | A9 多 seed | 0.8129 (ep28) | 0.7973 | ✅ 完成；早停 ep43；本轮最佳 |
| 23 | hifag_a9_fine_audio_xattn | 44 | A9 多 seed | 0.8004 (ep12) | 0.7630 | ✅ 完成；早停 ep27 |
| 24 | hifag_a9_fine_audio_xattn | 45 | A9 多 seed | 0.8023 (ep19) | 0.7843 | ✅ 完成；早停 ep34 |
| 25 | hifag_a9_fine_audio_xattn | 46 | A9 多 seed | 0.8324 (ep44) | 0.7796 | ✅ 完成；早停 ep59 |
| 26 | hifag_a10_full_xattn_film | 42 | A10 FiLM 层级交互 | 0.8121 (ep32) | 0.7723 | ✅ 完成；早停 ep47 |
| 27 | hifag_a10_full_xattn_film | 43 | A10 多 seed | 0.7949 (ep25) | 0.7819 | ✅ 完成；早停 ep40 |
| 28 | hifag_a10_full_xattn_film | 44 | A10 多 seed | 0.7875 (ep10) | 0.7861 | ✅ 完成；早停 ep25 |
| 29 | hifag_a10_full_xattn_film | 45 | A10 多 seed | 0.7953 (ep2) | 0.7818 | ✅ 完成；早停 ep17 |
| 30 | hifag_a10_full_xattn_film | 46 | A10 多 seed | 0.8082 (ep1) | 0.7900 | ✅ 完成；早停 ep16；best 在 ep1 |
| 31~35 | hifag_a4_coarse_nosym | 42~46 | A4 去对称性特征 | — | — | ⏳ 待跑（预登记） |
| 36~40 | hifag_a5_coarse_geom | 42~46 | A5 只留几何特征 | — | — | ⏳ 待跑（预登记） |
| 41~45 | hifag_a6_coarse_full_edges | 42~46 | A6 全连接边 | — | — | ⏳ 待跑（预登记） |

## 多 seed 汇总（seeds 42~46，2026-07-28）

| 配置 | test AUC (mean ± std) | valid AUC 范围 | 备注 |
|------|----------------------|----------------|------|
| A2 粗图单独 | 0.6599 ± 0.0095 | 0.695 ~ 0.715 | 逐 seed: .6728/.6596/.6551/.6643/.6476 |
| A3 细+粗 | **0.6709 ± 0.0045** | 0.719 ~ 0.751 | 逐 seed: .6714/.6775/.6691/.6652/.6710 |
| A7 细+粗+音频(concat) | 0.7756 ± 0.0085 | 0.794 ~ 0.818 | 逐 seed: .7782/.7880/.7756/.7651/.7713 |
| A8 细+粗+音频(x-attn) | **0.7863 ± 0.0088** | 0.797 ~ 0.821 | 逐 seed: .7938/.7934/.7906/.7769/.7766 |
| A9 细+音频(x-attn,无粗) | 0.7805 ± 0.0123 | 0.800 ~ 0.832 | 逐 seed: .7783/.7973/.7630/.7843/.7796 |
| A10 A8+FiLM 层级交互 | 0.7824 ± 0.0066 | 0.788 ~ 0.812 | 逐 seed: .7723/.7819/.7861/.7818/.7900 |

- 配对（同 seed）A3−A2：-0.0014 / +0.0179 / +0.0140 / +0.0009 / +0.0234，
  5 个 seed 中 4 个为正，均值 +0.011（配对 t≈2.27, p≈0.086，边缘显著）。
- A3 的 seed 间波动（std 0.0045）明显小于 A2（0.0095）——细分支提升了稳定性。
- valid AUC 上 A3 一致更高（0.73+ vs 0.70±），差距比 test 上更清晰。
- A7 vs A3：+0.105，音频模态是主要增量来源，与预期一致。
- A7 vs AFGNN `face_enhanced_focal` 基线 0.797 ± 0.011：**−0.021**，未追平。
  差异候选因素：融合方式（A7 concat vs 基线 cross-attention）、基线训练细节。
- ⚠️ A7 有 3/5 seeds 的 best valid 出现在 ep3~5（早停 ep18~20），
  音频分支收敛很快，需留意是否欠训练（lr/scheduler/patience 或值得复查）。
- 配对（同 seed）A8−A7：+0.0156 / +0.0054 / +0.0150 / +0.0118 / +0.0053，
  **5/5 全为正，均值 +0.0106（配对 t≈4.70, p<0.01）**——cross-attention 融合显著优于 concat。
- A8 vs 基线 0.797 ± 0.011：**−0.011，差距落在基线 1 个 std 内，可认为基本追平**。
- A8 训练轮次明显更健康（best valid 多在 ep13~41，仅 seed 45 在 ep2），
  A7 的"ep3~5 早停"现象基本消失——concat 下音频分支确实欠训练。
- 配对（同 seed）A8−A9（粗分支在完整模态下的净增量）：
  +0.0155 / −0.0039 / +0.0276 / −0.0074 / −0.0030，**仅 2/5 为正，均值 +0.0058，
  t≈0.85（不显著）**。粗分支对均值有小正贡献但不显著；
  对稳定性有贡献（A8 std 0.0088 vs A9 0.0123）。
- 结论：粗分支以 concat/x-attn 并联方式接入时增量有限，
  第二阶段层级交互（fine↔coarse 消息传递）成为放大粗图贡献的关键路径。
- 配对（同 seed）A10−A8（FiLM 层级交互的净效应）：
  −0.0215 / −0.0115 / −0.0045 / +0.0049 / +0.0134，**2/5 为正，均值 −0.0039，
  t≈−0.63（不显著）——FiLM 输入调制未能放大粗图贡献，阴性结果**。
  唯一亮点：seed 间波动进一步收窄（std 0.0066，三轮中最低）。
- A10 阴性结果后，粗图的立足点回到：sanity 强信号 + A2 单独判别力 + 稳定性收益
  （A8 std 0.0088 / A10 std 0.0066 vs A9 std 0.0123）。均值增量路径均未奏效
  （并联 +0.0058 n.s.；FiLM −0.0039 n.s.）。

## A4~A6 消融计划（预登记 2026-07-28，exp_31~exp_45）

均以 A2 coarse-only（0.6599 ± 0.0095）为基底，seeds 42~46，归因最干净。
跑完后做的配对比较（同 seed 配对 t 检验）：

| 对比 | 回答的问题 | 预期依据 |
|------|-----------|---------|
| A2 − A4（去对称性） | 对称性特征在 GNN 里的净贡献 | sanity：sym_* 的 std |d|≈0.5，p~1e-19 |
| A2 − A5（只留几何） | 运动特征（含对称性）的净贡献 | sanity：top 特征全是时序 std，多来自运动组 |
| A6 − A2（全连接 vs 解剖学） | 解剖学边先验是否有用 | 9 节点图小，全连接成本低；800 样本下先验可能正收益 |

解读预案：
- 若 A2−A4 显著为正 → 对称性是粗图独有价值的支柱，论文重点写；
- 若 A2−A5 大幅为正 → 粗图主要靠运动动态（与 sanity 一致），几何是陪衬；
- A6 若无差异 → 保留解剖学边（参数少、叙事好）；若全连接显著更好 → 换默认并复查 A8。

## 基线对照（2026-07-28 核实 AFGNN 配置后修正）

- AFGNN `face_enhanced_focal` Test AUC **0.797 ± 0.011**（5 seeds）是 **face + audio**
  + cross-attention 融合的结果（`use_audio: true`），**不是纯面部基线**。
- AFGNN 纯面部 `face_only_enhanced`（weighted_bce，单 seed）Test AUC **0.657**。
- 因此 A2/A3（无音频）的合理对照是 0.657，两者均已略超；与 0.797 的差距是音频模态带来的。
