# HiFAG 实验索引

> 每次训练/评估完成后更新此表。exp_N 编号由 `src/hifag/utils/experiment.py`
> 运行时自动扫描分配（max+1），下表是人工维护的权威索引。
>
> **下一个可用编号：exp_11**

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

## 多 seed 汇总（seeds 42~46，2026-07-28）

| 配置 | test AUC (mean ± std) | valid AUC 范围 | 备注 |
|------|----------------------|----------------|------|
| A2 粗图单独 | 0.6599 ± 0.0095 | 0.695 ~ 0.715 | 逐 seed: .6728/.6596/.6551/.6643/.6476 |
| A3 细+粗 | **0.6709 ± 0.0045** | 0.719 ~ 0.751 | 逐 seed: .6714/.6775/.6691/.6652/.6710 |

- 配对（同 seed）A3−A2：-0.0014 / +0.0179 / +0.0140 / +0.0009 / +0.0234，
  5 个 seed 中 4 个为正，均值 +0.011（配对 t≈2.27, p≈0.086，边缘显著）。
- A3 的 seed 间波动（std 0.0045）明显小于 A2（0.0095）——细分支提升了稳定性。
- valid AUC 上 A3 一致更高（0.73+ vs 0.70±），差距比 test 上更清晰。

## 基线对照（2026-07-28 核实 AFGNN 配置后修正）

- AFGNN `face_enhanced_focal` Test AUC **0.797 ± 0.011**（5 seeds）是 **face + audio**
  + cross-attention 融合的结果（`use_audio: true`），**不是纯面部基线**。
- AFGNN 纯面部 `face_only_enhanced`（weighted_bce，单 seed）Test AUC **0.657**。
- 因此 A2/A3（无音频）的合理对照是 0.657，两者均已略超；与 0.797 的差距是音频模态带来的。
