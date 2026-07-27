# HiFAG 实验索引

> 每次训练/评估完成后更新此表。exp_N 编号由 `src/hifag/utils/experiment.py`
> 运行时自动扫描分配（max+1），下表是人工维护的权威索引。
>
> **下一个可用编号：exp_1**

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
| —   | —    | —    | —    | —              | —        | 尚无已完成实验 |
