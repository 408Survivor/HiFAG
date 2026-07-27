# HiFAG: Hierarchical Facial-Audio Graph Network

HiFAG 在 AFGNN 的 68-landmark 细粒度面部图之上增加 9-region 粗粒度面部图，构成层级化面部建模，用于 D-Vlog 抑郁检测。

- 细粒度分支：复用 AFGNN FaceGNN（不修改 `AFGNN/` 内任何文件）
- 粗粒度分支：9 区域节点时空图，节点特征为手工几何/运动/对称性描述子（零参数）
- 音频分支：复用 AFGNN AudioGNN
- 融合：concat（起步）

详见 `DESIGN.md`。

## 环境

```bash
conda activate DVlog
```

## 快速开始

```bash
cd /home/ltq/DepressionCode/DepGNN/HiFAG
python src/train.py --config experiments/configs/<config>.yaml --seed 42
python src/test.py  --config experiments/configs/<config>.yaml --split test
python -m pytest tests/ -q
```
