# 论文冻结模型回放

最终优化输入统一保存在上一级 `inputs/`。本目录保留生成这些输入所用的冻结运动模型、配置、校准结果和数据来源。D4/D5 的参考位置诊断必要数组单独保存在根目录 `diagnostics/data/`，不再重复保存包含 A/B/局部均值的整份银行。

在本目录执行以下命令可核对冻结代码，并从原始记录重新生成 D4/D5：

```
python extend_sequences.py verify
python extend_sequences.py download
python extend_sequences.py replay
python extend_sequences.py export
```

D2 使用 seed 42，D3 使用 seeds 42/43/44，均使用同一份 `calibration_response.json` 和 `source_motion/configs/default.json`。它们的回放入口是 `source_motion/run.py replay`，通过 `--datasets`、`--seeds`、`--data`、`--out`、`--calibration` 和 `--config` 指定；下载入口为 `run.py download --source mirror`。论文优化实验可直接使用已保存矩阵，无需重跑滤波。

原始记录来自历史固定镜像版本，记录于 `EXTENSION_PROTOCOL.json` 与 `raw_provenance/`。原冻结协议保留为来源记录；包内仅提供本文采用的运动响应模型，没有旧模型对比入口。
