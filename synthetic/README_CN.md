# 正文合成证书与附录数值检查

保留当前论文 V.B、I.A、I.B、G 对应的输入、代码与结果：

- `code/run_main.py`：96 组矩阵、192 目标、960 候选的证书紧度。
- `code/run_high_precision.py`：基于原始 A/B 的 36 点 80 位独立复核。
- `code/run_stress.py`：32 组奇异/病态输入与下模型探测。
- `code/run_exact.py`：7 组输入、两种准则的 14 个完整精确检查流程。
- `source/`：上述流程使用的原精确和数值实现。

运行前复制本目录；这些入口会写回 `data/`。`run_exact.py` 仅移除了论文没有报告的另一组指定近最优候选测试，保留完整流程的计算。论文图 2 和表 I 从 `certificate_results.csv` 汇总；归档 `certificate_summary.csv` 用于重绘核对。
