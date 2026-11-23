"""Create the bilingual reporting inputs from completed frozen benchmark CSVs."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parent
suites=['synthetic','recorded_existing','recorded_additional']

def read_csv(path):return list(csv.DictReader(path.open()))

def main():
    summaries=[];pairs=[];quality=[]
    for suite in suites:
        p=ROOT/'results'/suite
        summaries+=read_csv(p/'summary.csv');pairs+=read_csv(p/'paired_summary.csv')
        quality.append(dict(suite=suite,**json.loads((p/'quality.json').read_text())))
    def select(rows,**fields):
        found=[r for r in rows if all((abs(float(r[k])-v)<1e-12 if isinstance(v,(int,float)) else r[k]==v) for k,v in fields.items())]
        assert len(found)==1,(fields,len(found))
        return found[0]
    datasets=[]
    for ds in [2,3,4,5]:
        ss={m:select(summaries,dataset=ds,method=m,tolerance=.001) for m in ['original-fw','blas-fw','newton-fw','newton-cheap']}
        pp={k:select(pairs,dataset=ds,contrast=k,tolerance=.001) for k in ['matrix','newton','certificate','complete']}
        datasets.append(dict(dataset=ds,cases=int(ss['original-fw']['cases']),methods=ss,contrasts=pp))
    result=dict(quality=quality,datasets=datasets,all_rows=sum(q['rows'] for q in quality),all_failures=sum(q['failed'] for q in quality))
    (ROOT/'PAPER_NUMBERS.json').write_text(json.dumps(result,indent=2))
    lines=['# SCI / TSP 扩展实验结果','',f'本轮完成 **{result["all_rows"]:,} 次求解**；独立数值精度检查未通过数为 **{result["all_failures"]}**。所有正式输入、原始记录、失败状态、参考区间和绘图数据均已保存。','',
      '## 1. 覆盖范围','',
      '- 240组新合成协方差，分别评价trace和logdet，共480个目标问题。规模面板8种(d,s)组合×20组输入；几何面板4类×20组输入。',
      '- 4条真实记录：D2、D3、D4、D5，共1970个输入实例。D3包含同一记录的3种初始化，不计作3条记录。',
      '- 实测容差由3档扩展至7档；合成采用3档。所有在线方法均匀初始化，每组3次计时重复。182760是求解次数，不是独立样本量。','',
      '## 2. 真实记录：完整时间与加速','',
      '下表采用ε=1e-3。时间为逐输入3次计时的中位数再跨输入取中位数；加速比为逐输入配对比值的中位数。P10—P90是输入间分布，不是置信区间。','',
      '| 记录 | 输入数 | 原实现/ms | 完整实现/ms | 配对加速 | P10—P90 | 加速输入比例 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for d in datasets:
        p=d['contrasts']['complete'];s=d['methods'];lines.append(f"| D{d['dataset']} | {d['cases']} | {float(s['original-fw']['median_end_to_end_seconds'])*1e3:.3f} | {float(s['newton-cheap']['median_end_to_end_seconds'])*1e3:.3f} | {float(p['median_speedup']):.3f}× | {float(p['p10_speedup']):.2f}—{float(p['p90_speedup']):.2f} | {float(p['fraction_faster'])*100:.2f}% |")
    lines+=['','各组件贡献（对应相邻实现之间的配对比）：','','| 记录 | 批量矩阵 / 原实现 | Newton / SLSQP | 二次证书 / FW |','| --- | ---: | ---: | ---: |']
    for d in datasets:lines.append('| D'+str(d['dataset'])+' | '+' | '.join(f"{float(d['contrasts'][k]['median_speedup']):.3f}×" for k in ['matrix','newton','certificate'])+' |')
    lines+=['','完整实现的加速主要来自矩阵计算和候选更新。固定Newton后，证书可以改善剩余误差的评估，但不保证减少迭代或总时间。各组件的中位加速比不能简单相乘得到整体中位加速比。','',
      '## 3. 规模与几何的结果','',
      '- d=128、s=5、ε=1e-3：trace与logdet配对加速分别为9.38×和9.59×；绝对中位时间分别从251.9降至26.0 ms、259.5降至28.1 ms。',
      '- d=25、s=20：对应加速5.98×和6.35×。规模扩大时，算子重写与Newton更新的收益更突出。',
      '- 内部、平坦和病态几何组在ε=1e-3下整体中位加速1.19—2.21×；严格最优顶点的trace组为0.83×，logdet为1.02×。容易的边界问题上不能承诺整体更快。',
      '- 固定Newton后，内部logdet组在ε=1e-3时二次证书平均节省1次检查，中位加速1.10×；多数其他几何组在该容差没有减少检查次数。','',
      '几何面板具有解析最优点，采用共同矩阵C，作用是隔离机制；一般非交换输入由规模面板提供。平坦最优集是单纯形内的仿射截面，不是单纯形的一个面。所有A_i正定；本次速度实验不实现一般奇异A_i。','',
      '## 4. 精度、完整性和计时口径','',
      '| 套件 | 求解数 | 不通过 | 最大参考区间宽度 | 最大协方差相对差异 | 最大融合均值相对差异 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for q in quality:lines.append(f"| {q['suite']} | {q['rows']:,} | {q['failed']} | {q['max_reference_gap']:.3g} | {q['max_covariance_relative_error']:.3g} | {q['max_fused_relative_error']:.3g} |")
    lines+=['','参考计算逐分支直接求解A_i+w_iB_i，与被测谱分解路径分开。这里是float64数值核查，不是严格区间算术认证。全部计时含新建模型、优化及所有证书调用、融合输出；事后核查不计时。证书和oracle耗时都可能位于solver阶段内部，不能相加。所有算法源文件保持不变，正式套件串行执行。','',
      '## 5. 新记录的估计器诊断','',
      'D4/D5使用完全冻结的历史模型与反馈。D4位置RMSE为4.06 cm、名义95%椭圆覆盖81.62%；D5为64.41 cm、74.88%，其中robot5达到1.355 m。该负面结果完整保留。求解器精度和速度可以通过，而上游非线性估计器仍可能失准。这些输入上的配对计算不能替代新求解器各自推进闭环轨迹的测试。','',
      '## 6. 如何支撑论文','',
      '| 实验 | 论文中的作用 |','| --- | --- |',
      '| 旧同点证书比较与精确示例 | 支撑几何分类、二次误差界的紧致性与最优权重的解释 |',
      '| 七容差×四记录 | 检验精度要求变化后的实际计算成本和序列覆盖 |',
      '| 维度与传感器数扫描 | 说明效率优势随问题规模如何变化，分离矩阵和候选更新贡献 |',
      '| 四类解析几何 | 检验边界、非唯一和病态结构，并呈现无加速或变慢的例外 |',
      '| 配对分布、尾部耗时、成本分解 | 避免只报告总体平均时间，明确哪些输入受益以及成本花在哪里 |',
      '| 独立公式复核 | 确认加速没有通过降低目标精度或改变融合公式换来 |','',
      '当前证据支持“几何与证书理论，加上可复现的高效实现”这一主线。若后续要把“改善在线估计/控制”作为主要贡献，需另做各方法独立反馈的闭环实验；本轮没有将计算测试包装成该结论。']
    (ROOT/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    table=[r'\begin{table}[t]',r'\caption{Paired complete-method speedup at $\varepsilon=10^{-3}$}\label{tab:runtime}',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Dataset & Median & P10 & P90 & Faster (\%)\\',r'\midrule']
    for d in datasets:
        p=d['contrasts']['complete'];table.append('D%d & %.2f & %.2f & %.2f & %.1f\\\\'%(d['dataset'],float(p['median_speedup']),float(p['p10_speedup']),float(p['p90_speedup']),100*float(p['fraction_faster'])))
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'runtime_table.tex').write_text('\n'.join(table)+'\n')
    print(json.dumps({k:result[k] for k in ['all_rows','all_failures']}))
if __name__=='__main__':main()
