#!/usr/bin/env python3
"""Draw the expanded SCI study from saved per-input benchmark summaries.

No solver or timing is run. Timing repetitions have already been reduced to a
per-input median by run_expanded.py. Paired speedups are formed per input before
any across-input aggregation. Failures are retained in exported counts; paired
speedup summaries condition on both solvers passing the prescribed checks.
ECDFs use all attempted pairs as the denominator, so failed pairs do not vanish.

Example, after the three result directories are complete::
    python plot_expanded.py

The source CSVs are never modified. Seaborn draws continuous curves without
aggregation, confidence intervals, smoothing, or interpolation. Figure labels
and panel layout use the shared publication_style module. Fixed 6.5-inch canvases; no
bbox='tight' rescaling is used. The whiskers in the geometry figure are observed
10th--90th percentiles across input cases, not confidence intervals.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, LogFormatterMathtext, FuncFormatter

from publication_style import (set_style, tidy, line, panel, save,
    DOUBLE, BLUE, ORANGE, TEAL, GRAY, INK)
import seaborn as sns

ROOT = Path(__file__).resolve().parent
METHODS = (
    ('original-fw', 'Original', GRAY, 'o', '-', 3.2),
    ('blas-fw', 'Batched', BLUE, 's', '-', 3.2),
    ('newton-cheap', 'Newton–quadratic', ORANGE, 'D', '-', 2.8),
    ('newton-fw', 'Newton–FW', TEAL, 'o', (0, (4, 2)), 4.2),
)
CONTRASTS = (
    ('matrix', 'Matrix evaluation', BLUE, 's', '-'),
    ('newton', 'Newton candidates', TEAL, '^', (0, (4, 2))),
    ('complete', 'Complete method', ORANGE, 'o', '-'),
)
REGIMES = ('interior', 'strict_boundary', 'flat_nonunique', 'ill_conditioned')
REGIME_LABELS = ('Interior', 'Boundary', 'Nonunique', 'Ill-\nconditioned')


def method_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 3, 2]
    fig.legend([handles[j] for j in order], [labels[j] for j in order],
        loc='upper center', bbox_to_anchor=(.53, .99), ncol=4,
        columnspacing=1.7, handlelength=2.1)


def bool_column(frame, name):
    if frame[name].dtype != bool:
        frame[name] = frame[name].astype(str).str.lower().eq('true')


def load_tables(results, suites):
    per_case, pairs, provenance = [], [], []
    for suite in suites:
        directory = results / suite
        for filename in ('per_case.csv', 'paired.csv', 'completion.json'):
            if not (directory / filename).is_file():
                raise FileNotFoundError(f'Incomplete suite: {directory / filename}')
        a = pd.read_csv(directory / 'per_case.csv')
        b = pd.read_csv(directory / 'paired.csv')
        for name in ('success', 'reference_resolved'):
            bool_column(a, name)
        bool_column(b, 'paired_success')
        a['suite'] = suite
        b['suite'] = suite
        if a.duplicated(['case_id', 'method', 'tolerance']).any():
            raise ValueError(f'Duplicate per-input records in {suite}')
        if b.duplicated(['case_id', 'contrast', 'tolerance']).any():
            raise ValueError(f'Duplicate paired records in {suite}')
        per_case.append(a)
        pairs.append(b)
        provenance.append(dict(suite=suite,
            completion=json.loads((directory / 'completion.json').read_text()),
            quality=json.loads((directory / 'quality.json').read_text())))
    a = pd.concat(per_case, ignore_index=True)
    b = pd.concat(pairs, ignore_index=True)
    if a.duplicated(['case_id', 'method', 'tolerance']).any():
        raise ValueError('The selected suites overlap; inputs must not be double counted.')
    return a, b, provenance


def group_pairs(frame, keys):
    """Aggregate ratios only after matching cases; retain every attempted count."""
    records = []
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        if len(keys) == 1:
            key = (key,)
        rec = dict(zip(keys, key))
        valid = group[group.paired_success & np.isfinite(group.paired_speedup)
                      & (group.paired_speedup > 0)]
        rec.update(attempted_pairs=len(group), valid_pairs=len(valid),
            failed_pairs=len(group)-len(valid),
            median_speedup=valid.paired_speedup.median(),
            p10_speedup=valid.paired_speedup.quantile(.10),
            p90_speedup=valid.paired_speedup.quantile(.90),
            fraction_faster=(valid.paired_speedup > 1).mean() if len(valid) else np.nan,
            mean_checks_saved=valid.checks_saved.mean(),
            p10_checks_saved=valid.checks_saved.quantile(.10),
            p90_checks_saved=valid.checks_saved.quantile(.90),
            fraction_checks_saved=(valid.checks_saved > 0).mean() if len(valid) else np.nan)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def valid_pair_note(table):
    if not len(table):
        return 'No paired data.'
    if (table.valid_pairs == table.attempted_pairs).all():
        counts = sorted(table.attempted_pairs.unique())
        if len(counts) == 1:
            return f'All {counts[0]} pairs per setting passed the accuracy checks.'
        return 'All pairs passed the accuracy checks.'
    valid = f'{int(table.valid_pairs.min())}--{int(table.valid_pairs.max())}'
    total = f'{int(table.attempted_pairs.min())}--{int(table.attempted_pairs.max())}'
    return f'Valid pairs per setting: {valid} of {total}; failed-pair counts accompany the data.'


def runtime_figure(per_case, out):
    data = per_case[per_case.panel.eq('recorded')].copy()
    if data.empty:
        return
    rows = []
    for key, a in data.groupby(['dataset', 'method', 'tolerance']):
        # Keep all attempted-time values; failed solves are separately counted.
        rows.append(dict(zip(['dataset', 'method', 'tolerance'], key)) | dict(
            attempted_cases=len(a), successful_cases=int(a.success.sum()),
            finite_times=int(np.isfinite(a.end_to_end_seconds).sum()),
            median_ms=1000*a.end_to_end_seconds.median(),
            p10_ms=1000*a.end_to_end_seconds.quantile(.1),
            p90_ms=1000*a.end_to_end_seconds.quantile(.9)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / 'runtime_plot_data.csv', index=False)
    datasets = sorted(int(v) for v in tab.dataset.unique())
    if len(datasets) != 4:
        raise ValueError(f'Four datasets required for the 2x2 runtime figure; got {datasets}')
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 3.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=.095, right=.98, bottom=.16, top=.82, wspace=.20, hspace=.48)
    max_ms = tab.median_ms.max()
    for k, (ax, dataset) in enumerate(zip(axes.flat, datasets)):
        here = tab[tab.dataset.eq(dataset)]
        for method, label, color, marker, linestyle, size in METHODS:
            a = here[here.method.eq(method)].sort_values('tolerance', ascending=False)
            line(ax,a.tolerance,a.median_ms,color=color,marker=marker,ls=linestyle,
                 ms=size,open_marker=method=='newton-fw',label=label,
                 zorder=4 if method=='newton-fw' else 3)
            if method=='newton-fw':ax.lines[-1].set_markerfacecolor('none')
        tidy(ax, logx=True)
        ax.set_xlabel('')
        ax.set_xlim(.0145, .0000069)
        ax.set_ylim(0, max_ms*1.10)
        ax.set_xticks([1e-2, 1e-3, 1e-4, 1e-5])
        ax.set_yticks([0,2,4,6])
        panel(ax,chr(97+k),f'Dataset {dataset}',x=-.15)
        if (here.successful_cases < here.attempted_cases).any():
            ax.text(.02,.96,'Accuracy failures: see data',transform=ax.transAxes,
                    va='top',fontsize=7,color=ORANGE)
    for ax in axes[:,0]:ax.set_ylabel('Total time (ms)')
    fig.supxlabel(r'Objective tolerance, $\varepsilon$',y=.025,fontsize=8.5)
    method_legend(fig,axes[0,0])
    save(fig, out, 'runtime_expanded')


def scaling_figure(pairs, out, tolerance):
    source = pairs[pairs.panel.eq('scaling') & np.isclose(pairs.tolerance, tolerance)
                   & pairs.contrast.isin([c[0] for c in CONTRASTS])]
    if source.empty:
        return
    tab = group_pairs(source, ['mode', 'd', 's', 'contrast', 'tolerance'])
    tab.to_csv(out / 'scaling_plot_data.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 3.05), sharey=True)
    fig.subplots_adjust(left=.095,right=.98,bottom=.17,top=.82,wspace=.20,hspace=.50)
    values=tab.median_speedup[np.isfinite(tab.median_speedup)]
    ymin,ymax=min(.75,values.min()*.87),max(1.4,values.max()*1.18)
    yticks=[2.**i for i in range(-3,12) if ymin<=2.**i<=ymax]
    for row,mode in enumerate(('trace','logdet')):
        for col,(axis,constant,xlabel,label) in enumerate((
            ('d','s',r'State dimension, $d$',r'$s=5$'),
            ('s','d',r'Number of estimates, $s$',r'$d=25$'))):
            ax=axes[row,col]
            data=tab[(tab['mode']==mode)&(tab[constant]==(5 if constant=='s' else 25))]
            for contrast,text,color,marker,linestyle in CONTRASTS:
                a=data[data.contrast.eq(contrast)].sort_values(axis)
                line(ax,a[axis],a.median_speedup,color=color,marker=marker,ls=linestyle,
                     ms=3.3,label=text)
            tidy(ax)
            ax.set_yscale('log',base=2);ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylim(ymin,ymax);ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
            ax.axhline(1,color='#C5C5C5',lw=.65,ls=(0,(2,2)),zorder=1)
            ticks=sorted(data[axis].unique())
            ax.set_xscale('log',base=2);ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks(ticks,[str(int(v)) for v in ticks])
            if len(ticks)>1:ax.set_xlim(min(ticks)*.84,max(ticks)*1.15)
            if row==1:ax.set_xlabel(xlabel)
            else:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel('')
            name='Trace' if mode=='trace' else 'Log-determinant'
            panel(ax,chr(97+row*2+col),f'{name}, {label}',x=-.15)
        axes[row,0].set_ylabel('Paired speedup')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.99),
               ncol=3,columnspacing=2,handlelength=2)
    save(fig,out,'scaling_expanded')


def geometry_figure(pairs, out, tolerances=(1e-3, 1e-5)):
    source = pairs[pairs.panel.eq('geometry') & pairs.contrast.eq('certificate')
                   & pairs.tolerance.map(lambda x: any(np.isclose(x, t) for t in tolerances))]
    if source.empty:
        return
    tab = group_pairs(source, ['mode', 'regime', 'tolerance'])
    tab.to_csv(out / 'geometry_plot_data.csv', index=False)
    fig,axes=plt.subplots(2,2,figsize=(DOUBLE,3.15))
    fig.subplots_adjust(left=.095,right=.98,bottom=.20,top=.82,wspace=.33,hspace=.48)
    colors,markers=[BLUE,ORANGE],['o','s']
    for row,mode in enumerate(('trace','logdet')):
        mode_data=tab[tab['mode'].eq(mode)]
        for j,tol in enumerate(tolerances):
            a=mode_data[np.isclose(mode_data.tolerance,tol)].set_index('regime').reindex(REGIMES)
            x=np.arange(len(REGIMES))+(j-.5)*.19
            mid,lo,hi=a.median_speedup.to_numpy(),a.p10_speedup.to_numpy(),a.p90_speedup.to_numpy()
            valid=np.isfinite(mid)&np.isfinite(lo)&np.isfinite(hi)
            axes[row,0].errorbar(x[valid],mid[valid],
                yerr=np.vstack([np.maximum(0,mid[valid]-lo[valid]),np.maximum(0,hi[valid]-mid[valid])]),
                fmt=markers[j],color=colors[j],capsize=2,capthick=.7,elinewidth=.85,
                ms=3.8,mew=.7,label=rf'$\varepsilon=10^{{{int(round(np.log10(tol)))}}}$')
            sns.scatterplot(x=x,y=a.mean_checks_saved.to_numpy(),ax=axes[row,1],
                color=colors[j],marker=markers[j],s=15,edgecolor=colors[j],
                linewidth=.7,legend=False,zorder=3)
        for col in range(2):
            ax=axes[row,col];tidy(ax)
            ax.set_xlim(-.45,3.45);ax.set_xticks(range(4),REGIME_LABELS)
            ax.tick_params(axis='x',labelsize=7.6)
            ax.set_xlabel('')
            if row==0:ax.tick_params(axis='x',labelbottom=False)
            name='Trace' if mode=='trace' else 'Log-determinant'
            panel(ax,chr(97+row*2+col),name,x=-.17)
        axes[row,0].axhline(1,color='#C5C5C5',ls=(0,(2,2)),lw=.65)
        axes[row,1].axhline(0,color='#C5C5C5',ls=(0,(2,2)),lw=.65)
        axes[row,0].set_ylabel('Paired speedup')
        axes[row,1].set_ylabel('Checks saved')
    low,high=min(1.,tab.p10_speedup.min()),max(1.,tab.p90_speedup.max())
    pad=max(.06,.12*(high-low))
    for ax in axes[:,0]:ax.set_ylim(max(0,low-pad),high+pad)
    lowc,highc=min(0.,tab.mean_checks_saved.min()),max(0.,tab.mean_checks_saved.max())
    for ax in axes[:,1]:ax.set_ylim(lowc-.1,highc+max(.15,.15*(highc-lowc)))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.99),
               ncol=2,columnspacing=3,handlelength=2)
    save(fig,out,'geometry_expanded')


def ecdf_figure(pairs, out, tolerance=1e-3):
    source = pairs[pairs.panel.eq('recorded') & np.isclose(pairs.tolerance, tolerance)
                   & pairs.contrast.isin(['complete', 'certificate'])]
    if source.empty:
        return
    datasets = sorted(int(v) for v in source.dataset.unique())
    palette = [BLUE, ORANGE, TEAL, GRAY]
    styles = ['-', (0, (4, 2.2)), '-.', ':']
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.35), sharey=True)
    fig.subplots_adjust(left=.095, right=.98, bottom=.23, top=.78, wspace=.23)
    for j, contrast in enumerate(['complete', 'certificate']):
        ax = axes[j]
        for k, dataset in enumerate(datasets):
            group = source[source.dataset.eq(dataset) & source.contrast.eq(contrast)]
            valid = group[group.paired_success & np.isfinite(group.paired_speedup)
                          & (group.paired_speedup > 0)]
            xs = np.sort(valid.paired_speedup.to_numpy())
            ys = np.arange(1, len(xs)+1)/len(group)
            if len(xs):
                line(ax,xs,ys,color=palette[k],ls=styles[k],lw=1.25,marker=None,
                     drawstyle='steps-post',label=f'Dataset {dataset}')
            for x, y in zip(xs, ys):
                rows.append(dict(dataset=dataset, contrast=contrast, tolerance=tolerance,
                    speedup=x, cumulative_fraction=y, valid_pairs=len(valid), attempted_pairs=len(group)))
        tidy(ax)
        ax.axvline(1, color=GRAY, ls=':', lw=.8)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('Paired speedup')
        panel(ax,chr(97+j),'Complete method' if j==0 else 'Quadratic stopping',x=-.15)
        finite = source[source.contrast.eq(contrast) & source.paired_success].paired_speedup
        xmax = max(1.15, finite.max()*1.025) if len(finite) else 2.
        ax.set_xlim(0, xmax)
    axes[0].set_ylabel('Fraction of attempted pairs')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.99),
               ncol=4,handlelength=2,columnspacing=1.8)
    pd.DataFrame(rows).to_csv(out / 'speedup_ecdf_plot_data.csv', index=False)
    save(fig, out, 'speedup_ecdf_expanded')


def scaling_absolute_figure(per_case, out, tolerance):
    """Companion to the paired-gain figure: measured total time, not ratios."""
    source = per_case[per_case.panel.eq('scaling')
                      & np.isclose(per_case.tolerance, tolerance)]
    if source.empty:
        return
    keys = ['mode', 'd', 's', 'method', 'tolerance']
    rows = []
    for key, a in source.groupby(keys):
        rows.append(dict(zip(keys, key)) | dict(
            attempted_cases=len(a), successful_cases=int(a.success.sum()),
            finite_times=int(np.isfinite(a.end_to_end_seconds).sum()),
            median_ms=1000*a.end_to_end_seconds.median(),
            p10_ms=1000*a.end_to_end_seconds.quantile(.1),
            p90_ms=1000*a.end_to_end_seconds.quantile(.9)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / 'scaling_absolute_plot_data.csv', index=False)
    fig,axes=plt.subplots(2,2,figsize=(DOUBLE,3.15),sharey=True)
    fig.subplots_adjust(left=.095,right=.98,bottom=.17,top=.82,wspace=.20,hspace=.50)
    finite=tab.median_ms[np.isfinite(tab.median_ms)&(tab.median_ms>0)]
    if finite.empty:raise ValueError('No positive finite scaling timings to plot.')
    low,high=finite.min()*.80,finite.max()*1.25
    for row,mode in enumerate(('trace','logdet')):
        for col,(axis,constant,xlabel,label) in enumerate((
            ('d','s',r'State dimension, $d$',r'$s=5$'),
            ('s','d',r'Number of estimates, $s$',r'$d=25$'))):
            ax=axes[row,col]
            data=tab[tab['mode'].eq(mode)&tab[constant].eq(5 if constant=='s' else 25)]
            for method,text,color,marker,linestyle,size in METHODS:
                a=data[data.method.eq(method)].sort_values(axis)
                line(ax,a[axis],a.median_ms,color=color,marker=marker,ls=linestyle,
                     ms=size,open_marker=method=='newton-fw',label=text,
                     zorder=4 if method=='newton-fw' else 3)
                if method=='newton-fw':ax.lines[-1].set_markerfacecolor('none')
            tidy(ax,logy=True);ax.set_ylim(low,high)
            ticks=sorted(data[axis].unique())
            ax.set_xscale('log',base=2);ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks(ticks,[str(int(v)) for v in ticks])
            if len(ticks)>1:ax.set_xlim(min(ticks)*.84,max(ticks)*1.15)
            if row==1:ax.set_xlabel(xlabel)
            else:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel('')
            name='Trace' if mode=='trace' else 'Log-determinant'
            panel(ax,chr(97+row*2+col),f'{name}, {label}',x=-.15)
        axes[row,0].set_ylabel('Total time (ms)')
    method_legend(fig,axes[0,0])
    save(fig,out,'scaling_absolute_expanded')


def breakdown_table(per_case, out):
    """Export a nonadditive median table; certificate cost is within the solver.

    Repetition medians were taken independently for each timing column upstream.
    Therefore neither component medians nor their across-input medians are
    guaranteed to add to the median end-to-end time. The export names preserve
    that distinction and no stacked chart makes a false additive claim.
    """
    keys = ['suite', 'panel', 'regime', 'd', 's', 'mode', 'dataset', 'method', 'tolerance']
    rows = []
    columns = ['construction_seconds', 'solver_seconds', 'fusion_seconds',
               'end_to_end_seconds', 'certificate_seconds', 'oracle_solver_seconds']
    for key, a in per_case.groupby(keys, dropna=False):
        rec = dict(zip(keys, key)) | dict(attempted_cases=len(a),
            successful_cases=int(a.success.sum()), failed_cases=int((~a.success).sum()))
        for name in columns:
            stem = name.removesuffix('_seconds')
            rec[f'median_{stem}_ms'] = 1000*a[name].median()
            rec[f'finite_{stem}_cases'] = int(np.isfinite(a[name]).sum())
        fraction = a.certificate_seconds/a.solver_seconds.where(a.solver_seconds > 0)
        rec['median_per_input_certificate_fraction_of_solver'] = fraction.median()
        rec['certificate_cost_is_included_in_solver'] = True
        rec['component_medians_are_not_additive'] = True
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out / 'cost_breakdown_data.csv', index=False)
    (out / 'cost_breakdown_notes.txt').write_text(
        'All times are milliseconds. Each input first uses the median of repeated timings.\n'
        'The table then reports medians across input cases in each listed setting.\n'
        'Construction, solver, and fusion are sequential timed stages, but their separately\n'
        'reduced medians need not sum to the separately reduced total-time median.\n'
        'Certificate and oracle solver costs are included within solver time; do not add them.\n'
        'Every attempted input is retained; success and finite-time counts accompany each row.\n'
        'Timing repetitions and alternative initializations are not independent trajectories.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    parser.add_argument('--output', type=Path, default=ROOT/'figures')
    parser.add_argument('--suites', nargs='+', default=['synthetic','recorded_existing','recorded_additional'])
    plot_names = ['runtime','scaling','geometry','ecdf','scaling_absolute','breakdown']
    parser.add_argument('--only', nargs='+', choices=plot_names, default=plot_names)
    parser.add_argument('--scale-tolerance', type=float, default=1e-3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()
    per_case, pairs, provenance = load_tables(args.results, args.suites)
    if 'runtime' in args.only:
        runtime_figure(per_case, args.output)
    if 'scaling' in args.only:
        scaling_figure(pairs, args.output, args.scale_tolerance)
    if 'geometry' in args.only:
        geometry_figure(pairs, args.output)
    if 'ecdf' in args.only:
        ecdf_figure(pairs, args.output)
    if 'scaling_absolute' in args.only:
        scaling_absolute_figure(per_case, args.output, args.scale_tolerance)
    if 'breakdown' in args.only:
        breakdown_table(per_case, args.output)
    quality = per_case.groupby(['suite','panel','mode','method','tolerance'],as_index=False).agg(
        attempted_cases=('case_id','size'),successful_cases=('success','sum'),
        unresolved_references=('reference_resolved',lambda a:int((~a).sum())))
    quality.to_csv(args.output/'figure_quality_counts.csv', index=False)
    (args.output/'figure_provenance.json').write_text(json.dumps(dict(
        source_suites=provenance, scale_tolerance=args.scale_tolerance,
        reductions='per-input median of timing repetitions, then across-input statistics',
        paired_summaries='successful pairs only, with attempted and failed counts retained',
        geometry_whiskers='observed 10th and 90th percentiles, not confidence intervals',
        ecdf_denominator='all attempted pairs; failed pairs treated as beyond the finite axis',
        width_inches=DOUBLE, formats=['pdf','svg','450 dpi png'],
        styling='Seaborn ticks/paper; Nature-inspired compact sans-serif panels; original statistics unchanged',
        continuous_curves='seaborn.lineplot(estimator=None,errorbar=None,sort=False); no interpolation or smoothing',
        geometry='categorical dot-whisker display; categories are not joined' ), indent=2))
    print(f'Figures and exact plotting tables written to {args.output}')


if __name__ == '__main__':
    main()
