"""Seaborn publication theme; fixed-size, vector-safe scientific figures.

Stylistically inspired by compact Nature-family figures, not a journal template.
Precomputed statistics are plotted without automatic aggregation or intervals.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, LogFormatterMathtext
from matplotlib.lines import Line2D
import seaborn as sns

SINGLE, DOUBLE = 3.125, 6.5
BLUE, ORANGE, TEAL, GRAY = '#3C5488', '#E64B35', '#00A087', '#9098A1'
INK, GRID = '#252525', '#E6E6E6'

def set_style():
    sns.set_theme(context='paper', style='ticks', font='Liberation Sans',
        palette=[BLUE, ORANGE, TEAL, GRAY], rc={
        'font.family':'sans-serif',
        'font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
        'mathtext.fontset':'dejavusans',
        'font.size':8.5, 'axes.labelsize':8.5, 'axes.titlesize':8.5,
        'axes.titleweight':'normal', 'legend.fontsize':8,
        'xtick.labelsize':8, 'ytick.labelsize':8,
        'axes.edgecolor':INK, 'axes.labelcolor':INK,
        'text.color':INK, 'xtick.color':INK, 'ytick.color':INK,
        'axes.linewidth':.65, 'axes.grid':False,
        'xtick.direction':'out', 'ytick.direction':'out',
        'xtick.major.size':3, 'ytick.major.size':3,
        'xtick.major.width':.65, 'ytick.major.width':.65,
        'lines.linewidth':1.35, 'lines.markersize':3.5,
        'legend.frameon':False, 'legend.handlelength':2,
        'legend.handletextpad':.55, 'legend.borderpad':0,
        'pdf.fonttype':42, 'ps.fonttype':42,
        'figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white', 'svg.fonttype':'none'})

def tidy(ax, *, logx=False, logy=False):
    sns.despine(ax=ax, top=True, right=True, offset=0, trim=False)
    ax.grid(False)
    ax.tick_params(pad=3)
    if logx:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
    if logy:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

def line(ax, x, y, *, color=BLUE, marker='o', ls='-', ms=3.5,
         open_marker=False, label=None, lw=1.35, zorder=3, **kwargs):
    return sns.lineplot(x=x,y=y,ax=ax,estimator=None,errorbar=None,
        sort=False,legend=False,color=color,linestyle=ls,marker=marker,
        markersize=ms,markerfacecolor='white' if open_marker else color,
        markeredgecolor=color,markeredgewidth=.8,linewidth=lw,
        label=label,zorder=zorder,**kwargs)

def panel(ax, letter, title='', *, x=-.15, y=1.05):
    ax.text(x,y,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',
            ha='left',va='bottom',color=INK,clip_on=False)
    if title: ax.set_title(title,loc='left',pad=9)

def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'{name}.{ext}',dpi=450)
    plt.close(fig)

set_style()
