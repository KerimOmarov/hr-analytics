"""
viz_utils.py
Reusable chart functions for HR Analytics project.
Author: Your Name
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

PALETTE = {'0 — Staying': '#4C9BE8', '1 — Seeking': '#E87B4C'}
COLORS  = {'blue': '#4C9BE8', 'orange': '#E87B4C',
           'green': '#6DB87A', 'purple': '#9B7FD4', 'red': '#E84C4C'}
OUT     = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)


def set_style():
    """Apply consistent rcParams for all project charts."""
    plt.rcParams.update({
        'figure.dpi':        150,
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.titlesize':    13,
        'axes.titleweight':  'bold',
        'axes.labelsize':    11,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'font.family':       'sans-serif',
        'legend.frameon':    False,
    })


def save(fig: plt.Figure, name: str, tight: bool = True):
    """Save figure to outputs/figures/ with consistent settings."""
    path = OUT / name
    fig.savefig(path, bbox_inches='tight' if tight else None, dpi=150)
    print(f"Saved → {path}")


def plot_target_distribution(series: pd.Series,
                              save_name: str = None) -> plt.Figure:
    """Bar chart of binary target class distribution."""
    set_style()
    counts = series.value_counts().sort_index()
    labels = ['Not seeking (0)', 'Seeking (1)']
    colors = [COLORS['blue'], COLORS['orange']]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors,
                  edgecolor='white', width=0.45)

    for bar, v in zip(bars, counts.values):
        pct = v / counts.sum() * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                f'{v:,}\n({pct:.1f}%)', ha='center', fontsize=10)

    ax.set_title("Target class distribution")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    if save_name:
        save(fig, save_name)
    return fig


def plot_stacked_bar(df: pd.DataFrame,
                     feature: str,
                     target_col: str = 'target_label',
                     title: str = None,
                     xlabel: str = None,
                     category_order: list = None,
                     figsize: tuple = (12, 5),
                     save_name: str = None) -> tuple:
    """
    Stacked proportional bar chart of feature vs. target class.
    Returns (fig, ax).
    """
    set_style()
    data = (df.groupby([feature, target_col])
              .size()
              .unstack(fill_value=0))
    if category_order:
        data = data.reindex(
            [c for c in category_order if c in data.index])
    pct = data.div(data.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=figsize)
    pct.plot(kind='bar', stacked=True, ax=ax,
             color=[PALETTE.get(c, '#888') for c in pct.columns],
             edgecolor='white', linewidth=0.4, width=0.7)
    ax.set_title(title or f"{feature} vs. attrition intent")
    ax.set_xlabel(xlabel or feature.replace('_', ' ').title())
    ax.set_ylabel("Proportion (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title="Status", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    if save_name:
        save(fig, save_name)
    return fig, ax


def plot_kde_comparison(df: pd.DataFrame,
                         feature: str,
                         target_col: str = 'target_label',
                         title: str = None,
                         save_name: str = None) -> plt.Figure:
    """KDE + boxplot side-by-side for a continuous feature vs. target."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.kdeplot(data=df, x=feature, hue=target_col,
                palette=PALETTE, fill=True, alpha=0.45, ax=axes[0])
    axes[0].set_title(f"{feature} — density by class")
    axes[0].set_xlabel(feature.replace('_', ' ').title())
    axes[0].set_ylabel("Density")

    sns.boxplot(data=df, x=target_col, y=feature,
                palette=PALETTE, ax=axes[1], width=0.45, linewidth=1.2)
    axes[1].set_title(f"{feature} — spread by class")
    axes[1].set_xlabel("")

    plt.suptitle(title or f"{feature} vs. attrition status",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_name:
        save(fig, save_name)
    return fig


def plot_missing_heatmap(df: pd.DataFrame,
                          title: str = "Missing value map",
                          figsize: tuple = (12, 4),
                          save_name: str = None) -> plt.Figure:
    """Heatmap showing where missing values occur across the DataFrame."""
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False,
                cmap='viridis', ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    if save_name:
        save(fig, save_name)
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame,
                              labels: list = None,
                              title: str = "Feature correlation matrix",
                              save_name: str = None) -> plt.Figure:
    """Lower-triangle correlation heatmap."""
    set_style()
    if labels:
        corr = corr.copy()
        corr.columns = labels
        corr.index   = labels

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 9})
    ax.set_title(title)
    plt.tight_layout()
    if save_name:
        save(fig, save_name)
    return fig


def plot_confusion_matrix(y_true, y_pred,
                           title: str = "Confusion matrix",
                           save_name: str = None) -> plt.Figure:
    """Annotated confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    set_style()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted: Staying', 'Predicted: Seeking'],
                yticklabels=['Actual: Staying', 'Actual: Seeking'],
                ax=ax, linewidths=0.5, cbar=False)
    ax.set_title(title)
    plt.tight_layout()
    if save_name:
        save(fig, save_name)
    return fig


def plot_risk_tier_distribution(risk_series: pd.Series,
                                 save_name: str = None) -> plt.Figure:
    """Bar chart of employee risk tier counts."""
    set_style()
    tier_colors = {
        'High Risk':   '#E84C4C',
        'Medium Risk': '#E8A44C',
        'Low Risk':    '#4C9BE8'
    }
    counts = risk_series.value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index,
                  counts.values,
                  color=[tier_colors.get(t, '#888') for t in counts.index],
                  edgecolor='white', linewidth=0.5, width=0.5)

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 15,
                f'n={count:,}', ha='center', fontsize=10)

    ax.set_title("Employee risk tier distribution — validation set")
    ax.set_ylabel("Number of employees")
    plt.tight_layout()
    if save_name:
        save(fig, save_name)
    return fig
