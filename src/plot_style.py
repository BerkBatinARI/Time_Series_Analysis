# src/plot_style.py
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def apply_finance_style() -> None:
    """
    Apply a clean, professional matplotlib style suitable for finance plots.
    Centralizing style keeps all figures consistent across scripts.
    """
    plt.style.use("default")

    mpl.rcParams.update(
        {
            # Figure + export
            "figure.figsize": (12, 5.8),
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            # Typography
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            # Axes + grid
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Lines
            "lines.linewidth": 1.6,
            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )


def format_time_axis(ax: plt.Axes) -> None:
    """Readable date formatting for long daily time series."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.margins(x=0.01)


def add_subtitle(ax: plt.Axes, subtitle: str) -> None:
    """Small subtitle under the main title (left-aligned)."""
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        alpha=0.85,
    )