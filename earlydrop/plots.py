"""Shared matplotlib style for the report figures (light, print-friendly)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

# Blackboard palette (light-surface steps for print), same hue order as the dashboard.
SERIES = ["#a87a10", "#2f78bd", "#c2413a", "#7c5cc4", "#23896a"]
# Academic states: Struggling, Stable, Strong (always direct-labelled).
STATE = ["#c2413a", "#a87a10", "#2f78bd"]
INK, MUTED, GRID = "#14181f", "#5b6372", "#e6e8ec"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES), "lines.linewidth": 2,
})


def save(fig, name: str) -> str:
    C.FIGS.mkdir(parents=True, exist_ok=True)
    path = C.FIGS / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name
