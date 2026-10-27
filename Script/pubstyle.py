#!/usr/bin/env python3
"""
pubstyle.py — shared publication style for TMI figures. Import and call set_style().
Vector output (PDF), large legible fonts, colorblind-safe palette, tight layout.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# colorblind-safe (Okabe-Ito) — consistent roles across all figures
C = {
    "flow":   "#0072B2",   # blue  — our model
    "ddpm":   "#D55E00",   # vermillion — diffusion
    "copy":   "#999999",   # grey  — baseline
    "accent": "#009E73",   # green — calibrated / good
    "warn":   "#CC79A7",   # magenta — raw / miscalibrated
    "ideal":  "#000000",   # black — diagonal / reference
}

def set_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,          # TrueType (editable, embeddable) — required by many journals
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,             # base — large enough at column width
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })
