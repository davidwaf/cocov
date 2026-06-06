"""
analysis/generate_results.py
-----------------------------
Generate all LaTeX tables and publication-quality figures for the
COCOV multi-backbone thesis chapter.

Usage
-----
    cd /opt/code/ps/cocov
    python analysis/generate_results.py \
        --results_dir /opt/data/cocov/results \
        --output_dir  /opt/data/cocov/analysis

Outputs
-------
    figures/
        fig1_main_auc_grouped.pdf        AUC bar chart — all encoders × methods
        fig2_main_eer_grouped.pdf        EER bar chart — all encoders × methods
        fig3_cocov_vs_static_scatter.pdf COCOV vs Static Enrollment scatter
        fig4_ablation_heatmap.pdf        Ablation component importance heatmap
        fig5_ablation_bars.pdf           Ablation bar chart per encoder
        fig6_cross_dataset_auc.pdf       Cross-dataset AUC (VGGFace2 / CACD / FG-NET)
        fig7_cross_dataset_eer.pdf       Cross-dataset EER
        fig8_updates_bar.pdf             Update counts — selectivity analysis
        fig9_radar.pdf                   Radar chart — COCOV profile per encoder
        fig10_cocov_gain.pdf             COCOV gain over Static Enrollment

    tables/
        tab1_main_results.tex            Main results table (AUC, EER, TAR@1%)
        tab2_cross_dataset.tex           Cross-dataset results
        tab3_ablation.tex                Ablation study table
        tab4_encoder_summary.tex         Encoder configuration summary

Author: David Wafula
Project: COCOV — Continuous Identity Representation Adaptation
"""

import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ENCODERS = {
    "ENC01_FACENET":       {"label": "FaceNet",       "backbone": "InceptionResNetV1", "loss": "Triplet",          "data": "VGGFace2",    "short": "FN"},
    "ENC02_ARCFACE_R50":   {"label": "ArcFace-R50",   "backbone": "IResNet-50",        "loss": "ArcFace",          "data": "WebFace600K", "short": "AF50"},
    "ENC03_ARCFACE_R100":  {"label": "ArcFace-R100",  "backbone": "IResNet-100",       "loss": "ArcFace",          "data": "Glint360K",   "short": "AF100"},
    "ENC04_MOBILEFACENET": {"label": "MobileFaceNet", "backbone": "MobileFaceNet",     "loss": "ArcFace",          "data": "WebFace600K", "short": "MFN"},
    "ENC04_ADAFACE":       {"label": "AdaFace",       "backbone": "IResNet-101",       "loss": "AdaFace",          "data": "MS1MV3",      "short": "ADA"},
}

METHODS = {
    "static":  {"label": "Static",  "color": "#6B7280", "marker": "o"},
    "ols":     {"label": "OLS",     "color": "#F59E0B", "marker": "s"},
    "replay":  {"label": "Replay",  "color": "#3B82F6", "marker": "^"},
    "buffer":  {"label": "Buffer",  "color": "#8B5CF6", "marker": "D"},
    "cocov":   {"label": "COCOV",   "color": "#10B981", "marker": "*"},
}

ABLATION_COMPONENTS = {
    "cocov_full":         "Full COCOV",
    "cocov_no_drift":     "w/o Drift Gate",
    "cocov_no_merge":     "w/o Merge",
    "cocov_no_reviewer":  "w/o Reviewer",
    "cocov_unbounded":    "w/o Bound",
    "cocov_single_proto": "Single Proto.",
}

# Publication style
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.5,
    "savefig.dpi":      300,
})

ENC_COLORS = {
    "ENC01_FACENET":       "#6366F1",
    "ENC02_ARCFACE_R50":   "#F59E0B",
    "ENC03_ARCFACE_R100":  "#10B981",
    "ENC04_MOBILEFACENET": "#EF4444",
    "ENC04_ADAFACE":       "#8B5CF6",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict:
    """Load all result JSONs from the results directory."""
    data = {}
    for enc_dir in sorted(results_dir.iterdir()):
        if not enc_dir.is_dir():
            continue
        enc_code = enc_dir.name
        if enc_code not in ENCODERS:
            continue

        rec = {"enc_code": enc_code}

        # Main aggregated results
        agg = enc_dir / "aggregated_results.json"
        if agg.exists():
            rec["main"] = json.loads(agg.read_text())

        # Ablation
        abl = enc_dir / "ablation_results.json"
        if abl.exists():
            rec["ablation"] = json.loads(abl.read_text())

        # Cross-dataset
        xds = enc_dir / "cross_dataset_results.json"
        if xds.exists():
            rec["cross_dataset"] = json.loads(xds.read_text())

        data[enc_code] = rec
        logger.info(f"Loaded: {enc_code}  "
                    f"main={'main' in rec}  "
                    f"ablation={'ablation' in rec}  "
                    f"cross_dataset={'cross_dataset' in rec}")

    return data


def extract_metric(data, enc_code, method_key, metric, dataset="vggface2"):
    """
    Safely extract mean ± std for a metric.

    JSON structure:
        aggregated_results.json  -> flat: data[method_key][metric_mean]
        cross_dataset_results.json -> data[dataset][method_key][metric_mean]
        ablation_results.json    -> flat: data[ablation_key][metric_mean]

    Metric key mapping:
        "auc"             -> "auc_mean" / "auc_std"
        "eer"             -> "eer_mean" / "eer_std"
        "tar_at_1_percent"-> "tar_at_far1_mean" / "tar_at_far1_std"
        "updates"         -> "total_updates_mean" / "total_updates_std"
    """
    METRIC_MAP = {
        "auc":              "auc",
        "eer":              "eer",
        "tar_at_1_percent": "tar_at_far1",
        "updates":          "total_updates",
    }
    mkey = METRIC_MAP.get(metric, metric)
    try:
        rec = data[enc_code]
        if dataset == "vggface2":
            m = rec["main"][method_key]
        else:
            m = rec["cross_dataset"][dataset][method_key]
        mean = m[f"{mkey}_mean"]
        std  = m.get(f"{mkey}_std", 0.0)
        return mean, std
    except (KeyError, TypeError):
        return None, None


def get_method_key(method_label):
    """Map display label back to JSON key."""
    for k in METHODS:
        if k == method_label or METHODS[k]["label"] == method_label:
            return k
    return method_label


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def savefig(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out)
    fig.savefig(path / name.replace(".pdf", ".png"))
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Main AUC grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_main_auc(data, fig_dir):
    encs = [e for e in ENCODERS if e in data and "main" in data[e]]
    methods = list(METHODS.keys())
    n_enc = len(encs)
    n_meth = len(methods)

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.14
    x = np.arange(n_enc)

    for i, mkey in enumerate(methods):
        means, stds = [], []
        for enc in encs:
            m, s = extract_metric(data, enc, mkey, "auc")
            means.append(m if m is not None else 0)
            stds.append(s if s is not None else 0)

        offset = (i - n_meth / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width,
                      yerr=stds, capsize=3,
                      color=METHODS[mkey]["color"],
                      alpha=0.85, label=METHODS[mkey]["label"],
                      error_kw={"linewidth": 0.8, "alpha": 0.6})

    ax.set_xticks(x)
    ax.set_xticklabels([ENCODERS[e]["label"] for e in encs], fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_title("Verification Performance Across Encoders and Methods (VGGFace2)",
                 fontsize=12, pad=10)
    ax.set_ylim(0.94, 1.005)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.legend(ncols=5, loc="lower right", framealpha=0.9, fontsize=8.5)
    ax.axhline(0.98, color="gray", lw=0.6, ls="--", alpha=0.5)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig1_main_auc_grouped.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Main EER grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_main_eer(data, fig_dir):
    encs = [e for e in ENCODERS if e in data and "main" in data[e]]
    methods = list(METHODS.keys())
    n_enc, n_meth = len(encs), len(methods)

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.14
    x = np.arange(n_enc)

    for i, mkey in enumerate(methods):
        means, stds = [], []
        for enc in encs:
            m, s = extract_metric(data, enc, mkey, "eer")
            means.append((m * 100) if m is not None else 0)
            stds.append((s * 100) if s is not None else 0)

        offset = (i - n_meth / 2 + 0.5) * width
        ax.bar(x + offset, means, width,
               yerr=stds, capsize=3,
               color=METHODS[mkey]["color"],
               alpha=0.85, label=METHODS[mkey]["label"],
               error_kw={"linewidth": 0.8, "alpha": 0.6})

    ax.set_xticks(x)
    ax.set_xticklabels([ENCODERS[e]["label"] for e in encs], fontsize=9)
    ax.set_ylabel("EER (%)")
    ax.set_title("Equal Error Rate Across Encoders and Methods (VGGFace2)",
                 fontsize=12, pad=10)
    ax.legend(ncols=5, loc="upper right", framealpha=0.9, fontsize=8.5)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig2_main_eer_grouped.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — COCOV vs Static scatter
# ─────────────────────────────────────────────────────────────────────────────

def fig_cocov_vs_static_scatter(data, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (metric, label, better) in zip(
        axes,
        [("auc", "AUC", "higher"), ("eer", "EER (%)", "lower")]
    ):
        for enc_code, enc_info in ENCODERS.items():
            if enc_code not in data or "main" not in data[enc_code]:
                continue

            static_m, _ = extract_metric(data, enc_code, "static", metric)
            cocov_m, _  = extract_metric(data, enc_code, "cocov",  metric)

            if static_m is None or cocov_m is None:
                continue

            if metric == "eer":
                static_m *= 100; cocov_m *= 100

            color = ENC_COLORS.get(enc_code, "#555")
            ax.scatter(static_m, cocov_m, s=120,
                       color=color, zorder=5,
                       label=enc_info["label"], edgecolors="white", lw=0.8)
            ax.annotate(enc_info["short"],
                        (static_m, cocov_m),
                        textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, color=color)

        lo = ax.get_xlim()[0]; hi = ax.get_xlim()[1]
        rng = [min(lo, ax.get_ylim()[0]), max(hi, ax.get_ylim()[1])]
        ax.plot(rng, rng, "--", color="gray", lw=0.8, alpha=0.5, label="y = x")
        ax.set_xlabel(f"Static Enrollment {label}")
        ax.set_ylabel(f"COCOV {label}")
        ax.set_title(f"COCOV vs Static — {label}")
        if better == "higher":
            ax.annotate("COCOV\nbetter ↑", xy=(0.05, 0.92),
                        xycoords="axes fraction", fontsize=8, color="#10B981",
                        fontweight="bold")
        else:
            ax.annotate("COCOV\nbetter ↓", xy=(0.05, 0.08),
                        xycoords="axes fraction", fontsize=8, color="#10B981",
                        fontweight="bold")
        ax.legend(fontsize=7.5, ncols=2, loc="lower right")

    fig.suptitle("COCOV vs Static Enrollment — All Encoders (VGGFace2)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig3_cocov_vs_static_scatter.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Ablation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig_ablation_heatmap(data, fig_dir):
    encs = [e for e in ENCODERS if e in data and "ablation" in data[e]]
    abl_keys = list(ABLATION_COMPONENTS.keys())

    matrix = np.zeros((len(abl_keys), len(encs)))

    for j, enc in enumerate(encs):
        abl = data[enc].get("ablation", {})
        full_auc = None
        for k in abl_keys:
            try:
                v = abl[k]["auc_mean"]
                if k == "cocov_full":
                    full_auc = v
                matrix[abl_keys.index(k), j] = v
            except (KeyError, TypeError):
                matrix[abl_keys.index(k), j] = np.nan

    # Normalize each column relative to full COCOV (row 0)
    full_row = matrix[0, :].copy()
    delta = matrix - full_row[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(max(6, len(encs) * 1.5), 4.5))
    im = ax.imshow(delta, cmap="RdYlGn", aspect="auto",
                   vmin=-0.015, vmax=0.005)

    ax.set_xticks(range(len(encs)))
    ax.set_xticklabels([ENCODERS[e]["label"] for e in encs],
                       rotation=20, ha="right")
    ax.set_yticks(range(len(abl_keys)))
    ax.set_yticklabels([ABLATION_COMPONENTS[k] for k in abl_keys])

    # Annotate cells
    for i in range(len(abl_keys)):
        for j in range(len(encs)):
            val = delta[i, j]
            if not np.isnan(val):
                txt = f"{val:+.4f}" if abs(val) > 0.0001 else "0"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7.5, color="black")

    plt.colorbar(im, ax=ax, label="ΔAUC vs Full COCOV",
                 shrink=0.7, pad=0.02)
    ax.set_title("Ablation Study — AUC Difference from Full COCOV",
                 fontsize=12, pad=10)
    ax.axhline(0.5, color="white", lw=1.5)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig4_ablation_heatmap.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Ablation bars per encoder
# ─────────────────────────────────────────────────────────────────────────────

def fig_ablation_bars(data, fig_dir):
    encs = [e for e in ENCODERS if e in data and "ablation" in data[e]]
    abl_keys = list(ABLATION_COMPONENTS.keys())
    colors = ["#10B981", "#EF4444", "#3B82F6", "#F59E0B", "#8B5CF6", "#6B7280"]

    n_enc = len(encs)
    fig, axes = plt.subplots(1, n_enc, figsize=(3.2 * n_enc, 4.5),
                             sharey=True)
    if n_enc == 1:
        axes = [axes]

    for ax, enc in zip(axes, encs):
        abl = data[enc].get("ablation", {})
        means, stds, labels = [], [], []
        for k in abl_keys:
            try:
                means.append(abl[k]["auc_mean"])
                stds.append(abl[k].get("auc_std", 0.0))
            except (KeyError, TypeError):
                means.append(0.0); stds.append(0.0)
            labels.append(ABLATION_COMPONENTS[k])

        bars = ax.barh(range(len(abl_keys)), means, xerr=stds,
                       color=colors, alpha=0.85, capsize=3,
                       error_kw={"linewidth": 0.8})
        ax.set_yticks(range(len(abl_keys)))
        ax.set_yticklabels(labels if ax == axes[0] else [], fontsize=8.5)
        ax.set_title(ENCODERS[enc]["label"], fontsize=10)
        ax.set_xlabel("AUC", fontsize=9)
        ax.set_xlim(0.95, 1.0)
        ax.axvline(means[0], color="#10B981", lw=1, ls="--", alpha=0.6)

    fig.suptitle("Ablation Study — COCOV Components by Encoder",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig5_ablation_bars.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Cross-dataset AUC
# ─────────────────────────────────────────────────────────────────────────────

def fig_cross_dataset(data, fig_dir, metric="auc"):
    datasets = ["vggface2", "cacd", "fgnet"]
    ds_labels = {"vggface2": "VGGFace2", "cacd": "CACD", "fgnet": "FG-NET"}
    encs = [e for e in ENCODERS if e in data]
    methods_plot = ["static", "cocov"]
    mstyles = {"static": ("--", "o"), "cocov": ("-", "*")}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, ds in zip(axes, datasets):
        for mkey in methods_plot:
            vals, labels = [], []
            for enc in encs:
                if ds == "vggface2":
                    m, _ = extract_metric(data, enc, mkey, metric, "vggface2")
                else:
                    m, _ = extract_metric(data, enc, mkey, metric, ds)
                vals.append((m * 100 if metric == "eer" else m)
                             if m is not None else np.nan)
                labels.append(ENCODERS[enc]["short"])

            ls, mk = mstyles[mkey]
            color = METHODS.get(mkey, {"color": "#6B7280"})["color"]
            ax.plot(range(len(encs)), vals,
                    ls=ls, marker=mk, color=color,
                    label=METHODS.get(mkey, {}).get("label", mkey),
                    ms=8, lw=1.8)

        ax.set_xticks(range(len(encs)))
        ax.set_xticklabels([ENCODERS[e]["short"] for e in encs], fontsize=9)
        ax.set_title(ds_labels[ds], fontsize=11)
        ylabel = "AUC" if metric == "auc" else "EER (%)"
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8.5)

    metric_title = "AUC" if metric == "auc" else "EER"
    fig.suptitle(f"Cross-Dataset Evaluation — {metric_title}: VGGFace2 / CACD / FG-NET",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fname = f"fig6_cross_dataset_{metric}.pdf"
    savefig(fig, fig_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Update counts (selectivity)
# ─────────────────────────────────────────────────────────────────────────────

def fig_updates(data, fig_dir):
    encs = [e for e in ENCODERS if e in data and "main" in data[e]]
    methods_with_updates = [
        "ols", "replay",
        "buffer", "cocov"
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.18
    x = np.arange(len(encs))

    for i, mkey in enumerate(methods_with_updates):
        updates = []
        for enc in encs:
            try:
                u = data[enc]["main"][mkey]["total_updates_mean"]
                updates.append(u)
            except (KeyError, TypeError):
                updates.append(0)

        offset = (i - len(methods_with_updates) / 2 + 0.5) * width
        ax.bar(x + offset, updates, width,
               color=METHODS[mkey]["color"],
               alpha=0.85,
               label=METHODS[mkey]["label"])

    ax.set_xticks(x)
    ax.set_xticklabels([ENCODERS[e]["label"] for e in encs])
    ax.set_ylabel("Mean Update Count per Run")
    ax.set_title("Memory Update Selectivity — COCOV vs Unsupervised Baselines",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"
    ))
    fig.tight_layout()
    savefig(fig, fig_dir, "fig8_updates_bar.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Radar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_radar(data, fig_dir):
    encs = [e for e in ENCODERS if e in data and "main" in data[e]]
    metrics_radar = [
        ("auc",     "AUC",       1,    0.95, 1.0),
        ("eer",     "1-EER",     -100, 0.90, 1.0),  # inverted
        ("tar_at_1_percent", "TAR@1%", 1, 0.70, 1.0),
    ]

    # Build cross-dataset AUC for CACD and FG-NET
    categories = ["VGG\nAUC", "1−EER", "TAR\n@1%", "CACD\nAUC", "FG-NET\nAUC"]
    n_cat = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw={"polar": True})

    for enc in encs:
        rec = data[enc]
        main = rec.get("main", {})
        xds  = rec.get("cross_dataset", {})
        color = ENC_COLORS.get(enc, "#555")

        def safe(d, *keys):
            try:
                v = d
                for k in keys:
                    v = v[k]
                return float(v)
            except (KeyError, TypeError):
                return 0.0

        auc     = safe(main, "cocov", "auc_mean")
        eer     = 1 - safe(main, "cocov", "eer_mean")
        tar1    = safe(main, "cocov", "tar_at_far1_mean")
        cacd_auc = safe(xds, "cacd", "cocov", "auc_mean")
        fgnet_auc = safe(xds, "fgnet", "cocov", "auc_mean")

        vals = [auc, eer, tar1, cacd_auc, fgnet_auc]
        vals += vals[:1]

        ax.plot(angles, vals, "o-", lw=2, color=color,
                label=ENCODERS[enc]["label"], ms=5)
        ax.fill(angles, vals, alpha=0.07, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9)
    ax.set_ylim(0.7, 1.0)
    ax.set_yticks([0.75, 0.85, 0.95, 1.0])
    ax.set_yticklabels(["0.75", "0.85", "0.95", "1.00"], fontsize=7)
    ax.set_title("COCOV Performance Profile — All Encoders",
                 fontsize=12, pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05),
              fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig9_radar.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — COCOV gain over Static
# ─────────────────────────────────────────────────────────────────────────────

def fig_cocov_gain(data, fig_dir):
    encs = [e for e in ENCODERS if e in data and "main" in data[e]]
    datasets = [("vggface2", "VGGFace2"), ("cacd", "CACD"), ("fgnet", "FG-NET")]
    width = 0.25
    x = np.arange(len(encs))

    fig, ax = plt.subplots(figsize=(11, 5))
    ds_colors = {"vggface2": "#6366F1", "cacd": "#F59E0B", "fgnet": "#10B981"}

    for i, (ds, ds_label) in enumerate(datasets):
        gains = []
        for enc in encs:
            if ds == "vggface2":
                static_m, _ = extract_metric(data, enc, "static", "auc", "vggface2")
                cocov_m, _  = extract_metric(data, enc, "cocov",  "auc", "vggface2")
            else:
                static_m, _ = extract_metric(data, enc, "static", "auc", ds)
                cocov_m, _  = extract_metric(data, enc, "cocov",  "auc", ds)

            if static_m is not None and cocov_m is not None:
                gains.append((cocov_m - static_m) * 100)
            else:
                gains.append(0)

        offset = (i - len(datasets) / 2 + 0.5) * width
        bars = ax.bar(x + offset, gains, width,
                      color=ds_colors[ds], alpha=0.85,
                      label=ds_label)
        for bar, val in zip(bars, gains):
            if abs(val) > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{val:+.2f}",
                        ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ENCODERS[e]["label"] for e in encs])
    ax.set_ylabel("COCOV − Static (AUC × 100 pp)")
    ax.set_title("COCOV Gain over Static Enrollment — All Encoders × Datasets",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig10_cocov_gain.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Tables
# ─────────────────────────────────────────────────────────────────────────────

def fmt(mean, std, pct=False, bold=False):
    if mean is None:
        return "--"
    scale = 100 if pct else 1
    s = f"{mean * scale:.4f}" if not pct else f"{mean * scale:.2f}"
    if std:
        s += f"$\\pm${std * scale:.4f}" if not pct else f"$\\pm${std * scale:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def tab_main_results(data, tab_dir):
    encs = [e for e in ENCODERS if e in data and "main" in data[e]]
    methods = list(METHODS.keys())

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Verification performance on VGGFace2 (30 runs, mean $\pm$ std). "
                 r"Bold: best per encoder. COCOV is our method.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{ll" + "ccc" * len(encs) + r"}")
    lines.append(r"\toprule")

    # Header row 1 — encoder names
    header1 = r"& "
    for enc in encs:
        header1 += r"& \multicolumn{3}{c}{" + ENCODERS[enc]["label"] + r"}"
    lines.append(header1 + r" \\")

    # Cmidrule
    cmi = ""
    for i, enc in enumerate(encs):
        col_start = 3 + i * 3
        cmi += f"\\cmidrule(lr){{{col_start}-{col_start+2}}}"
    lines.append(cmi)

    # Header row 2 — metric names
    header2 = r"Encoder & Method"
    for _ in encs:
        header2 += r" & AUC & EER(\%) & TAR@1\%"
    lines.append(header2 + r" \\")
    lines.append(r"\midrule")

    for mkey in methods:
        is_cocov = (mkey == "cocov")
        row_enc_label = ENCODERS[encs[0]]["label"] if mkey == methods[0] else ""

        # Find best AUC per encoder to bold
        best_aucs = {}
        for enc in encs:
            best = -np.inf
            for mk in methods:
                m, _ = extract_metric(data, enc, mk, "auc")
                if m and m > best:
                    best = m
            best_aucs[enc] = best

        row = f"& {METHODS[mkey]['label']}"
        if is_cocov:
            row = f"& \\textbf{{{METHODS[mkey]['label']}}}"

        for enc in encs:
            auc_m, auc_s = extract_metric(data, enc, mkey, "auc")
            eer_m, eer_s = extract_metric(data, enc, mkey, "eer")
            tar_m, tar_s = extract_metric(data, enc, mkey, "tar_at_1_percent")

            is_best = auc_m is not None and abs(auc_m - best_aucs[enc]) < 1e-6
            row += f" & {fmt(auc_m, auc_s, bold=is_best)}"
            row += f" & {fmt(eer_m, eer_s, pct=True)}"
            row += f" & {fmt(tar_m, tar_s, pct=True)}"

        if is_cocov:
            lines.append(r"\midrule")
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = tab_dir / "tab1_main_results.tex"
    out.write_text("\n".join(lines))
    logger.info(f"  Saved: {out}")


def tab_cross_dataset(data, tab_dir):
    encs = [e for e in ENCODERS if e in data]
    datasets = [("vggface2", "VGGFace2"), ("cacd", "CACD"), ("fgnet", "FG-NET")]
    methods_show = ["static", "cocov"]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-dataset evaluation: AUC (mean $\pm$ std). "
                 r"Static = Static Enrollment baseline.}")
    lines.append(r"\label{tab:cross_dataset}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")

    n_cols = 1 + len(datasets) * 2  # method + 2 per dataset
    lines.append(r"\begin{tabular}{l" + "cc" * len(datasets) + r"}")
    lines.append(r"\toprule")

    # Headers
    h1 = r"Encoder"
    for _, ds_label in datasets:
        h1 += f" & \\multicolumn{{2}}{{c}}{{{ds_label}}}"
    lines.append(h1 + r" \\")
    cmi = ""
    for i in range(len(datasets)):
        c0 = 2 + i * 2
        cmi += f"\\cmidrule(lr){{{c0}-{c0+1}}}"
    lines.append(cmi)

    h2 = ""
    for _ in datasets:
        h2 += r" & Static & COCOV"
    lines.append(h2 + r" \\")
    lines.append(r"\midrule")

    for enc in encs:
        row = ENCODERS[enc]["label"]
        for ds, _ in datasets:
            for mkey in methods_show:
                if ds == "vggface2":
                    m, s = extract_metric(data, enc, mkey, "auc", "vggface2")
                else:
                    m, s = extract_metric(data, enc, mkey, "auc", ds)
                is_bold = (mkey == "cocov")
                row += f" & {fmt(m, s, bold=is_bold)}"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = tab_dir / "tab2_cross_dataset.tex"
    out.write_text("\n".join(lines))
    logger.info(f"  Saved: {out}")


def tab_ablation(data, tab_dir):
    encs = [e for e in ENCODERS if e in data and "ablation" in data[e]]
    abl_keys = list(ABLATION_COMPONENTS.keys())

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study: AUC (mean $\pm$ std) across encoders. "
                 r"Full COCOV is the complete system; remaining rows remove one component.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{l" + "c" * len(encs) + r"}")
    lines.append(r"\toprule")

    header = r"Configuration"
    for enc in encs:
        header += f" & {ENCODERS[enc]['label']}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for k in abl_keys:
        row = ABLATION_COMPONENTS[k]
        if k == "cocov_full":
            row = r"\textbf{" + row + r"}"
        for enc in encs:
            try:
                m = data[enc]["ablation"][k]["auc_mean"]
                s = data[enc]["ablation"][k].get("auc_std", 0)
                bold = (k == "cocov_full")
                row += f" & {fmt(m, s, bold=bold)}"
            except (KeyError, TypeError):
                row += " & --"
        if k == "cocov_full":
            lines.append(r"\midrule")
        lines.append(row + r" \\")
        if k == "COCOV (Full)":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = tab_dir / "tab3_ablation.tex"
    out.write_text("\n".join(lines))
    logger.info(f"  Saved: {out}")


def tab_encoder_summary(tab_dir):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Encoder configuration summary. All encoders use "
                 r"$\ell_2$-normalised 512-dimensional embeddings and are fixed "
                 r"(no fine-tuning) throughout all experiments.}")
    lines.append(r"\label{tab:encoders}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llllll}")
    lines.append(r"\toprule")
    lines.append(r"Code & Encoder & Backbone & Loss & Training Data & Input \\")
    lines.append(r"\midrule")

    config = {
        "ENC01_FACENET":       ("FaceNet",       "InceptionResNetV1", "Triplet",   "VGGFace2",    "160×160"),
        "ENC02_ARCFACE_R50":   ("ArcFace-R50",   "IResNet-50",        "ArcFace",   "WebFace600K", "112×112"),
        "ENC03_ARCFACE_R100":  ("ArcFace-R100",  "IResNet-100",       "ArcFace",   "Glint360K",   "112×112"),
        "ENC04_MOBILEFACENET": ("MobileFaceNet", "MobileFaceNet",     "ArcFace",   "WebFace600K", "112×112"),
        "ENC04_ADAFACE":       ("AdaFace",       "IResNet-101",       "AdaFace",   "MS1MV3",      "112×112"),
    }

    for code, (name, backbone, loss, data_src, inp) in config.items():
        short_code = code.split("_", 1)[1]
        lines.append(f"{short_code} & {name} & {backbone} & {loss} & {data_src} & {inp} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = tab_dir / "tab4_encoder_summary.tex"
    out.write_text("\n".join(lines))
    logger.info(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate COCOV thesis tables and figures"
    )
    parser.add_argument(
        "--results_dir",
        default="/opt/data/cocov/results",
        help="Root directory containing per-encoder result folders"
    )
    parser.add_argument(
        "--output_dir",
        default="/opt/data/cocov/analysis",
        help="Output directory for figures and tables"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    fig_dir     = output_dir / "figures"
    tab_dir     = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results dir : {results_dir}")
    logger.info(f"Output dir  : {output_dir}")

    # Load
    data = load_results(results_dir)
    if not data:
        logger.error("No result data found. Check --results_dir.")
        return

    logger.info(f"\nLoaded {len(data)} encoders: {list(data.keys())}")

    # ── Figures ──────────────────────────────────────────────────────────────
    logger.info("\n── Generating figures ──")

    logger.info("  Figure 1: Main AUC grouped bar")
    fig_main_auc(data, fig_dir)

    logger.info("  Figure 2: Main EER grouped bar")
    fig_main_eer(data, fig_dir)

    logger.info("  Figure 3: COCOV vs Static scatter")
    fig_cocov_vs_static_scatter(data, fig_dir)

    logger.info("  Figure 4: Ablation heatmap")
    fig_ablation_heatmap(data, fig_dir)

    logger.info("  Figure 5: Ablation bars")
    fig_ablation_bars(data, fig_dir)

    logger.info("  Figure 6: Cross-dataset AUC")
    fig_cross_dataset(data, fig_dir, metric="auc")

    logger.info("  Figure 7: Cross-dataset EER")
    fig_cross_dataset(data, fig_dir, metric="eer")

    logger.info("  Figure 8: Update counts")
    fig_updates(data, fig_dir)

    logger.info("  Figure 9: Radar chart")
    fig_radar(data, fig_dir)

    logger.info("  Figure 10: COCOV gain")
    fig_cocov_gain(data, fig_dir)

    # ── Tables ───────────────────────────────────────────────────────────────
    logger.info("\n── Generating LaTeX tables ──")

    logger.info("  Table 1: Main results")
    tab_main_results(data, tab_dir)

    logger.info("  Table 2: Cross-dataset")
    tab_cross_dataset(data, tab_dir)

    logger.info("  Table 3: Ablation")
    tab_ablation(data, tab_dir)

    logger.info("  Table 4: Encoder summary")
    tab_encoder_summary(tab_dir)

    logger.info(f"\nDone. Outputs written to {output_dir}")
    logger.info(f"  Figures : {fig_dir}")
    logger.info(f"  Tables  : {tab_dir}")


if __name__ == "__main__":
    main()
