"""
analysis/plots.py
-----------------
Figure generation for Chapter 6 experimental results.

Generates all plots reported in the thesis including:
    - ROC curves per method and dataset
    - Drift trajectory plots per identity
    - Drift distribution comparisons
    - Prototype count evolution
    - Update count comparisons
    - Sensitivity analysis curves
    - Ablation study bar charts
    - Performance across sequence position

All figures are saved as high-resolution PDFs and PNGs
suitable for thesis inclusion. Figure style follows a
consistent palette and typography throughout.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import logging
from pathlib import Path
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Style Configuration
# ----------------------------------------------------------

# Consistent colour palette across all figures
METHOD_COLORS = {
    'Static Enrollment':      '#2196F3',  # Blue
    'Naive OLS Expansion':    '#F44336',  # Red
    'Replay Dual Memory':     '#FF9800',  # Orange
    'Fixed Buffer Averaging': '#4CAF50',  # Green
    'COCOV':                  '#9C27B0',  # Purple
}

METHOD_LINESTYLES = {
    'Static Enrollment':      '-',
    'Naive OLS Expansion':    '--',
    'Replay Dual Memory':     '-.',
    'Fixed Buffer Averaging': ':',
    'COCOV':                  '-',
}

METHOD_MARKERS = {
    'Static Enrollment':      'o',
    'Naive OLS Expansion':    's',
    'Replay Dual Memory':     '^',
    'Fixed Buffer Averaging': 'D',
    'COCOV':                  '*',
}

ABLATION_COLORS = {
    'COCOV (Full)':           '#9C27B0',
    'COCOV-NoDrift':          '#E91E63',
    'COCOV-NoMerge':          '#FF5722',
    'COCOV-NoReviewer':       '#795548',
    'COCOV-Unbounded':        '#607D8B',
    'COCOV-SinglePrototype':  '#009688',
}

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'pdf'
FONT_SIZE = 11
TITLE_SIZE = 12
LEGEND_SIZE = 9

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': LEGEND_SIZE,
    'figure.dpi': FIGURE_DPI,
    'savefig.dpi': FIGURE_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


class PlotGenerator:
    """
    Generates all thesis figures from saved experiment results.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing JSON result files.
    figures_dir : str or Path
        Directory where figures are saved.
    """

    def __init__(
        self,
        results_dir: str | Path,
        figures_dir: str | Path
    ):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"PlotGenerator: saving figures to "
            f"{self.figures_dir}"
        )

    # --------------------------------------------------
    # ROC Curves
    # --------------------------------------------------

    def plot_roc_curves(
        self,
        run_metrics: dict[str, list],
        title: str = "ROC Curves",
        filename: str = "roc_curves"
    ) -> None:
        """
        Plot ROC curves for all methods.

        One curve per method, computed from the first run.
        AUC values shown in legend.

        Parameters
        ----------
        run_metrics : dict
            Maps method display name to list of RunMetrics.
        title : str
            Figure title.
        filename : str
            Output filename without extension.
        """
        fig, ax = plt.subplots(figsize=(6, 5))

        for method_name, metrics_list in \
                run_metrics.items():
            if not metrics_list:
                continue

            # Use first run for ROC curve shape
            m = metrics_list[0]
            mean_auc = np.mean([
                mi.auc for mi in metrics_list
            ])

            color = METHOD_COLORS.get(
                method_name, '#666666'
            )
            ls = METHOD_LINESTYLES.get(method_name, '-')

            ax.plot(
                m.fpr, m.tpr,
                color=color,
                linestyle=ls,
                linewidth=1.8,
                label=f"{method_name} "
                      f"(AUC={mean_auc:.4f})"
            )

        # Diagonal reference
        ax.plot(
            [0, 1], [0, 1],
            'k--', linewidth=0.8, alpha=0.5,
            label='Random'
        )

        ax.set_xlabel('False Accept Rate')
        ax.set_ylabel('True Accept Rate')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc='lower right')
        ax.set_title(title)

        self._save(fig, filename)

    def plot_roc_curves_lowfar(
        self,
        run_metrics: dict[str, list],
        title: str = "ROC Curves (Low FAR)",
        filename: str = "roc_curves_lowfar"
    ) -> None:
        """
        Plot ROC curves zoomed to FAR <= 10%.

        Highlights differences in the security-sensitive
        low false accept rate operating region.

        Parameters
        ----------
        run_metrics : dict
            Maps method display name to list of RunMetrics.
        title : str
            Figure title.
        filename : str
            Output filename without extension.
        """
        fig, ax = plt.subplots(figsize=(6, 5))

        for method_name, metrics_list in \
                run_metrics.items():
            if not metrics_list:
                continue

            m = metrics_list[0]
            mean_auc = np.mean([
                mi.auc for mi in metrics_list
            ])
            color = METHOD_COLORS.get(
                method_name, '#666666'
            )
            ls = METHOD_LINESTYLES.get(method_name, '-')

            # Filter to FAR <= 0.10
            mask = m.fpr <= 0.10
            ax.plot(
                m.fpr[mask], m.tpr[mask],
                color=color,
                linestyle=ls,
                linewidth=1.8,
                label=f"{method_name} "
                      f"(AUC={mean_auc:.4f})"
            )

        ax.set_xlabel('False Accept Rate')
        ax.set_ylabel('True Accept Rate')
        ax.set_xlim([0.0, 0.10])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc='lower right')
        ax.set_title(title)

        self._save(fig, filename)

    # --------------------------------------------------
    # Drift Analysis
    # --------------------------------------------------

    def plot_drift_trajectories(
        self,
        drift_data: dict[str, list[list[float]]],
        identity_ids: list[str],
        filename: str = "drift_trajectories"
    ) -> None:
        """
        Plot drift trajectories for selected identities.

        One subplot per method showing drift over the
        probe sequence for a sample of identities from
        the diagnostic subset.

        Parameters
        ----------
        drift_data : dict
            Maps method name to list of per-identity
            drift sequences.
        identity_ids : list of str
            Identity IDs corresponding to drift sequences.
        filename : str
            Output filename without extension.
        """
        methods = list(drift_data.keys())
        n_methods = len(methods)
        n_identities = min(
            5, len(identity_ids)
        )

        fig, axes = plt.subplots(
            1, n_methods,
            figsize=(4 * n_methods, 4),
            sharey=True
        )

        if n_methods == 1:
            axes = [axes]

        for ax, method_name in zip(axes, methods):
            color = METHOD_COLORS.get(
                method_name, '#666666'
            )
            sequences = drift_data[method_name]

            for i, seq in enumerate(
                sequences[:n_identities]
            ):
                alpha = 0.6 if i > 0 else 1.0
                lw = 1.0 if i > 0 else 1.8
                ax.plot(
                    seq,
                    color=color,
                    alpha=alpha,
                    linewidth=lw
                )

            ax.set_title(method_name, fontsize=9)
            ax.set_xlabel('Probe Index')
            if ax == axes[0]:
                ax.set_ylabel('Cosine Distance (Drift)')
            ax.set_ylim([0, 2])

        fig.suptitle(
            'Drift Trajectories Across Methods',
            fontsize=TITLE_SIZE
        )
        plt.tight_layout()
        self._save(fig, filename)

    def plot_drift_distributions(
        self,
        drift_data: dict[str, np.ndarray],
        filename: str = "drift_distributions"
    ) -> None:
        """
        Plot drift magnitude distributions per method.

        Kernel density estimates of drift values aggregated
        across all identities and runs.

        Parameters
        ----------
        drift_data : dict
            Maps method name to array of all drift values.
        filename : str
            Output filename without extension.
        """
        fig, ax = plt.subplots(figsize=(7, 4))

        for method_name, drifts in drift_data.items():
            if len(drifts) == 0:
                continue
            color = METHOD_COLORS.get(
                method_name, '#666666'
            )
            sns.kdeplot(
                drifts,
                ax=ax,
                color=color,
                label=method_name,
                linewidth=1.8,
                fill=True,
                alpha=0.15
            )

        ax.set_xlabel('Cosine Distance (Drift)')
        ax.set_ylabel('Density')
        ax.set_xlim([0, 2])
        ax.legend()
        ax.set_title('Drift Magnitude Distributions')

        self._save(fig, filename)

    # --------------------------------------------------
    # Update Counts
    # --------------------------------------------------

    def plot_update_counts(
        self,
        update_counts: dict[str, dict],
        filename: str = "update_counts"
    ) -> None:
        """
        Bar chart of total update counts per method.

        Parameters
        ----------
        update_counts : dict
            Maps method name to update count statistics.
            Each value has 'mean' and 'std' keys.
        filename : str
            Output filename without extension.
        """
        fig, ax = plt.subplots(figsize=(8, 4))

        methods = list(update_counts.keys())
        means = [
            update_counts[m]['mean'] for m in methods
        ]
        stds = [
            update_counts[m]['std'] for m in methods
        ]
        colors = [
            METHOD_COLORS.get(m, '#666666')
            for m in methods
        ]

        x = np.arange(len(methods))
        bars = ax.bar(
            x, means,
            yerr=stds,
            color=colors,
            alpha=0.85,
            capsize=4,
            edgecolor='white',
            linewidth=0.5
        )

        ax.set_ylabel('Total Update Operations')
        ax.set_title('Update Counts per Method')
        ax.set_xticks(x)
        ax.set_xticklabels(
            methods,
            rotation=20,
            ha='right'
        )

        # Annotate bars with values
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(means) * 0.01,
                f'{mean:.0f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

        plt.tight_layout()
        self._save(fig, filename)

    # --------------------------------------------------
    # Prototype Evolution
    # --------------------------------------------------

    def plot_prototype_counts(
        self,
        prototype_data: dict[str, list[int]],
        filename: str = "prototype_counts"
    ) -> None:
        """
        Plot prototype count evolution over probe sequence.

        Shows how prototype counts change over time for
        the COCOV configurations in the ablation study.

        Parameters
        ----------
        prototype_data : dict
            Maps configuration name to list of prototype
            counts recorded at each update event.
        filename : str
            Output filename without extension.
        """
        fig, ax = plt.subplots(figsize=(7, 4))

        for config_name, counts in \
                prototype_data.items():
            color = ABLATION_COLORS.get(
                config_name, '#666666'
            )
            ax.plot(
                counts,
                color=color,
                label=config_name,
                linewidth=1.5
            )

        ax.set_xlabel('Update Event Index')
        ax.set_ylabel('Prototype Count')
        ax.set_title('Prototype Count Evolution')
        ax.legend(fontsize=8)

        self._save(fig, filename)

    # --------------------------------------------------
    # Sensitivity Analysis
    # --------------------------------------------------

    def plot_threshold_sensitivity(
        self,
        sensitivity_data: dict,
        filename: str = "threshold_sensitivity"
    ) -> None:
        """
        Plot AUC and EER as functions of tau_ver.

        Parameters
        ----------
        sensitivity_data : list of dict
            tau_ver sweep results from calibration.
        filename : str
            Output filename without extension.
        """
        tau_ver_data = sensitivity_data.get('tau_ver', [])
        if not tau_ver_data:
            logger.warning(
                "No tau_ver sensitivity data available."
            )
            return

        thresholds = [d['threshold'] for d in tau_ver_data]
        tars = [d['tar'] for d in tau_ver_data]
        fars = [d['far'] for d in tau_ver_data]

        fig, axes = plt.subplots(
            1, 2, figsize=(10, 4)
        )

        # TAR vs threshold
        axes[0].plot(
            thresholds, tars,
            color=METHOD_COLORS['COCOV'],
            linewidth=1.8
        )
        axes[0].set_xlabel(r'$\tau_{\mathrm{ver}}$')
        axes[0].set_ylabel('True Accept Rate')
        axes[0].set_title(
            r'TAR vs $\tau_{\mathrm{ver}}$'
        )

        # FAR vs threshold
        axes[1].plot(
            thresholds, fars,
            color=METHOD_COLORS['COCOV'],
            linewidth=1.8
        )
        axes[1].set_xlabel(r'$\tau_{\mathrm{ver}}$')
        axes[1].set_ylabel('False Accept Rate')
        axes[1].set_title(
            r'FAR vs $\tau_{\mathrm{ver}}$'
        )

        plt.tight_layout()
        self._save(fig, filename)

    def plot_drift_threshold_sensitivity(
        self,
        sensitivity_data: dict,
        filename: str = "drift_threshold_sensitivity"
    ) -> None:
        """
        Plot escalation rate as function of tau_delta.

        Parameters
        ----------
        sensitivity_data : dict
            Calibration sensitivity results.
        filename : str
            Output filename without extension.
        """
        tau_delta_data = sensitivity_data.get(
            'tau_delta', []
        )
        if not tau_delta_data:
            logger.warning(
                "No tau_delta sensitivity data available."
            )
            return

        thresholds = [
            d['threshold'] for d in tau_delta_data
        ]
        escalation = [
            d['escalation_rate'] for d in tau_delta_data
        ]
        genuine_escalation = [
            d['genuine_escalation_rate']
            for d in tau_delta_data
        ]

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            thresholds, escalation,
            color='#F44336',
            linewidth=1.8,
            label='All trials'
        )
        ax.plot(
            thresholds, genuine_escalation,
            color='#9C27B0',
            linewidth=1.8,
            linestyle='--',
            label='Genuine trials'
        )

        ax.set_xlabel(r'$\tau_{\Delta}$')
        ax.set_ylabel('Escalation Rate')
        ax.set_title(
            r'Escalation Rate vs $\tau_{\Delta}$'
        )
        ax.legend()

        self._save(fig, filename)

    def plot_kmax_sensitivity(
        self,
        sensitivity_data: dict,
        filename: str = "kmax_sensitivity"
    ) -> None:
        """
        Plot AUC and mean prototype count vs K_max.

        Parameters
        ----------
        sensitivity_data : dict
            Calibration sensitivity results.
        filename : str
            Output filename without extension.
        """
        kmax_data = sensitivity_data.get(
            'max_prototypes', []
        )
        if not kmax_data:
            logger.warning(
                "No K_max sensitivity data available."
            )
            return

        k_values = [d['k_max'] for d in kmax_data]
        aucs = [d['auc'] for d in kmax_data]
        mean_protos = [
            d['mean_prototype_count'] for d in kmax_data
        ]

        fig, axes = plt.subplots(
            1, 2, figsize=(10, 4)
        )

        axes[0].plot(
            k_values, aucs,
            color=METHOD_COLORS['COCOV'],
            linewidth=1.8,
            marker='o',
            markersize=5
        )
        axes[0].set_xlabel(r'$K_{\max}$')
        axes[0].set_ylabel('AUC')
        axes[0].set_title(
            r'AUC vs $K_{\max}$'
        )

        axes[1].plot(
            k_values, mean_protos,
            color='#FF9800',
            linewidth=1.8,
            marker='s',
            markersize=5
        )
        axes[1].set_xlabel(r'$K_{\max}$')
        axes[1].set_ylabel('Mean Prototype Count')
        axes[1].set_title(
            r'Mean Prototype Count vs $K_{\max}$'
        )

        plt.tight_layout()
        self._save(fig, filename)

    # --------------------------------------------------
    # Ablation Study
    # --------------------------------------------------

    def plot_ablation_results(
        self,
        ablation_results: dict,
        filename: str = "ablation_results"
    ) -> None:
        """
        Bar chart comparing ablation configurations.

        Shows AUC, EER, and TAR@FAR=1% for each
        COCOV ablation variant side by side.

        Parameters
        ----------
        ablation_results : dict
            Aggregated ablation metrics per configuration.
        filename : str
            Output filename without extension.
        """
        display_names = {
            'cocov_full': 'COCOV\n(Full)',
            'cocov_no_drift': 'No\nDrift',
            'cocov_no_merge': 'No\nMerge',
            'cocov_no_reviewer': 'No\nReviewer',
            'cocov_unbounded': 'Unbounded\nMemory',
            'cocov_single_proto': 'Single\nPrototype'
        }

        configs = list(ablation_results.keys())
        labels = [
            display_names.get(c, c) for c in configs
        ]

        aucs = [
            ablation_results[c]['auc_mean']
            for c in configs
        ]
        auc_stds = [
            ablation_results[c]['auc_std']
            for c in configs
        ]
        eers = [
            ablation_results[c]['eer_mean']
            for c in configs
        ]
        tars = [
            ablation_results[c]['tar_at_far1_mean']
            for c in configs
        ]

        fig, axes = plt.subplots(
            1, 3, figsize=(14, 4)
        )

        x = np.arange(len(configs))
        colors = [
            ABLATION_COLORS.get(
                display_names.get(c, c).replace('\n', ' '),
                '#9C27B0'
            )
            for c in configs
        ]

        # Use consistent purple shades for ablation
        bar_colors = [
            '#9C27B0', '#CE93D8', '#AB47BC',
            '#7B1FA2', '#6A1B9A', '#4A148C'
        ]

        for ax, values, stds, ylabel, title in zip(
            axes,
            [aucs, eers, tars],
            [auc_stds, None, None],
            ['AUC', 'EER', 'TAR@FAR=1%'],
            ['AUC', 'EER', 'TAR@FAR=1%']
        ):
            yerr = stds if stds else None
            ax.bar(
                x, values,
                yerr=yerr,
                color=bar_colors[:len(configs)],
                alpha=0.85,
                capsize=3,
                edgecolor='white'
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                labels, fontsize=8
            )
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        fig.suptitle(
            'Ablation Study: COCOV Component Contributions',
            fontsize=TITLE_SIZE
        )
        plt.tight_layout()
        self._save(fig, filename)

    # --------------------------------------------------
    # Performance vs Sequence Position
    # --------------------------------------------------

    def plot_performance_over_sequence(
        self,
        sequence_results: dict[str, list[float]],
        window: int = 50,
        filename: str = "performance_over_sequence"
    ) -> None:
        """
        Plot rolling TAR across probe sequence positions.

        Shows how verification accuracy evolves over the
        course of the evaluation stream, revealing whether
        methods improve, degrade, or remain stable over time.

        Parameters
        ----------
        sequence_results : dict
            Maps method name to list of binary outcomes
            (1=correct, 0=incorrect) in stream order.
        window : int
            Rolling window size for smoothing. Default 50.
        filename : str
            Output filename without extension.
        """
        fig, ax = plt.subplots(figsize=(8, 4))

        for method_name, outcomes in \
                sequence_results.items():
            if not outcomes:
                continue

            outcomes_arr = np.array(outcomes,
                                    dtype=float)

            # Rolling mean
            if len(outcomes_arr) >= window:
                rolling = np.convolve(
                    outcomes_arr,
                    np.ones(window) / window,
                    mode='valid'
                )
                x = np.arange(
                    window - 1,
                    window - 1 + len(rolling)
                )
            else:
                rolling = outcomes_arr
                x = np.arange(len(rolling))

            color = METHOD_COLORS.get(
                method_name, '#666666'
            )
            ls = METHOD_LINESTYLES.get(method_name, '-')

            ax.plot(
                x, rolling,
                color=color,
                linestyle=ls,
                linewidth=1.5,
                label=method_name,
                alpha=0.9
            )

        ax.set_xlabel('Stream Position')
        ax.set_ylabel(f'Rolling TAR (window={window})')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=8)
        ax.set_title(
            'Verification Accuracy Over Sequence'
        )

        self._save(fig, filename)

    # --------------------------------------------------
    # Summary Comparison
    # --------------------------------------------------

    def plot_summary_comparison(
        self,
        aggregated_results: dict,
        filename: str = "summary_comparison"
    ) -> None:
        """
        Summary figure comparing all methods across
        AUC, EER, TAR@FAR=1%, and update counts.

        Parameters
        ----------
        aggregated_results : dict
            Aggregated metrics per method from
            aggregated_results.json.
        filename : str
            Output filename without extension.
        """
        method_display = {
            'static': 'Static\nEnrollment',
            'ols': 'Naive\nOLS',
            'replay': 'Replay\nDual Memory',
            'buffer': 'Buffer\nAveraging',
            'cocov': 'COCOV'
        }

        methods = list(aggregated_results.keys())
        labels = [
            method_display.get(m, m) for m in methods
        ]
        colors = [
            METHOD_COLORS.get(
                method_display.get(
                    m, m
                ).replace('\n', ' '),
                '#666666'
            )
            for m in methods
        ]

        # Use consistent method colors
        plot_colors = [
            '#2196F3', '#F44336', '#FF9800',
            '#4CAF50', '#9C27B0'
        ]

        fig, axes = plt.subplots(
            2, 2, figsize=(12, 8)
        )
        axes = axes.flatten()

        metrics_config = [
            ('auc_mean', 'auc_std', 'AUC',
             'higher is better'),
            ('eer_mean', 'eer_std', 'EER',
             'lower is better'),
            ('tar_at_far1_mean', 'tar_at_far1_std',
             'TAR@FAR=1%', 'higher is better'),
            ('total_updates_mean', 'total_updates_std',
             'Total Updates', 'lower is better')
        ]

        x = np.arange(len(methods))

        for ax, (mean_key, std_key, ylabel, note) in \
                zip(axes, metrics_config):
            means = [
                aggregated_results[m][mean_key]
                for m in methods
            ]
            stds = [
                aggregated_results[m][std_key]
                for m in methods
            ]

            ax.bar(
                x, means,
                yerr=stds,
                color=plot_colors[:len(methods)],
                alpha=0.85,
                capsize=3,
                edgecolor='white'
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} ({note})')

        fig.suptitle(
            'Summary Comparison Across Methods',
            fontsize=TITLE_SIZE + 1
        )
        plt.tight_layout()
        self._save(fig, filename)

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def generate_all(
        self,
        run_metrics: dict = None,
        aggregated_results: dict = None,
        ablation_results: dict = None,
        calibration_results: dict = None
    ) -> None:
        """
        Generate all available figures from saved results.

        Parameters
        ----------
        run_metrics : dict, optional
            Per-run metrics for ROC curves.
        aggregated_results : dict, optional
            Aggregated results for summary plots.
        ablation_results : dict, optional
            Ablation study results.
        calibration_results : dict, optional
            Calibration and sensitivity results.
        """
        logger.info("Generating all figures...")

        if run_metrics:
            self.plot_roc_curves(
                run_metrics,
                filename='roc_curves_all_methods'
            )
            self.plot_roc_curves_lowfar(
                run_metrics,
                filename='roc_curves_lowfar'
            )
            logger.info("ROC curves generated.")

        if aggregated_results:
            self.plot_summary_comparison(
                aggregated_results,
                filename='summary_comparison'
            )
            logger.info("Summary comparison generated.")

            # Update counts
            update_data = {
                m: {
                    'mean': aggregated_results[m][
                        'total_updates_mean'
                    ],
                    'std': aggregated_results[m][
                        'total_updates_std'
                    ]
                }
                for m in aggregated_results
            }
            self.plot_update_counts(
                update_data,
                filename='update_counts'
            )
            logger.info("Update counts plot generated.")

        if ablation_results:
            self.plot_ablation_results(
                ablation_results,
                filename='ablation_results'
            )
            logger.info("Ablation plot generated.")

        if calibration_results:
            sensitivity = calibration_results.get(
                'sensitivity', {}
            )
            self.plot_threshold_sensitivity(
                sensitivity,
                filename='tau_ver_sensitivity'
            )
            self.plot_drift_threshold_sensitivity(
                sensitivity,
                filename='tau_delta_sensitivity'
            )
            self.plot_kmax_sensitivity(
                sensitivity,
                filename='kmax_sensitivity'
            )
            logger.info("Sensitivity plots generated.")

        logger.info(
            f"All figures saved to {self.figures_dir}"
        )

    def _save(
        self,
        fig: plt.Figure,
        filename: str
    ) -> None:
        """
        Save figure as both PDF and PNG.

        Parameters
        ----------
        fig : plt.Figure
            Figure to save.
        filename : str
            Base filename without extension.
        """
        for ext in ['pdf', 'png']:
            path = self.figures_dir / f"{filename}.{ext}"
            fig.savefig(str(path), format=ext)

        plt.close(fig)
        logger.debug(f"Saved: {filename}")
