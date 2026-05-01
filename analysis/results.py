"""
analysis/results.py
-------------------
Results aggregation, statistical testing, and reporting
for the COCOV experimental evaluation.

Loads saved JSON results from experiment runs, performs
statistical tests across methods, generates LaTeX tables,
and produces summary reports for Chapter 6.

Statistical testing follows the protocol defined in
Chapter 6 Section 6.7:
    - Paired t-tests at alpha=0.05 across runs
    - Wilcoxon signed-rank tests where normality fails
    - Effect sizes reported alongside p-values

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import json
import numpy as np
import logging
from pathlib import Path
from scipy import stats
from itertools import combinations

logger = logging.getLogger(__name__)


class ResultsAnalyser:
    """
    Loads and analyses experimental results.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing JSON result files from
        run_experiment.py, ablation.py, and
        cross_dataset.py.
    """

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)

        logger.info(
            f"ResultsAnalyser: {self.results_dir}"
        )

    # --------------------------------------------------
    # Loading
    # --------------------------------------------------

    def load_aggregated_results(self) -> dict:
        """
        Load aggregated results from main experiment.

        Returns
        -------
        dict
            Aggregated metrics per method.

        Raises
        ------
        FileNotFoundError
            If aggregated_results.json does not exist.
        """
        path = self.results_dir / 'aggregated_results.json'
        if not path.exists():
            raise FileNotFoundError(
                f"Aggregated results not found at {path}. "
                f"Run run_experiment.py first."
            )
        with open(path, 'r') as f:
            return json.load(f)

    def load_run_results(self) -> dict[str, list[dict]]:
        """
        Load per-run results for statistical testing.

        Returns
        -------
        dict
            Maps method name to list of per-run result dicts.
        """
        run_dirs = sorted(
            self.results_dir.glob('run_*')
        )
        if not run_dirs:
            raise FileNotFoundError(
                "No run directories found. "
                "Run run_experiment.py first."
            )

        run_results = {}
        for run_dir in run_dirs:
            for result_file in run_dir.glob('*.json'):
                method_name = result_file.stem
                if method_name not in run_results:
                    run_results[method_name] = []
                with open(result_file, 'r') as f:
                    run_results[method_name].append(
                        json.load(f)
                    )

        return run_results

    def load_ablation_results(self) -> dict:
        """
        Load ablation study results.

        Returns
        -------
        dict
            Ablation metrics per configuration.
        """
        path = self.results_dir / 'ablation_results.json'
        if not path.exists():
            raise FileNotFoundError(
                f"Ablation results not found at {path}."
            )
        with open(path, 'r') as f:
            return json.load(f)

    def load_cross_dataset_results(self) -> dict:
        """
        Load cross-dataset evaluation results.

        Returns
        -------
        dict
            Results per dataset and method.
        """
        path = (
            self.results_dir / 'cross_dataset_results.json'
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Cross-dataset results not found at {path}."
            )
        with open(path, 'r') as f:
            return json.load(f)

    def load_calibration_results(self) -> dict:
        """
        Load calibration and sensitivity results.

        Returns
        -------
        dict
            Calibration results including optimal thresholds
            and sensitivity curves.
        """
        path = (
            self.results_dir / 'calibration_results.json'
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Calibration results not found at {path}."
            )
        with open(path, 'r') as f:
            return json.load(f)

    # --------------------------------------------------
    # Statistical Testing
    # --------------------------------------------------

    def pairwise_significance_tests(
        self,
        per_run_metrics: dict[str, list[float]],
        alpha: float = 0.05
    ) -> dict:
        """
        Perform pairwise statistical tests between methods.

        Uses paired t-tests where normality is satisfied
        (Shapiro-Wilk p > 0.05), otherwise Wilcoxon
        signed-rank tests.

        Parameters
        ----------
        per_run_metrics : dict
            Maps method name to list of metric values
            across runs (one value per run).
        alpha : float
            Significance level. Default 0.05.

        Returns
        -------
        dict
            Pairwise test results with p-values,
            test type, and significance flags.
        """
        methods = list(per_run_metrics.keys())
        results = {}

        for m1, m2 in combinations(methods, 2):
            values1 = np.array(per_run_metrics[m1])
            values2 = np.array(per_run_metrics[m2])

            if len(values1) < 3 or len(values2) < 3:
                logger.warning(
                    f"Insufficient runs for statistical "
                    f"testing: {m1} vs {m2}. "
                    f"Need at least 3 runs."
                )
                continue

            # Test normality of differences
            differences = values1 - values2
            if len(differences) >= 8:
                _, normality_p = stats.shapiro(differences)
                use_parametric = normality_p > 0.05
            else:
                # Too few samples for reliable normality test
                use_parametric = False

            if use_parametric:
                stat, p_value = stats.ttest_rel(
                    values1, values2
                )
                test_type = 'paired_t'
            else:
                stat, p_value = stats.wilcoxon(
                    differences,
                    zero_method='zsplit'
                )
                test_type = 'wilcoxon'

            # Effect size: Cohen's d for paired samples
            d = (
                np.mean(differences) /
                np.std(differences, ddof=1)
            ) if np.std(differences, ddof=1) > 0 else 0.0

            pair_key = f"{m1}_vs_{m2}"
            results[pair_key] = {
                'method_1': m1,
                'method_2': m2,
                'test_type': test_type,
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': bool(p_value < alpha),
                'cohens_d': float(d),
                'mean_difference': float(
                    np.mean(differences)
                ),
                'alpha': alpha
            }

        return results

    def run_all_significance_tests(
        self,
        run_dirs: list[Path],
        metric: str = 'auc'
    ) -> dict:
        """
        Run pairwise significance tests for a given metric
        across all experimental runs.

        Parameters
        ----------
        run_dirs : list of Path
            Directories containing per-run result files.
        metric : str
            Metric to test. Default 'auc'.

        Returns
        -------
        dict
            Pairwise test results.
        """
        per_run_metrics = {}

        for run_dir in run_dirs:
            for result_file in run_dir.glob('*.json'):
                method_name = result_file.stem
                with open(result_file, 'r') as f:
                    data = json.load(f)

                if method_name not in per_run_metrics:
                    per_run_metrics[method_name] = []

                # Per-run metric value must be stored
                # in run result files
                if metric in data:
                    per_run_metrics[method_name].append(
                        data[metric]
                    )

        if not per_run_metrics:
            logger.warning(
                f"No per-run {metric} values found. "
                f"Statistical tests skipped."
            )
            return {}

        return self.pairwise_significance_tests(
            per_run_metrics
        )

    # --------------------------------------------------
    # LaTeX Table Generation
    # --------------------------------------------------

    def generate_main_results_table(
        self,
        aggregated: dict,
        caption: str = None,
        label: str = "tab:comparative-results"
    ) -> str:
        """
        Generate LaTeX table for main comparative results.

        Parameters
        ----------
        aggregated : dict
            Aggregated metrics per method.
        caption : str, optional
            Table caption.
        label : str
            LaTeX label for cross-referencing.

        Returns
        -------
        str
            Complete LaTeX table string.
        """
        if caption is None:
            caption = (
                "Verification performance across baselines "
                "under continuous verification on VGGFace2. "
                "Results are means $\\pm$ standard deviations "
                "across five independent runs."
            )

        method_display = {
            'static': 'Static Enrollment',
            'ols': 'Naive OLS Expansion',
            'replay': 'Replay Dual Memory',
            'buffer': 'Fixed Buffer Averaging',
            'cocov': 'COCOV'
        }

        lines = [
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "\\textbf{Method} & "
            "\\textbf{AUC} & "
            "\\textbf{EER} & "
            "\\textbf{TAR@FAR=1\\%} & "
            "\\textbf{Updates} \\\\",
            "\\hline"
        ]

        # Find best values for bold formatting
        aucs = {
            m: v['auc_mean']
            for m, v in aggregated.items()
        }
        eers = {
            m: v['eer_mean']
            for m, v in aggregated.items()
        }
        tars = {
            m: v['tar_at_far1_mean']
            for m, v in aggregated.items()
        }
        updates = {
            m: v['total_updates_mean']
            for m, v in aggregated.items()
        }

        best_auc = max(aucs.values())
        best_eer = min(eers.values())
        best_tar = max(tars.values())
        best_updates = min(
            v for m, v in updates.items()
            if m != 'static'
        )

        for method_key, metrics in aggregated.items():
            display = method_display.get(
                method_key, method_key
            )

            auc_str = self._fmt(
                metrics['auc_mean'],
                metrics['auc_std'],
                bold=(metrics['auc_mean'] == best_auc)
            )
            eer_str = self._fmt(
                metrics['eer_mean'],
                metrics['eer_std'],
                bold=(metrics['eer_mean'] == best_eer)
            )
            tar_str = self._fmt(
                metrics['tar_at_far1_mean'],
                metrics['tar_at_far1_std'],
                bold=(
                    metrics['tar_at_far1_mean'] == best_tar
                )
            )
            upd_str = self._fmt_int(
                metrics['total_updates_mean'],
                metrics['total_updates_std'],
                bold=(
                    method_key != 'static' and
                    metrics['total_updates_mean'] ==
                    best_updates
                )
            )

            lines.append(
                f"{display} & {auc_str} & "
                f"{eer_str} & {tar_str} & "
                f"{upd_str} \\\\"
            )

        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return '\n'.join(lines)

    def generate_ablation_table(
        self,
        ablation: dict,
        caption: str = None,
        label: str = "tab:ablation"
    ) -> str:
        """
        Generate LaTeX table for ablation study results.

        Parameters
        ----------
        ablation : dict
            Ablation metrics per configuration.
        caption : str, optional
            Table caption.
        label : str
            LaTeX label.

        Returns
        -------
        str
            Complete LaTeX table string.
        """
        if caption is None:
            caption = (
                "Ablation study: contribution of individual "
                "COCOV components. Results are means "
                "$\\pm$ standard deviations across five runs."
            )

        display_names = {
            'cocov_full': 'COCOV (Full)',
            'cocov_no_drift': 'COCOV-NoDrift',
            'cocov_no_merge': 'COCOV-NoMerge',
            'cocov_no_reviewer': 'COCOV-NoReviewer',
            'cocov_unbounded': 'COCOV-Unbounded',
            'cocov_single_proto': 'COCOV-SinglePrototype'
        }

        lines = [
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "\\textbf{Configuration} & "
            "\\textbf{AUC} & "
            "\\textbf{EER} & "
            "\\textbf{TAR@FAR=1\\%} & "
            "\\textbf{Updates} \\\\",
            "\\hline"
        ]

        for config_key, metrics in ablation.items():
            display = display_names.get(
                config_key, config_key
            )
            auc_str = self._fmt(
                metrics['auc_mean'],
                metrics['auc_std']
            )
            eer_str = self._fmt(
                metrics['eer_mean'],
                metrics['eer_std']
            )
            tar_str = self._fmt(
                metrics['tar_at_far1_mean'],
                metrics['tar_at_far1_std']
            )
            upd_str = self._fmt_int(
                metrics['total_updates_mean'],
                metrics['total_updates_std']
            )
            lines.append(
                f"{display} & {auc_str} & "
                f"{eer_str} & {tar_str} & "
                f"{upd_str} \\\\"
            )

        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return '\n'.join(lines)

    def generate_cross_dataset_table(
        self,
        cross_results: dict,
        dataset_name: str,
        caption: str = None,
        label: str = None
    ) -> str:
        """
        Generate LaTeX table for cross-dataset results.

        Parameters
        ----------
        cross_results : dict
            Cross-dataset results for one dataset.
        dataset_name : str
            Dataset name for caption and label.
        caption : str, optional
            Table caption.
        label : str, optional
            LaTeX label.

        Returns
        -------
        str
            Complete LaTeX table string.
        """
        if caption is None:
            caption = (
                f"Verification performance on "
                f"{dataset_name.upper()} under continuous "
                f"verification with thresholds calibrated "
                f"on VGGFace2."
            )
        if label is None:
            label = f"tab:{dataset_name}-results"

        method_display = {
            'static': 'Static Enrollment',
            'ols': 'Naive OLS Expansion',
            'replay': 'Replay Dual Memory',
            'buffer': 'Fixed Buffer Averaging',
            'cocov': 'COCOV'
        }

        lines = [
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "\\textbf{Method} & "
            "\\textbf{AUC} & "
            "\\textbf{EER} & "
            "\\textbf{TAR@FAR=1\\%} & "
            "\\textbf{Updates} \\\\",
            "\\hline"
        ]

        for method_key, metrics in cross_results.items():
            display = method_display.get(
                method_key, method_key
            )
            auc_str = self._fmt(
                metrics['auc_mean'],
                metrics['auc_std']
            )
            eer_str = self._fmt(
                metrics['eer_mean'],
                metrics['eer_std']
            )
            tar_str = self._fmt(
                metrics['tar_at_far1_mean'],
                metrics['tar_at_far1_std']
            )
            upd_str = self._fmt_int(
                metrics['total_updates_mean'],
                metrics['total_updates_std']
            )
            lines.append(
                f"{display} & {auc_str} & "
                f"{eer_str} & {tar_str} & "
                f"{upd_str} \\\\"
            )

        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return '\n'.join(lines)

    def generate_significance_table(
        self,
        test_results: dict,
        caption: str = None,
        label: str = "tab:significance"
    ) -> str:
        """
        Generate LaTeX table of pairwise significance tests.

        Parameters
        ----------
        test_results : dict
            Results from pairwise_significance_tests().
        caption : str, optional
            Table caption.
        label : str
            LaTeX label.

        Returns
        -------
        str
            Complete LaTeX table string.
        """
        if caption is None:
            caption = (
                "Pairwise statistical significance tests "
                "on AUC across five runs. "
                "$^*$: $p < 0.05$. "
                "PT: paired $t$-test. "
                "WX: Wilcoxon signed-rank test."
            )

        lines = [
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{llcccc}",
            "\\hline",
            "\\textbf{Method 1} & "
            "\\textbf{Method 2} & "
            "\\textbf{Test} & "
            "\\textbf{$p$-value} & "
            "\\textbf{Cohen's $d$} & "
            "\\textbf{Sig.} \\\\",
            "\\hline"
        ]

        for pair_key, result in test_results.items():
            test_abbr = (
                'PT' if result['test_type'] == 'paired_t'
                else 'WX'
            )
            sig = (
                '$^*$' if result['significant'] else ''
            )
            p_str = (
                f"$< 0.001$"
                if result['p_value'] < 0.001
                else f"${result['p_value']:.3f}$"
            )
            lines.append(
                f"{result['method_1']} & "
                f"{result['method_2']} & "
                f"{test_abbr} & "
                f"{p_str} & "
                f"${result['cohens_d']:.3f}$ & "
                f"{sig} \\\\"
            )

        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])

        return '\n'.join(lines)

    # --------------------------------------------------
    # Summary Report
    # --------------------------------------------------

    def generate_summary_report(self) -> str:
        """
        Generate a plain-text summary report of all results.

        Loads all available result files and prints a
        formatted summary suitable for review.

        Returns
        -------
        str
            Formatted summary report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("COCOV EXPERIMENTAL RESULTS SUMMARY")
        lines.append("=" * 60)

        # Main results
        try:
            agg = self.load_aggregated_results()
            lines.append("\nMAIN RESULTS (VGGFace2)")
            lines.append("-" * 60)
            lines.append(
                f"{'Method':<25} {'AUC':>8} "
                f"{'EER':>8} {'TAR@1%':>8} "
                f"{'Updates':>10}"
            )
            lines.append("-" * 60)
            for method, metrics in agg.items():
                lines.append(
                    f"{method:<25} "
                    f"{metrics['auc_mean']:>8.4f} "
                    f"{metrics['eer_mean']:>8.4f} "
                    f"{metrics['tar_at_far1_mean']:>8.4f} "
                    f"{metrics['total_updates_mean']:>10.0f}"
                )
        except FileNotFoundError:
            lines.append(
                "\nMain results not yet available."
            )

        # Ablation results
        try:
            abl = self.load_ablation_results()
            lines.append("\nABLATION RESULTS")
            lines.append("-" * 60)
            for config, metrics in abl.items():
                lines.append(
                    f"{config:<30} "
                    f"AUC={metrics['auc_mean']:.4f} "
                    f"EER={metrics['eer_mean']:.4f}"
                )
        except FileNotFoundError:
            lines.append(
                "\nAblation results not yet available."
            )

        # Cross-dataset results
        try:
            cross = self.load_cross_dataset_results()
            for dataset, results in cross.items():
                lines.append(
                    f"\nCROSS-DATASET: {dataset.upper()}"
                )
                lines.append("-" * 60)
                for method, metrics in results.items():
                    lines.append(
                        f"{method:<25} "
                        f"AUC={metrics['auc_mean']:.4f} "
                        f"EER={metrics['eer_mean']:.4f}"
                    )
        except FileNotFoundError:
            lines.append(
                "\nCross-dataset results not yet available."
            )

        # Calibration
        try:
            cal = self.load_calibration_results()
            lines.append("\nCALIBRATION")
            lines.append("-" * 60)
            lines.append(
                f"tau_ver:  {cal['tau_ver_optimal']:.4f}"
            )
            lines.append(
                f"tau_delta:{cal['tau_delta_optimal']:.4f}"
            )
            lines.append(
                f"EER at calibration: "
                f"{cal['eer_at_calibration']:.4f}"
            )
        except FileNotFoundError:
            lines.append(
                "\nCalibration results not yet available."
            )

        lines.append("\n" + "=" * 60)
        return '\n'.join(lines)

    def save_all_tables(self) -> None:
        """
        Generate and save all LaTeX tables to results_dir.
        """
        tables_dir = self.results_dir / 'latex_tables'
        tables_dir.mkdir(exist_ok=True)

        # Main results table
        try:
            agg = self.load_aggregated_results()
            table = self.generate_main_results_table(agg)
            path = tables_dir / 'main_results.tex'
            with open(path, 'w') as f:
                f.write(table)
            logger.info(f"Saved: {path}")
        except FileNotFoundError as e:
            logger.warning(str(e))

        # Ablation table
        try:
            abl = self.load_ablation_results()
            table = self.generate_ablation_table(abl)
            path = tables_dir / 'ablation_results.tex'
            with open(path, 'w') as f:
                f.write(table)
            logger.info(f"Saved: {path}")
        except FileNotFoundError as e:
            logger.warning(str(e))

        # Cross-dataset tables
        try:
            cross = self.load_cross_dataset_results()
            for dataset, results in cross.items():
                table = self.generate_cross_dataset_table(
                    results, dataset
                )
                path = (
                    tables_dir /
                    f'{dataset}_results.tex'
                )
                with open(path, 'w') as f:
                    f.write(table)
                logger.info(f"Saved: {path}")
        except FileNotFoundError as e:
            logger.warning(str(e))

        logger.info(
            f"All tables saved to {tables_dir}"
        )

    # --------------------------------------------------
    # Formatting Utilities
    # --------------------------------------------------

    def _fmt(
        self,
        mean: float,
        std: float,
        bold: bool = False,
        decimals: int = 4
    ) -> str:
        """
        Format mean ± std for LaTeX table cell.

        Parameters
        ----------
        mean : float
            Mean value.
        std : float
            Standard deviation.
        bold : bool
            If True, wrap in \\textbf{}.
        decimals : int
            Decimal places.

        Returns
        -------
        str
            Formatted LaTeX string.
        """
        fmt = f"{{:.{decimals}f}}"
        value_str = (
            f"{fmt.format(mean)} "
            f"$\\pm$ {fmt.format(std)}"
        )
        if bold:
            return f"\\textbf{{{value_str}}}"
        return value_str

    def _fmt_int(
        self,
        mean: float,
        std: float,
        bold: bool = False
    ) -> str:
        """
        Format integer mean ± std for LaTeX table cell.

        Parameters
        ----------
        mean : float
            Mean value (displayed as integer).
        std : float
            Standard deviation (displayed as integer).
        bold : bool
            If True, wrap in \\textbf{}.

        Returns
        -------
        str
            Formatted LaTeX string.
        """
        value_str = (
            f"{mean:.0f} $\\pm$ {std:.0f}"
        )
        if bold:
            return f"\\textbf{{{value_str}}}"
        return value_str
