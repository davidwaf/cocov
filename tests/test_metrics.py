"""
tests/test_metrics.py
---------------------
Unit tests for verification metrics computation,
aggregation across runs, and LaTeX table formatting.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from verification.metrics import (
    MetricsCalculator,
    RunMetrics,
    AggregatedMetrics
)


# ----------------------------------------------------------
# Fixtures
# ----------------------------------------------------------

@pytest.fixture
def calculator():
    """Standard MetricsCalculator instance."""
    return MetricsCalculator()


def make_scores(
    n_genuine: int = 500,
    n_impostor: int = 500,
    genuine_mean: float = 0.7,
    impostor_mean: float = 0.3,
    std: float = 0.1,
    seed: int = 42
) -> tuple[list, list]:
    """
    Create synthetic similarity scores and labels.

    Returns
    -------
    tuple of (similarities, labels)
    """
    np.random.seed(seed)
    genuine = np.random.normal(genuine_mean, std, n_genuine)
    impostor = np.random.normal(impostor_mean, std, n_impostor)
    similarities = np.clip(
        np.concatenate([genuine, impostor]), -1, 1
    ).tolist()
    labels = [1] * n_genuine + [0] * n_impostor
    return similarities, labels


# ----------------------------------------------------------
# RunMetrics Computation Tests
# ----------------------------------------------------------

class TestRunMetrics:

    def test_auc_range(self, calculator):
        """AUC is in [0, 1]."""
        sims, labels = make_scores()
        metrics = calculator.compute_run_metrics(
            sims, labels
        )
        assert 0.0 <= metrics.auc <= 1.0

    def test_eer_range(self, calculator):
        """EER is in [0, 1]."""
        sims, labels = make_scores()
        metrics = calculator.compute_run_metrics(
            sims, labels
        )
        assert 0.0 <= metrics.eer <= 1.0

    def test_tar_at_far1_range(self, calculator):
        """TAR@FAR=1% is in [0, 1]."""
        sims, labels = make_scores()
        metrics = calculator.compute_run_metrics(
            sims, labels
        )
        assert 0.0 <= metrics.tar_at_far1 <= 1.0

    def test_good_separation_high_auc(self, calculator):
        """Well-separated scores produce high AUC."""
        sims, labels = make_scores(
            genuine_mean=0.9,
            impostor_mean=0.1,
            std=0.05
        )
        metrics = calculator.compute_run_metrics(
            sims, labels
        )
        assert metrics.auc > 0.99

    def test_no_separation_auc_near_half(self, calculator):
        """Overlapping scores produce AUC near 0.5."""
        sims, labels = make_scores(
            genuine_mean=0.5,
            impostor_mean=0.5,
            std=0.1
        )
        metrics = calculator.compute_run_metrics(
            sims, labels
        )
        assert 0.4 <= metrics.auc <= 0.6

    def test_trial_counts(self, calculator):
        """Genuine and impostor trial counts are correct."""
        n_genuine = 300
        n_impostor = 400
        sims, labels = make_scores(
            n_genuine=n_genuine,
            n_impostor=n_impostor
        )
        metrics = calculator.compute_run_metrics(
            sims, labels
        )
        assert metrics.n_genuine == n_genuine
        assert metrics.n_impostor == n_impostor

    def test_update_counts_stored(self, calculator):
        """Update counts are stored in metrics."""
        sims, labels = make_scores()
        update_counts = {
            'total_updates': 450,
            'assignments': 300,
            'insertions': 100,
            'merges': 50
        }
        metrics = calculator.compute_run_metrics(
            sims, labels,
            update_counts=update_counts
        )
        assert metrics.total_updates == 450
        assert metrics.assignments == 300
        assert metrics.insertions == 100
        assert metrics.merges == 50

    def test_drift_values_stored(self, calculator):
        """Drift values are stored in metrics."""
        sims, labels = make_scores()
        drifts = list(np.random.uniform(0.1, 0.9, 1000))
        metrics = calculator.compute_run_metrics(
            sims, labels, drift_values=drifts
        )
        assert len(metrics.drift_values) == 1000

    def test_escalation_rate(self, calculator):
        """Escalation rate computed correctly."""
        sims, labels = make_scores(
            n_genuine=500, n_impostor=500
        )
        n_escalated = 100
        metrics = calculator.compute_run_metrics(
            sims, labels, n_escalated=n_escalated
        )
        expected_rate = 100 / 1000
        assert abs(
            metrics.escalation_rate - expected_rate
        ) < 1e-6

    def test_single_label_raises(self, calculator):
        """Single label class raises ValueError."""
        sims = [0.5] * 100
        labels = [1] * 100  # All genuine, no impostors
        with pytest.raises(ValueError):
            calculator.compute_run_metrics(sims, labels)

    def test_roc_curve_stored(self, calculator):
        """ROC curve arrays are stored in metrics."""
        sims, labels = make_scores()
        metrics = calculator.compute_run_metrics(
            sims, labels
        )
        assert len(metrics.fpr) > 0
        assert len(metrics.tpr) > 0
        assert len(metrics.fpr) == len(metrics.tpr)

    def test_fpr_starts_at_zero(self, calculator):
        """ROC curve FPR starts at 0."""
        sims, labels = make_scores()
        metrics = calculator.compute_run_metrics(
            sims, labels
        )
        assert metrics.fpr[0] == 0.0

    def test_tpr_ends_at_one(self, calculator):
        """ROC curve TPR ends at 1."""
        sims, labels = make_scores()
        metrics = calculator.compute_run_metrics(
            sims, labels
        )
        assert metrics.tpr[-1] == 1.0


# ----------------------------------------------------------
# TAR at FAR Tests
# ----------------------------------------------------------

class TestTARatFAR:

    def test_tar_at_far_perfect_classifier(self, calculator):
        """Perfect classifier has TAR=1 at any FAR."""
        genuine = [1.0] * 500
        impostor = [-1.0] * 500
        sims = genuine + impostor
        labels = [1] * 500 + [0] * 500
        metrics = calculator.compute_run_metrics(
            sims, labels
        )
        assert metrics.tar_at_far1 > 0.99

    def test_tar_at_far_worst_classifier(self, calculator):
        """Worst classifier (inverted) has TAR near 0."""
        genuine = [-1.0] * 500
        impostor = [1.0] * 500
        sims = genuine + impostor
        labels = [1] * 500 + [0] * 500
        metrics = calculator.compute_run_metrics(
            sims, labels
        )
        assert metrics.tar_at_far1 < 0.05

    def test_tar_monotone_with_far(self, calculator):
        """TAR increases as FAR increases."""
        sims, labels = make_scores()
        fpr = np.linspace(0, 1, 100)
        from sklearn.metrics import roc_curve
        fpr_arr, tpr_arr, _ = roc_curve(
            labels, sims, pos_label=1
        )
        # TAR should be non-decreasing
        assert all(
            tpr_arr[i] <= tpr_arr[i + 1]
            for i in range(len(tpr_arr) - 1)
        )


# ----------------------------------------------------------
# Aggregation Tests
# ----------------------------------------------------------

class TestAggregation:

    def _make_run_metrics(
        self,
        calculator,
        seed: int = 42
    ) -> RunMetrics:
        sims, labels = make_scores(seed=seed)
        return calculator.compute_run_metrics(
            sims, labels,
            update_counts={
                'total_updates': 400 + seed,
                'assignments': 200,
                'insertions': 100,
                'merges': 50
            }
        )

    def test_aggregate_single_run(self, calculator):
        """Aggregation of single run has zero std."""
        metrics = self._make_run_metrics(calculator)
        agg = calculator.aggregate([metrics])
        assert agg.auc_std == 0.0
        assert agg.eer_std == 0.0
        assert agg.n_runs == 1

    def test_aggregate_multiple_runs(self, calculator):
        """Aggregation across multiple runs succeeds."""
        run_list = [
            self._make_run_metrics(calculator, seed=i)
            for i in range(5)
        ]
        agg = calculator.aggregate(run_list)
        assert agg.n_runs == 5
        assert agg.auc_mean > 0
        assert agg.auc_std >= 0

    def test_aggregate_mean_in_range(self, calculator):
        """Aggregated mean is within individual run range."""
        run_list = [
            self._make_run_metrics(calculator, seed=i)
            for i in range(5)
        ]
        individual_aucs = [m.auc for m in run_list]
        agg = calculator.aggregate(run_list)
        assert min(individual_aucs) <= agg.auc_mean
        assert agg.auc_mean <= max(individual_aucs)

    def test_aggregate_empty_raises(self, calculator):
        """Aggregating empty list raises ValueError."""
        with pytest.raises(ValueError):
            calculator.aggregate([])

    def test_aggregate_update_counts(self, calculator):
        """Update counts are averaged correctly."""
        run_list = [
            self._make_run_metrics(calculator, seed=i)
            for i in range(3)
        ]
        agg = calculator.aggregate(run_list)
        expected_mean = np.mean([
            m.total_updates for m in run_list
        ])
        assert abs(
            agg.total_updates_mean - expected_mean
        ) < 1e-6


# ----------------------------------------------------------
# LaTeX Formatting Tests
# ----------------------------------------------------------

class TestLaTeXFormatting:

    def test_format_table_row_structure(self, calculator):
        """LaTeX table row has correct structure."""
        sims, labels = make_scores()
        metrics = calculator.compute_run_metrics(
            sims, labels,
            update_counts={'total_updates': 450,
                          'assignments': 0,
                          'insertions': 0,
                          'merges': 0}
        )
        agg = calculator.aggregate([metrics])
        row = calculator.format_table_row('COCOV', agg)

        assert 'COCOV' in row
        assert '$\\pm$' in row
        assert '\\\\' in row
        assert '&' in row

    def test_format_row_has_four_columns(self, calculator):
        """LaTeX row has four metric columns."""
        sims, labels = make_scores()
        metrics = calculator.compute_run_metrics(
            sims, labels,
            update_counts={'total_updates': 100,
                          'assignments': 0,
                          'insertions': 0,
                          'merges': 0}
        )
        agg = calculator.aggregate([metrics])
        row = calculator.format_table_row('Method', agg)
        # Count ampersands: 4 columns = 4 separators
        assert row.count('&') == 4

    def test_format_row_method_name(self, calculator):
        """Method name appears in formatted row."""
        sims, labels = make_scores()
        metrics = calculator.compute_run_metrics(
            sims, labels,
            update_counts={'total_updates': 0,
                          'assignments': 0,
                          'insertions': 0,
                          'merges': 0}
        )
        agg = calculator.aggregate([metrics])
        method_name = "Static Enrollment"
        row = calculator.format_table_row(
            method_name, agg
        )
        assert method_name in row
