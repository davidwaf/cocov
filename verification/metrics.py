"""
metrics.py
----------
Evaluation metrics for continuous face verification.

Implements AUC, EER, TAR@FAR, update count aggregation,
and drift distribution statistics as defined in Chapter 6
of the accompanying thesis.

All metrics are computed from verification trial results
collected during experimental runs. Aggregation across
multiple runs uses mean and standard deviation.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    """
    Metrics computed from a single experimental run.

    Attributes
    ----------
    auc : float
        Area under the ROC curve.
    eer : float
        Equal error rate.
    tar_at_far1 : float
        True accept rate at FAR = 1%.
    fpr : np.ndarray
        False positive rates across thresholds.
    tpr : np.ndarray
        True positive rates across thresholds.
    thresholds : np.ndarray
        Decision thresholds corresponding to fpr/tpr.
    n_genuine : int
        Number of genuine trials evaluated.
    n_impostor : int
        Number of impostor trials evaluated.
    total_updates : int
        Total prototype update operations across all identities.
    assignments : int
        Total assignment operations.
    insertions : int
        Total insertion operations.
    merges : int
        Total merge operations.
    drift_values : np.ndarray
        All drift values recorded during the run.
    escalation_rate : float
        Fraction of observations escalated for review.
    """
    auc: float
    eer: float
    tar_at_far1: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    n_genuine: int
    n_impostor: int
    total_updates: int = 0
    assignments: int = 0
    insertions: int = 0
    merges: int = 0
    drift_values: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    escalation_rate: float = 0.0


@dataclass
class AggregatedMetrics:
    """
    Metrics aggregated across multiple experimental runs.

    Attributes
    ----------
    auc_mean, auc_std : float
        Mean and standard deviation of AUC across runs.
    eer_mean, eer_std : float
        Mean and standard deviation of EER across runs.
    tar_at_far1_mean, tar_at_far1_std : float
        Mean and standard deviation of TAR@FAR=1%.
    total_updates_mean, total_updates_std : float
        Mean and standard deviation of total update counts.
    drift_mean, drift_std : float
        Mean and standard deviation of drift values.
    escalation_rate_mean : float
        Mean escalation rate across runs.
    n_runs : int
        Number of runs aggregated.
    """
    auc_mean: float
    auc_std: float
    eer_mean: float
    eer_std: float
    tar_at_far1_mean: float
    tar_at_far1_std: float
    total_updates_mean: float
    total_updates_std: float
    drift_mean: float
    drift_std: float
    escalation_rate_mean: float
    n_runs: int


class MetricsCalculator:
    """
    Computes and aggregates verification performance metrics.

    Accepts lists of similarity scores and ground truth labels
    from verification trials, computes ROC-based metrics, and
    aggregates results across multiple experimental runs.
    """

    def compute_run_metrics(
        self,
        similarities: list[float],
        labels: list[int],
        drift_values: list[float] = None,
        update_counts: dict = None,
        n_escalated: int = 0
    ) -> RunMetrics:
        """
        Compute metrics from a single experimental run.

        Parameters
        ----------
        similarities : list of float
            Cosine similarity scores for all verification trials.
            Genuine trials have label 1, impostor trials label 0.
        labels : list of int
            Ground truth labels. 1 for genuine, 0 for impostor.
        drift_values : list of float, optional
            Drift values recorded at each verification event.
        update_counts : dict, optional
            Update count dictionary from IdentityMemory.
            Expected keys: total_updates, assignments,
            insertions, merges.
        n_escalated : int
            Number of observations escalated for review.

        Returns
        -------
        RunMetrics
            Computed metrics for this run.
        """
        similarities = np.array(similarities)
        labels = np.array(labels)

        if len(np.unique(labels)) < 2:
            raise ValueError(
                "Labels must contain both genuine (1) and "
                "impostor (0) trials."
            )

        # ROC curve
        fpr, tpr, thresholds = roc_curve(
            labels, similarities, pos_label=1
        )
        roc_auc = auc(fpr, tpr)

        # EER: point where FPR == FNR (1 - TPR)
        fnr = 1.0 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = float(np.mean([fpr[eer_idx], fnr[eer_idx]]))

        # TAR at FAR = 1%
        tar_at_far1 = self._tar_at_far(fpr, tpr, target_far=0.01)

        # Trial counts
        n_genuine = int(np.sum(labels == 1))
        n_impostor = int(np.sum(labels == 0))
        n_total = n_genuine + n_impostor

        # Update counts
        total_updates = 0
        assignments = 0
        insertions = 0
        merges = 0
        if update_counts is not None:
            total_updates = update_counts.get(
                'total_updates', 0
            )
            assignments = update_counts.get('assignments', 0)
            insertions = update_counts.get('insertions', 0)
            merges = update_counts.get('merges', 0)

        # Drift statistics
        drift_arr = np.array(
            drift_values if drift_values else []
        )

        # Escalation rate
        escalation_rate = (
            n_escalated / n_total if n_total > 0 else 0.0
        )

        return RunMetrics(
            auc=roc_auc,
            eer=eer,
            tar_at_far1=tar_at_far1,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            n_genuine=n_genuine,
            n_impostor=n_impostor,
            total_updates=total_updates,
            assignments=assignments,
            insertions=insertions,
            merges=merges,
            drift_values=drift_arr,
            escalation_rate=escalation_rate
        )

    def aggregate(
        self,
        run_metrics: list[RunMetrics]
    ) -> AggregatedMetrics:
        """
        Aggregate metrics across multiple experimental runs.

        Parameters
        ----------
        run_metrics : list of RunMetrics
            Metrics from each independent run.

        Returns
        -------
        AggregatedMetrics
            Mean and standard deviation across runs.
        """
        if not run_metrics:
            raise ValueError(
                "No run metrics provided for aggregation."
            )

        aucs = [m.auc for m in run_metrics]
        eers = [m.eer for m in run_metrics]
        tars = [m.tar_at_far1 for m in run_metrics]
        updates = [m.total_updates for m in run_metrics]
        escalations = [m.escalation_rate for m in run_metrics]

        all_drift = np.concatenate(
            [m.drift_values for m in run_metrics
             if len(m.drift_values) > 0]
        ) if any(
            len(m.drift_values) > 0 for m in run_metrics
        ) else np.array([0.0])

        return AggregatedMetrics(
            auc_mean=float(np.mean(aucs)),
            auc_std=float(np.std(aucs, ddof=1)
                          if len(aucs) > 1 else 0.0),
            eer_mean=float(np.mean(eers)),
            eer_std=float(np.std(eers, ddof=1)
                          if len(eers) > 1 else 0.0),
            tar_at_far1_mean=float(np.mean(tars)),
            tar_at_far1_std=float(np.std(tars, ddof=1)
                                  if len(tars) > 1 else 0.0),
            total_updates_mean=float(np.mean(updates)),
            total_updates_std=float(np.std(updates, ddof=1)
                                    if len(updates) > 1
                                    else 0.0),
            drift_mean=float(np.mean(all_drift)),
            drift_std=float(np.std(all_drift, ddof=1)
                            if len(all_drift) > 1 else 0.0),
            escalation_rate_mean=float(np.mean(escalations)),
            n_runs=len(run_metrics)
        )

    def format_table_row(
        self,
        method_name: str,
        metrics: AggregatedMetrics
    ) -> str:
        """
        Format aggregated metrics as a LaTeX table row.

        Parameters
        ----------
        method_name : str
            Name of the evaluated method.
        metrics : AggregatedMetrics
            Aggregated metrics to format.

        Returns
        -------
        str
            LaTeX table row string.
        """
        return (
            f"{method_name} & "
            f"{metrics.auc_mean:.4f} $\\pm$ "
            f"{metrics.auc_std:.4f} & "
            f"{metrics.eer_mean:.4f} $\\pm$ "
            f"{metrics.eer_std:.4f} & "
            f"{metrics.tar_at_far1_mean:.4f} $\\pm$ "
            f"{metrics.tar_at_far1_std:.4f} & "
            f"{metrics.total_updates_mean:.0f} $\\pm$ "
            f"{metrics.total_updates_std:.0f} \\\\"
        )

    def _tar_at_far(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        target_far: float = 0.01
    ) -> float:
        """
        Compute TAR at a fixed FAR operating point.

        Uses linear interpolation between the two ROC points
        bracketing the target FAR.

        Parameters
        ----------
        fpr : np.ndarray
            False positive rates from ROC curve.
        tpr : np.ndarray
            True positive rates from ROC curve.
        target_far : float
            Target false accept rate. Default 0.01 (1%).

        Returns
        -------
        float
            True accept rate at the specified FAR.
        """
        # Find indices bracketing target_far
        idx = np.searchsorted(fpr, target_far)

        if idx == 0:
            return float(tpr[0])
        if idx >= len(fpr):
            return float(tpr[-1])

        # Linear interpolation
        fpr_low, fpr_high = fpr[idx - 1], fpr[idx]
        tpr_low, tpr_high = tpr[idx - 1], tpr[idx]

        if fpr_high == fpr_low:
            return float(tpr_high)

        slope = (tpr_high - tpr_low) / (fpr_high - fpr_low)
        tar = tpr_low + slope * (target_far - fpr_low)
        return float(np.clip(tar, 0.0, 1.0))
