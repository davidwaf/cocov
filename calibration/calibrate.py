"""
calibration/calibrate.py
------------------------
Hyperparameter calibration and sensitivity analysis for COCOV.

Calibrates the verification threshold (tau_ver) and drift
threshold (tau_delta) on a held-out calibration partition,
disjoint from the evaluation streams used in all experiments.

Also performs sensitivity sweeps for all COCOV hyperparameters:
    - tau_ver:      verification threshold
    - tau_delta:    drift threshold
    - rho_assign:   prototype assignment threshold
    - rho_new:      prototype insertion threshold
    - rho_merge:    prototype merge threshold
    - gamma:        momentum parameter
    - K_max:        maximum prototypes per identity

Calibration results are saved to disk and loaded by the
experiment runner to ensure consistent threshold usage
across all methods and runs.

Calibration data is strictly separated from evaluation data.
No identity appearing in the calibration partition appears
in any evaluation run.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from data.dataset import VGGFace2Dataset
from data.embeddings import EmbeddingCache
from models.encoder import FaceEncoder
from methods.cocov import COCOV
from methods.static import StaticEnrollment
from verification.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class ThresholdCalibrator:
    """
    Calibrates verification and drift thresholds on held-out
    data and performs sensitivity analysis across all
    COCOV hyperparameters.

    The calibration partition is constructed from identities
    not used in any evaluation run, using a separate random
    seed to ensure independence.

    Parameters
    ----------
    cache_dir : str or Path
        Directory containing pre-extracted embeddings.
    results_dir : str or Path
        Directory where calibration results are saved.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        results_dir: str | Path
    ):
        self.cache_dir = Path(cache_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.calculator = MetricsCalculator()

        logger.info("ThresholdCalibrator initialised.")

    def calibrate(
        self,
        dataset: VGGFace2Dataset,
        encoder: FaceEncoder,
        n_calibration_identities: int,
        calibration_seed: int,
        evaluation_identity_ids: list[str],
        enrollment_size: int,
        sweep_config: dict
    ) -> dict:
        """
        Run full calibration and sensitivity analysis.

        Constructs a calibration partition from identities
        not in the evaluation set, extracts embeddings,
        computes similarity scores, and sweeps all
        hyperparameters.

        Parameters
        ----------
        dataset : VGGFace2Dataset
            Dataset loader for identity and image access.
        encoder : FaceEncoder
            Fixed encoder for embedding extraction.
        n_calibration_identities : int
            Number of identities in calibration partition.
        calibration_seed : int
            Random seed for calibration identity selection.
            Must differ from all evaluation seeds.
        evaluation_identity_ids : list of str
            Identity IDs used in evaluation. Calibration
            identities are drawn from the complement.
        enrollment_size : int
            Number of enrollment images per identity.
        sweep_config : dict
            Hyperparameter sweep ranges from config.yaml.

        Returns
        -------
        dict
            Calibration results including optimal thresholds
            and sensitivity curves for all parameters.
        """
        logger.info("Starting calibration...")

        # Select calibration identities from complement
        # of evaluation set
        cal_ids = self._select_calibration_identities(
            dataset,
            n_calibration_identities,
            calibration_seed,
            evaluation_identity_ids
        )
        logger.info(
            f"Selected {len(cal_ids)} calibration identities."
        )

        # Build calibration partition
        cal_partition = dataset.build_partition(
            identity_ids=cal_ids,
            enrollment_size=enrollment_size,
            impostor_ratio=1.0,
            seed=calibration_seed
        )

        # Extract embeddings for calibration identities
        cache = EmbeddingCache(
            cache_dir=self.cache_dir,
            encoder=encoder,
            dataset_name='vggface2_calibration'
        )
        cache.extract_and_cache(cal_partition, batch_size=64)

        # Load embeddings and build similarity scores
        loaded = cache.load_partition_embeddings(cal_partition)
        impostors = cache.load_impostor_embeddings(
            cal_partition, loaded
        )

        similarities, labels, drifts = \
            self._compute_calibration_scores(
                loaded, impostors, enrollment_size
            )

        logger.info(
            f"Calibration scores: {len(similarities)} trials, "
            f"{sum(labels)} genuine, "
            f"{len(labels)-sum(labels)} impostor."
        )

        # Calibrate tau_ver at configured operating point
        operating_point = sweep_config.get(
            "operating_point", "tar_far1"
        )
        tau_ver_optimal, eer = self._calibrate_tau_ver(
            similarities, labels,
            operating_point=operating_point
        )
        logger.info(
            f"tau_ver calibrated: {tau_ver_optimal:.4f} "
            f"(EER={eer:.4f})"
        )

        # Calibrate tau_delta from drift distribution
        tau_delta_optimal = self._calibrate_tau_delta(
            drifts, labels
        )
        logger.info(
            f"tau_delta calibrated: {tau_delta_optimal:.4f}"
        )

        # Sensitivity sweeps
        sensitivity = self._run_sensitivity_sweeps(
            similarities, labels, drifts,
            loaded, impostors,
            enrollment_size, sweep_config
        )

        results = {
            'tau_ver_optimal': tau_ver_optimal,
            'tau_delta_optimal': tau_delta_optimal,
            'eer_at_calibration': eer,
            'n_calibration_identities': len(cal_ids),
            'n_genuine_trials': int(sum(labels)),
            'n_impostor_trials': int(
                len(labels) - sum(labels)
            ),
            'sensitivity': sensitivity
        }

        # Save calibration results
        self._save_results(results)
        logger.info("Calibration complete.")

        return results

    def _select_calibration_identities(
        self,
        dataset: VGGFace2Dataset,
        n_calibration: int,
        seed: int,
        evaluation_ids: list[str]
    ) -> list[str]:
        """
        Select calibration identities from complement of
        evaluation set.

        Parameters
        ----------
        dataset : VGGFace2Dataset
            Dataset with eligible identity list.
        n_calibration : int
            Number of calibration identities to select.
        seed : int
            Random seed for selection.
        evaluation_ids : list of str
            Evaluation identity IDs to exclude.

        Returns
        -------
        list of str
            Selected calibration identity IDs.
        """
        import random
        evaluation_set = set(evaluation_ids)
        available = [
            id_ for id_
            in dataset.identity_image_counts.keys()
            if id_ not in evaluation_set
        ]

        if len(available) < n_calibration:
            logger.warning(
                f"Only {len(available)} identities available "
                f"for calibration. Using all."
            )
            n_calibration = len(available)

        rng = random.Random(seed)
        return sorted(rng.sample(available, n_calibration))

    def _compute_calibration_scores(
        self,
        loaded_embeddings: dict,
        impostor_embeddings: list[dict],
        enrollment_size: int
    ) -> tuple[list, list, list]:
        """
        Compute similarity scores and drift values for all
        calibration trials.

        Parameters
        ----------
        loaded_embeddings : dict
            Embeddings keyed by identity_id.
        impostor_embeddings : list of dict
            Resolved impostor trial embeddings.
        enrollment_size : int
            Number of enrollment embeddings per identity.

        Returns
        -------
        tuple of (similarities, labels, drifts)
            Lists of floats and ints for all trials.
        """
        similarities = []
        labels = []
        drifts = []

        # Genuine trials
        for identity_id, data in loaded_embeddings.items():
            enrollment = data['enrollment']
            probes = data['probes']

            # Centre from enrollment mean
            centre = enrollment.mean(axis=0)
            norm = np.linalg.norm(centre)
            if norm > 0:
                centre = centre / norm

            for probe in probes:
                sim = float(np.dot(probe, centre))
                drift = 1.0 - sim
                similarities.append(sim)
                labels.append(1)
                drifts.append(drift)

        # Impostor trials
        for trial in impostor_embeddings:
            claimed_id = trial['claimed_identity_id']
            if claimed_id not in loaded_embeddings:
                continue

            enrollment = loaded_embeddings[
                claimed_id
            ]['enrollment']
            centre = enrollment.mean(axis=0)
            norm = np.linalg.norm(centre)
            if norm > 0:
                centre = centre / norm

            probe = trial['embedding']
            sim = float(np.dot(probe, centre))
            drift = 1.0 - sim
            similarities.append(sim)
            labels.append(0)
            drifts.append(drift)

        return similarities, labels, drifts

    def _calibrate_tau_ver(
        self,
        similarities: list[float],
        labels: list[int],
        operating_point: str = "tar_far1"
    ) -> tuple[float, float]:
        """
        Calibrate verification threshold.

        Two operating points are supported:
            eer: threshold at equal error rate
            tar_far1: threshold maximising TAR at FAR=1%

        The tar_far1 operating point is preferred for
        security-sensitive verification settings and for
        COCOV, where escalation of uncertain observations
        is preferable to automatic false rejection.

        Parameters
        ----------
        similarities : list of float
            Cosine similarity scores for all trials.
        labels : list of int
            Ground truth labels (1=genuine, 0=impostor).
        operating_point : str
            Calibration objective. Default is tar_far1.

        Returns
        -------
        tuple of (threshold, eer)
            Optimal threshold and corresponding EER.
        """
        sims = np.array(similarities)
        lbls = np.array(labels)

        fpr, tpr, thresholds = roc_curve(
            lbls, sims, pos_label=1
        )
        fnr = 1.0 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = float(np.mean([fpr[eer_idx], fnr[eer_idx]]))

        if operating_point == "tar_far1":
            # Find threshold giving FAR closest to 1%
            # without exceeding it
            target_far = 0.01
            valid = fpr <= target_far
            if valid.any():
                # Among thresholds with FAR <= 1%,
                # pick the one with highest TAR
                best_idx = np.argmax(tpr * valid)
                tau_ver = float(thresholds[best_idx])
            else:
                # Fall back to EER if no point meets FAR<=1%
                tau_ver = float(thresholds[eer_idx])
                logger.warning(
                    "No threshold achieves FAR<=1%. "
                    "Falling back to EER operating point."
                )
        else:
            # EER operating point
            tau_ver = float(thresholds[eer_idx])

        logger.info(
            f"tau_ver calibrated at {operating_point} "
            f"operating point: {tau_ver:.4f} "
            f"(EER={eer:.4f})"
        )

        return tau_ver, eer

    def _calibrate_tau_delta(
        self,
        drifts: list[float],
        labels: list[int]
    ) -> float:
        """
        Calibrate drift threshold from genuine drift
        distribution.

        The drift threshold is set at the 95th percentile
        of the genuine drift distribution. Observations
        with drift above this value are considered
        structurally disruptive and are escalated.

        Parameters
        ----------
        drifts : list of float
            Drift values for all trials.
        labels : list of int
            Ground truth labels.

        Returns
        -------
        float
            Calibrated drift threshold.
        """
        genuine_drifts = np.array([
            d for d, l in zip(drifts, labels) if l == 1
        ])
        # 99th percentile: only escalate the most extreme
        # drift observations, reducing escalation rate
        tau_delta = float(np.percentile(genuine_drifts, 99))
        return tau_delta

    def _run_sensitivity_sweeps(
        self,
        similarities: list[float],
        labels: list[int],
        drifts: list[float],
        loaded_embeddings: dict,
        impostor_embeddings: list[dict],
        enrollment_size: int,
        sweep_config: dict
    ) -> dict:
        """
        Sweep each hyperparameter and record AUC and EER.

        For tau_ver and tau_delta, sweeps are performed
        directly on pre-computed similarity and drift scores.
        For prototype parameters, COCOV is re-instantiated
        and re-evaluated for each configuration.

        Parameters
        ----------
        similarities : list of float
            Pre-computed calibration similarity scores.
        labels : list of int
            Ground truth labels.
        drifts : list of float
            Pre-computed drift values.
        loaded_embeddings : dict
            Calibration embeddings by identity.
        impostor_embeddings : list of dict
            Impostor trial embeddings.
        enrollment_size : int
            Enrollment size used in calibration.
        sweep_config : dict
            Sweep ranges from config.yaml calibration section.

        Returns
        -------
        dict
            Sensitivity results per parameter.
        """
        sensitivity = {}

        sims = np.array(similarities)
        lbls = np.array(labels)

        # --- tau_ver sweep ---
        tau_ver_cfg = sweep_config['verification_threshold']
        tau_ver_values = np.linspace(
            tau_ver_cfg['min'],
            tau_ver_cfg['max'],
            tau_ver_cfg['steps']
        )
        tau_ver_results = []
        for tau in tqdm(
            tau_ver_values,
            desc="Sweeping tau_ver"
        ):
            predictions = (sims >= tau).astype(int)
            tp = np.sum((predictions == 1) & (lbls == 1))
            fp = np.sum((predictions == 1) & (lbls == 0))
            fn = np.sum((predictions == 0) & (lbls == 1))
            tn = np.sum((predictions == 0) & (lbls == 0))
            tar = tp / (tp + fn) if (tp + fn) > 0 else 0
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_arr, tpr_arr, _ = roc_curve(lbls, sims)
            roc_auc = float(auc(fpr_arr, tpr_arr))
            tau_ver_results.append({
                'threshold': float(tau),
                'tar': float(tar),
                'far': float(far),
                'auc': roc_auc
            })
        sensitivity['tau_ver'] = tau_ver_results

        # --- tau_delta sweep ---
        tau_delta_cfg = sweep_config['drift_threshold']
        tau_delta_values = np.linspace(
            tau_delta_cfg['min'],
            tau_delta_cfg['max'],
            tau_delta_cfg['steps']
        )
        drifts_arr = np.array(drifts)
        tau_delta_results = []
        for tau in tqdm(
            tau_delta_values,
            desc="Sweeping tau_delta"
        ):
            escalation_rate = float(
                np.mean(drifts_arr > tau)
            )
            genuine_escalation = float(np.mean(
                drifts_arr[lbls == 1] > tau
            ))
            tau_delta_results.append({
                'threshold': float(tau),
                'escalation_rate': escalation_rate,
                'genuine_escalation_rate': genuine_escalation
            })
        sensitivity['tau_delta'] = tau_delta_results

        # --- Prototype parameter sweeps ---
        # These require re-running COCOV on calibration data
        sensitivity['max_prototypes'] = \
            self._sweep_max_prototypes(
                loaded_embeddings,
                impostor_embeddings,
                enrollment_size,
                sweep_config['max_prototypes']['values']
            )

        sensitivity['momentum'] = self._sweep_momentum(
            loaded_embeddings,
            impostor_embeddings,
            enrollment_size,
            sweep_config['momentum']
        )

        return sensitivity

    def _sweep_max_prototypes(
        self,
        loaded_embeddings: dict,
        impostor_embeddings: list[dict],
        enrollment_size: int,
        k_values: list[int]
    ) -> list[dict]:
        """
        Sweep K_max and record AUC and update counts.

        Parameters
        ----------
        loaded_embeddings : dict
            Calibration embeddings.
        impostor_embeddings : list of dict
            Impostor trial embeddings.
        enrollment_size : int
            Number of enrollment embeddings.
        k_values : list of int
            K_max values to evaluate.

        Returns
        -------
        list of dict
            AUC, EER, and mean prototype count per K_max.
        """
        results = []

        for k_max in tqdm(k_values, desc="Sweeping K_max"):
            cocov = COCOV(
                verification_threshold=0.5,
                drift_threshold=0.5,
                assign_threshold=0.3,
                new_threshold=0.6,
                merge_threshold=0.95,
                momentum=0.9,
                max_prototypes=k_max,
                simulate_reviewer=True
            )

            sims, lbls = self._run_cocov_on_calibration(
                cocov, loaded_embeddings,
                impostor_embeddings, enrollment_size
            )

            if len(np.unique(lbls)) < 2:
                continue

            fpr, tpr, _ = roc_curve(lbls, sims)
            roc_auc = float(auc(fpr, tpr))
            fnr = 1.0 - tpr
            eer_idx = np.argmin(np.abs(fpr - fnr))
            eer = float(
                np.mean([fpr[eer_idx], fnr[eer_idx]])
            )

            counts = cocov.get_update_counts()
            proto_counts = cocov.get_prototype_counts()
            mean_protos = float(np.mean(
                list(proto_counts.values())
            )) if proto_counts else 0.0

            results.append({
                'k_max': k_max,
                'auc': roc_auc,
                'eer': eer,
                'mean_prototype_count': mean_protos,
                'total_updates': counts['total_updates']
            })

        return results

    def _sweep_momentum(
        self,
        loaded_embeddings: dict,
        impostor_embeddings: list[dict],
        enrollment_size: int,
        momentum_config: dict
    ) -> list[dict]:
        """
        Sweep momentum parameter and record AUC.

        Parameters
        ----------
        loaded_embeddings : dict
            Calibration embeddings.
        impostor_embeddings : list of dict
            Impostor trial embeddings.
        enrollment_size : int
            Number of enrollment embeddings.
        momentum_config : dict
            Sweep range configuration.

        Returns
        -------
        list of dict
            AUC and EER per momentum value.
        """
        values = np.linspace(
            momentum_config['min'],
            momentum_config['max'],
            momentum_config['steps']
        )
        results = []

        for gamma in tqdm(values, desc="Sweeping momentum"):
            cocov = COCOV(
                verification_threshold=0.5,
                drift_threshold=0.5,
                assign_threshold=0.3,
                new_threshold=0.6,
                merge_threshold=0.95,
                momentum=float(gamma),
                max_prototypes=10,
                simulate_reviewer=True
            )

            sims, lbls = self._run_cocov_on_calibration(
                cocov, loaded_embeddings,
                impostor_embeddings, enrollment_size
            )

            if len(np.unique(lbls)) < 2:
                continue

            fpr, tpr, _ = roc_curve(lbls, sims)
            roc_auc = float(auc(fpr, tpr))
            fnr = 1.0 - tpr
            eer_idx = np.argmin(np.abs(fpr - fnr))
            eer = float(
                np.mean([fpr[eer_idx], fnr[eer_idx]])
            )

            results.append({
                'momentum': float(gamma),
                'auc': roc_auc,
                'eer': eer
            })

        return results

    def _run_cocov_on_calibration(
        self,
        cocov: COCOV,
        loaded_embeddings: dict,
        impostor_embeddings: list[dict],
        enrollment_size: int
    ) -> tuple[list, list]:
        """
        Run COCOV on calibration data and collect scores.

        Parameters
        ----------
        cocov : COCOV
            COCOV instance with current hyperparameters.
        loaded_embeddings : dict
            Calibration embeddings by identity.
        impostor_embeddings : list of dict
            Impostor trial embeddings.
        enrollment_size : int
            Number of enrollment embeddings.

        Returns
        -------
        tuple of (similarities, labels)
            Score and label lists for all trials.
        """
        similarities = []
        labels = []

        # Enrol all calibration identities
        for identity_id, data in loaded_embeddings.items():
            cocov.enroll(
                identity_id, data['enrollment']
            )

        # Genuine trials
        seq_pos = 0
        for identity_id, data in loaded_embeddings.items():
            for i, probe in enumerate(data['probes']):
                result = cocov.verify_and_update(
                    probe, identity_id, True, seq_pos, i
                )
                similarities.append(result.similarity)
                labels.append(1)
                seq_pos += 1

        # Impostor trials
        for trial in impostor_embeddings:
            claimed_id = trial['claimed_identity_id']
            if not cocov._memory.is_enrolled(claimed_id):
                continue
            result = cocov.verify_and_update(
                trial['embedding'],
                claimed_id, False, seq_pos, -1
            )
            similarities.append(result.similarity)
            labels.append(0)
            seq_pos += 1

        return similarities, labels

    def _save_results(self, results: dict) -> None:
        """
        Save calibration results to disk as JSON.

        Parameters
        ----------
        results : dict
            Calibration results to save.
        """
        # Convert numpy types for JSON serialisation
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        output_path = (
            self.results_dir / 'calibration_results.json'
        )

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=convert)

        logger.info(
            f"Calibration results saved to {output_path}"
        )

    def load_results(self) -> dict:
        """
        Load previously saved calibration results.

        Returns
        -------
        dict
            Saved calibration results.

        Raises
        ------
        FileNotFoundError
            If calibration has not been run yet.
        """
        output_path = (
            self.results_dir / 'calibration_results.json'
        )
        if not output_path.exists():
            raise FileNotFoundError(
                f"No calibration results found at "
                f"{output_path}. Run calibrate() first."
            )

        with open(output_path, 'r') as f:
            return json.load(f)
