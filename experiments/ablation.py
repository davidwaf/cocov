"""
experiments/ablation.py
-----------------------
Ablation study for the COCOV framework.

Evaluates the contribution of individual COCOV components
by systematically disabling or modifying each one while
holding all others fixed. This isolates the effect of each
design decision on verification performance.

Ablation configurations evaluated:

    1. COCOV-Full
       Complete COCOV framework. Reference condition.

    2. COCOV-NoDrift
       Drift gate disabled. All accepted observations
       trigger prototype updates regardless of drift value.
       Isolates the contribution of drift-gated selectivity.

    3. COCOV-NoMerge
       Prototype merging disabled. Prototypes are never
       consolidated after insertion or assignment.
       Isolates the contribution of memory consolidation.

    4. COCOV-NoReviewer
       Reviewer escalation disabled. Escalated observations
       are discarded rather than confirmed or rejected.
       Isolates the contribution of collaborative input.

    5. COCOV-UnboundedMemory
       K_max removed (set to a very large value).
       Prototype count grows without bound.
       Isolates the contribution of bounded memory.

    6. COCOV-SinglePrototype
       K_max set to 1. Identity represented by a single
       adaptive prototype throughout.
       Tests whether multiple prototypes are necessary.

All ablation configurations use the same calibrated
thresholds as the full COCOV evaluation.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import VGGFace2Dataset
from data.embeddings import EmbeddingCache
from data.stream import VerificationStream
from models.encoder import FaceEncoder
from models.identity_memory import IdentityMemory
from methods.base import BaseVerificationMethod, MethodResult
from methods.cocov import COCOV
from verification.metrics import MetricsCalculator

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/data/logs/ablation.log')
    ]
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Ablation Variants
# ----------------------------------------------------------

class COCOVNoDrift(COCOV):
    """
    COCOV with drift gate disabled.

    All automatically accepted observations trigger prototype
    updates regardless of their drift value. The drift
    threshold tau_delta is not applied to update decisions.
    Reviewer escalation is still triggered by the similarity
    threshold alone.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method_name = "COCOV-NoDrift"

    def verify_and_update(
        self,
        embedding: np.ndarray,
        claimed_identity_id: str,
        is_genuine: bool,
        sequence_position: int,
        identity_sequence_position: int
    ) -> MethodResult:
        """
        Verify and update without drift gating.

        Escalation is triggered only by similarity threshold.
        All accepted observations update memory regardless
        of drift magnitude.
        """
        if not self._memory.is_enrolled(claimed_identity_id):
            return MethodResult(
                identity_id=claimed_identity_id,
                similarity=-1.0,
                drift=2.0,
                accepted=False,
                is_genuine=is_genuine,
                update_performed='none',
                sequence_position=sequence_position,
                identity_sequence_position=identity_sequence_position
            )

        centre = self._memory.get_centre(
            claimed_identity_id
        ).numpy()
        similarity = self._compute_similarity(embedding, centre)
        drift = self._compute_drift(embedding, centre)
        accepted = similarity >= self.verification_threshold

        # Escalation based on similarity only --- no drift gate
        escalated = similarity < self.verification_threshold
        update_op = 'no_update'

        if not escalated:
            # Apply update bypassing drift gate entirely
            update_op = self._apply_update(
                claimed_identity_id, embedding,
                drift=0.0  # Force drift=0 to bypass gate
            )
        elif escalated and self.simulate_reviewer:
            self._n_escalated += 1
            if is_genuine:
                self._n_reviewer_confirmed += 1
                update_op = self._apply_update(
                    claimed_identity_id, embedding,
                    drift=0.0,
                    reviewer_confirmed=True
                )
            else:
                self._n_reviewer_rejected += 1
                update_op = 'reviewer_rejected'

        result = MethodResult(
            identity_id=claimed_identity_id,
            similarity=similarity,
            drift=drift,
            accepted=accepted,
            is_genuine=is_genuine,
            escalated=escalated,
            update_performed=update_op,
            sequence_position=sequence_position,
            identity_sequence_position=identity_sequence_position
        )
        self._results.append(result)
        return result


class COCOVNoMerge(COCOV):
    """
    COCOV with prototype merging disabled.

    Prototype assignment and insertion operate normally.
    Merging is never applied. Prototype count grows until
    K_max is reached without consolidation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method_name = "COCOV-NoMerge"
        # Override identity memory with merge threshold
        # set to 1.0 (impossible to trigger)
        proto_cfg = {
            'embedding_dim': 512,
            'max_prototypes': self._memory.max_prototypes,
            'assign_threshold': self._memory.assign_threshold,
            'new_threshold': self._memory.new_threshold,
            'merge_threshold': 1.0,  # Never merges
            'momentum': self._memory.momentum
        }
        self._memory = IdentityMemory(**proto_cfg)


class COCOVNoReviewer(COCOV):
    """
    COCOV with reviewer escalation disabled.

    Escalated observations are discarded without memory
    update rather than being forwarded to a reviewer.
    Isolates the contribution of the collaborative component.
    """

    def __init__(self, **kwargs):
        kwargs['simulate_reviewer'] = False
        super().__init__(**kwargs)
        self.method_name = "COCOV-NoReviewer"

    def verify_and_update(
        self,
        embedding: np.ndarray,
        claimed_identity_id: str,
        is_genuine: bool,
        sequence_position: int,
        identity_sequence_position: int
    ) -> MethodResult:
        """
        Verify and update. Escalated observations are
        discarded without reviewer input.
        """
        if not self._memory.is_enrolled(claimed_identity_id):
            return MethodResult(
                identity_id=claimed_identity_id,
                similarity=-1.0,
                drift=2.0,
                accepted=False,
                is_genuine=is_genuine,
                update_performed='none',
                sequence_position=sequence_position,
                identity_sequence_position=identity_sequence_position
            )

        centre = self._memory.get_centre(
            claimed_identity_id
        ).numpy()
        similarity = self._compute_similarity(embedding, centre)
        drift = self._compute_drift(embedding, centre)
        accepted = similarity >= self.verification_threshold

        escalated = (
            similarity < self.verification_threshold or
            drift > self.drift_threshold
        )

        update_op = 'no_update'

        if not escalated:
            update_op = self._apply_update(
                claimed_identity_id, embedding, drift
            )
        else:
            # Discard escalated observations --- no reviewer
            self._n_escalated += 1
            update_op = 'discarded'

        result = MethodResult(
            identity_id=claimed_identity_id,
            similarity=similarity,
            drift=drift,
            accepted=accepted,
            is_genuine=is_genuine,
            escalated=escalated,
            update_performed=update_op,
            sequence_position=sequence_position,
            identity_sequence_position=identity_sequence_position
        )
        self._results.append(result)
        return result


class COCOVUnboundedMemory(COCOV):
    """
    COCOV with unbounded prototype memory.

    K_max is set to a large value (999) effectively removing
    the capacity constraint. Prototype count grows without
    bound subject only to insertion threshold conditions.
    """

    def __init__(self, **kwargs):
        kwargs['max_prototypes'] = 999
        super().__init__(**kwargs)
        self.method_name = "COCOV-UnboundedMemory"


class COCOVSinglePrototype(COCOV):
    """
    COCOV with a single prototype per identity.

    K_max is set to 1. The identity is represented by
    a single adaptive prototype updated via assignment
    at every accepted observation. No insertion or
    merging occurs.
    """

    def __init__(self, **kwargs):
        kwargs['max_prototypes'] = 1
        super().__init__(**kwargs)
        self.method_name = "COCOV-SinglePrototype"


# ----------------------------------------------------------
# Ablation Runner
# ----------------------------------------------------------

def build_ablation_methods(
    tau_ver: float,
    tau_delta: float,
    config: dict
) -> dict:
    """
    Instantiate all ablation configurations.

    Parameters
    ----------
    tau_ver : float
        Calibrated verification threshold.
    tau_delta : float
        Calibrated drift threshold.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    dict
        Maps configuration name to method instance.
    """
    proto = config['prototype']
    base_kwargs = dict(
        verification_threshold=tau_ver,
        drift_threshold=tau_delta,
        assign_threshold=proto['assign_threshold'],
        new_threshold=proto['new_threshold'],
        merge_threshold=proto['merge_threshold'],
        momentum=proto['momentum'],
        max_prototypes=proto['max_prototypes'],
        simulate_reviewer=True
    )

    return {
        'cocov_full': COCOV(**base_kwargs),
        'cocov_no_drift': COCOVNoDrift(**base_kwargs),
        'cocov_no_merge': COCOVNoMerge(**base_kwargs),
        'cocov_no_reviewer': COCOVNoReviewer(**base_kwargs),
        'cocov_unbounded': COCOVUnboundedMemory(**base_kwargs),
        'cocov_single_proto': COCOVSinglePrototype(**base_kwargs)
    }


def run_ablation(config_path: str) -> None:
    """
    Execute ablation study across all configurations.

    Uses the same evaluation infrastructure as the main
    experiment runner. Results are saved separately from
    main experiment results.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("=== COCOV Ablation Study ===")

    # Load calibration results
    results_dir = Path(config['paths']['results_dir'])
    cal_path = results_dir / 'calibration_results.json'

    if not cal_path.exists():
        raise FileNotFoundError(
            "Calibration results not found. "
            "Run run_experiment.py first."
        )

    with open(cal_path, 'r') as f:
        cal_results = json.load(f)

    tau_ver = cal_results['tau_ver_optimal']
    tau_delta = cal_results['tau_delta_optimal']
    logger.info(
        f"tau_ver={tau_ver:.4f}, tau_delta={tau_delta:.4f}"
    )

    # Encoder and dataset
    encoder = FaceEncoder(
        device=config['encoder']['device']
    )
    ds = VGGFace2Dataset(
        root=config['paths']['vggface2_root'],
        min_images=config['dataset']['vggface2'][
            'min_images_per_identity'
        ]
    )

    # Use same evaluation identities as main experiment
    eval_ids = ds.select_identities(
        n_identities=config['dataset']['vggface2'][
            'n_identities'
        ],
        seed=config['dataset']['vggface2']['random_seed']
    )

    calculator = MetricsCalculator()
    n_runs = config['experiment']['n_runs']
    seed_base = config['experiment']['random_seed_base']

    # Results per ablation configuration across runs
    all_run_metrics = {
        name: [] for name in [
            'cocov_full',
            'cocov_no_drift',
            'cocov_no_merge',
            'cocov_no_reviewer',
            'cocov_unbounded',
            'cocov_single_proto'
        ]
    }

    for run_idx in range(n_runs):
        run_seed = seed_base + run_idx
        logger.info(
            f"\n--- Ablation Run {run_idx + 1}/{n_runs} "
            f"(seed={run_seed}) ---"
        )

        # Build partition
        partition = ds.build_partition(
            identity_ids=eval_ids,
            enrollment_size=config['dataset'][
                'vggface2'
            ]['enrollment_size'],
            impostor_ratio=config['experiment'][
                'impostor_ratio'
            ],
            seed=run_seed
        )

        # Load cached embeddings
        cache = EmbeddingCache(
            cache_dir=config['paths']['embeddings_dir'],
            encoder=encoder,
            dataset_name='vggface2'
        )
        cache.extract_and_cache(partition, batch_size=64)
        loaded = cache.load_partition_embeddings(partition)
        impostors = cache.load_impostor_embeddings(
            partition, loaded
        )

        # Build stream
        stream = VerificationStream(seed=run_seed)
        stream.build(partition, loaded, impostors)

        # Instantiate ablation methods
        methods = build_ablation_methods(
            tau_ver, tau_delta, config
        )

        for config_name, method in methods.items():
            logger.info(
                f"Running {method.method_name}..."
            )
            method.reset()

            # Enrol
            for identity_id, data in loaded.items():
                method.enroll(
                    identity_id, data['enrollment']
                )

            similarities = []
            labels = []
            drifts = []
            n_escalated = 0

            # Process stream
            for event in tqdm(
                stream,
                desc=f"  {method.method_name}",
                leave=False
            ):
                result = method.verify_and_update(
                    embedding=event.embedding,
                    claimed_identity_id=(
                        event.claimed_identity_id
                    ),
                    is_genuine=event.is_genuine,
                    sequence_position=event.sequence_position,
                    identity_sequence_position=(
                        event.identity_sequence_position
                    )
                )
                similarities.append(result.similarity)
                labels.append(
                    1 if event.is_genuine else 0
                )
                drifts.append(result.drift)
                if result.escalated:
                    n_escalated += 1

            metrics = calculator.compute_run_metrics(
                similarities=similarities,
                labels=labels,
                drift_values=drifts,
                update_counts=method.get_update_counts(),
                n_escalated=n_escalated
            )
            all_run_metrics[config_name].append(metrics)

            logger.info(
                f"  AUC={metrics.auc:.4f} "
                f"EER={metrics.eer:.4f} "
                f"TAR@1%={metrics.tar_at_far1:.4f} "
                f"Updates={metrics.total_updates}"
            )

    # Aggregate and save
    logger.info("\n=== Ablation Results ===")
    ablation_output = {}
    latex_rows = []

    display_names = {
        'cocov_full': 'COCOV (Full)',
        'cocov_no_drift': 'COCOV-NoDrift',
        'cocov_no_merge': 'COCOV-NoMerge',
        'cocov_no_reviewer': 'COCOV-NoReviewer',
        'cocov_unbounded': 'COCOV-Unbounded',
        'cocov_single_proto': 'COCOV-SinglePrototype'
    }

    for config_name, run_metrics_list in \
            all_run_metrics.items():
        agg = calculator.aggregate(run_metrics_list)
        ablation_output[config_name] = {
            'auc_mean': agg.auc_mean,
            'auc_std': agg.auc_std,
            'eer_mean': agg.eer_mean,
            'eer_std': agg.eer_std,
            'tar_at_far1_mean': agg.tar_at_far1_mean,
            'tar_at_far1_std': agg.tar_at_far1_std,
            'total_updates_mean': agg.total_updates_mean,
            'total_updates_std': agg.total_updates_std
        }

        display = display_names[config_name]
        latex_rows.append(
            calculator.format_table_row(display, agg)
        )

        logger.info(
            f"{display}: "
            f"AUC={agg.auc_mean:.4f}±{agg.auc_std:.4f} "
            f"EER={agg.eer_mean:.4f}±{agg.eer_std:.4f}"
        )

    # Save JSON results
    ablation_path = results_dir / 'ablation_results.json'
    with open(ablation_path, 'w') as f:
        json.dump(ablation_output, f, indent=2)

    # Save LaTeX table
    latex_path = results_dir / 'ablation_table.tex'
    with open(latex_path, 'w') as f:
        f.write(
            "% Auto-generated ablation table\n"
            "% Chapter 6, Section: Ablation Study\n"
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\caption{Ablation study: contribution of "
            "individual COCOV components.}\n"
            "\\label{tab:ablation}\n"
            "\\begin{tabular}{lcccc}\n"
            "\\hline\n"
            "\\textbf{Configuration} & "
            "\\textbf{AUC} & "
            "\\textbf{EER} & "
            "\\textbf{TAR@FAR=1\\%} & "
            "\\textbf{Updates} \\\\\n"
            "\\hline\n"
        )
        for row in latex_rows:
            f.write(row + '\n')
        f.write(
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )

    logger.info(f"Ablation results saved to {results_dir}")
    logger.info("=== Ablation study complete ===")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Run COCOV ablation study'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='/opt/code/cocov/config/config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    run_ablation(args.config)
