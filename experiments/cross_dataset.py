"""
experiments/cross_dataset.py
-----------------------------
Cross-dataset evaluation on CACD and FG-NET.

Assesses whether verification behaviour observed on VGGFace2
generalises to datasets with explicit, verifiable temporal
appearance change. Two datasets are used:

    CACD (Cross-Age Celebrity Dataset)
        2,000 identities with year-stamped images spanning
        a ten-year acquisition period. Provides medium-scale
        evaluation under known temporal drift.

    FG-NET Aging Database
        82 identities with age-labelled images from infancy
        to late adulthood. Provides controlled per-identity
        analysis of long-range age progression effects.

All methods evaluated in the main experiment are also
evaluated here under identical protocols. Calibrated
thresholds from the VGGFace2 calibration are applied
directly without re-calibration, testing threshold
transferability across datasets.

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

from data.dataset import CACDDataset, FGNETDataset
from data.embeddings import EmbeddingCache
from data.stream import VerificationStream
from models.encoder import FaceEncoder
from methods.static import StaticEnrollment
from methods.ols import NaiveOLSExpansion
from methods.replay import ReplayDualMemory
from methods.buffer import FixedBufferAveraging
from methods.cocov import COCOV
from verification.metrics import MetricsCalculator
from experiments.run_experiment import build_methods

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            '/opt/data/logs/cross_dataset.log'
        )
    ]
)
logger = logging.getLogger(__name__)


def run_on_dataset(
    dataset_name: str,
    partition,
    loaded_embeddings: dict,
    impostor_embeddings: list,
    methods: dict,
    calculator: MetricsCalculator,
    seed: int
) -> dict:
    """
    Run all methods on a single dataset partition.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset for logging.
    partition : DatasetPartition
        Dataset partition with enrollment/probe splits.
    loaded_embeddings : dict
        Cached embeddings keyed by identity_id.
    impostor_embeddings : list
        Resolved impostor trial embeddings.
    methods : dict
        Instantiated method objects.
    calculator : MetricsCalculator
        Metrics computation instance.
    seed : int
        Stream construction seed.

    Returns
    -------
    dict
        Metrics per method for this dataset.
    """
    # Build stream
    stream = VerificationStream(seed=seed)
    stream.build(partition, loaded_embeddings,
                 impostor_embeddings)

    logger.info(
        f"{dataset_name} stream: {stream.stream_stats()}"
    )

    results = {}

    for method_name, method in methods.items():
        method.reset()

        # Enrol
        for identity_id, data in loaded_embeddings.items():
            method.enroll(
                identity_id, data['enrollment']
            )

        similarities = []
        labels = []
        drifts = []
        n_escalated = 0

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
            labels.append(1 if event.is_genuine else 0)
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

        results[method_name] = metrics
        logger.info(
            f"  {method.method_name}: "
            f"AUC={metrics.auc:.4f} "
            f"EER={metrics.eer:.4f} "
            f"TAR@1%={metrics.tar_at_far1:.4f} "
            f"Updates={metrics.total_updates}"
        )

    return results


def run_cross_dataset(config_path: str) -> None:
    """
    Execute cross-dataset evaluation on CACD and FG-NET.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("=== Cross-Dataset Evaluation ===")

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
        f"Applying VGGFace2 thresholds: "
        f"tau_ver={tau_ver:.4f}, "
        f"tau_delta={tau_delta:.4f}"
    )

    encoder = FaceEncoder(
        device=config['encoder']['device']
    )
    calculator = MetricsCalculator()
    n_runs = config['experiment']['n_runs']
    seed_base = config['experiment']['random_seed_base']

    all_results = {}

    # --------------------------------------------------
    # CACD Evaluation
    # --------------------------------------------------
    logger.info("\n--- CACD Evaluation ---")

    cacd_root = Path(config['paths']['cacd_root'])
    if cacd_root.exists():
        cacd = CACDDataset(
            root=cacd_root,
            min_images=config['dataset']['cacd'][
                'min_images_per_identity'
            ]
        )
        logger.info(
            f"CACD: {cacd.n_eligible} eligible identities"
        )

        cacd_run_metrics = {
            name: [] for name in [
                'static', 'ols', 'replay',
                'buffer', 'cocov'
            ]
        }

        for run_idx in range(n_runs):
            run_seed = seed_base + run_idx
            logger.info(
                f"CACD Run {run_idx + 1}/{n_runs} "
                f"(seed={run_seed})"
            )

            partition = cacd.build_partition(
                enrollment_size=config['dataset'][
                    'cacd'
                ]['enrollment_size'],
                impostor_ratio=config['experiment'][
                    'impostor_ratio'
                ],
                seed=run_seed
            )

            cache = EmbeddingCache(
                cache_dir=config['paths']['embeddings_dir'],
                encoder=encoder,
                dataset_name='cacd'
            )
            cache.extract_and_cache(
                partition, batch_size=64
            )
            loaded = cache.load_partition_embeddings(
                partition
            )
            impostors = cache.load_impostor_embeddings(
                partition, loaded
            )

            methods = build_methods(
                config, tau_ver, tau_delta
            )
            run_results = run_on_dataset(
                'CACD', partition, loaded,
                impostors, methods, calculator,
                seed=run_seed
            )

            for method_name, metrics in \
                    run_results.items():
                cacd_run_metrics[method_name].append(
                    metrics
                )

        # Aggregate CACD results
        cacd_aggregated = {}
        for method_name, run_metrics_list in \
                cacd_run_metrics.items():
            agg = calculator.aggregate(run_metrics_list)
            cacd_aggregated[method_name] = {
                'auc_mean': agg.auc_mean,
                'auc_std': agg.auc_std,
                'eer_mean': agg.eer_mean,
                'eer_std': agg.eer_std,
                'tar_at_far1_mean': agg.tar_at_far1_mean,
                'tar_at_far1_std': agg.tar_at_far1_std,
                'total_updates_mean': (
                    agg.total_updates_mean
                ),
                'total_updates_std': agg.total_updates_std
            }

        all_results['cacd'] = cacd_aggregated

    else:
        logger.warning(
            f"CACD root not found at {cacd_root}. "
            f"Skipping CACD evaluation."
        )

    # --------------------------------------------------
    # FG-NET Evaluation
    # --------------------------------------------------
    logger.info("\n--- FG-NET Evaluation ---")

    fgnet_root = Path(config['paths']['fgnet_root'])
    if fgnet_root.exists():
        fgnet = FGNETDataset(
            root=fgnet_root,
            enrollment_size=config['dataset'][
                'fgnet'
            ]['enrollment_size']
        )
        logger.info(
            f"FG-NET: {fgnet.n_identities} identities"
        )

        fgnet_run_metrics = {
            name: [] for name in [
                'static', 'ols', 'replay',
                'buffer', 'cocov'
            ]
        }

        for run_idx in range(n_runs):
            run_seed = seed_base + run_idx
            logger.info(
                f"FG-NET Run {run_idx + 1}/{n_runs} "
                f"(seed={run_seed})"
            )

            partition = fgnet.build_partition(
                seed=run_seed
            )

            cache = EmbeddingCache(
                cache_dir=config['paths']['embeddings_dir'],
                encoder=encoder,
                dataset_name='fgnet'
            )
            cache.extract_and_cache(
                partition, batch_size=64
            )
            loaded = cache.load_partition_embeddings(
                partition
            )
            impostors = cache.load_impostor_embeddings(
                partition, loaded
            )

            methods = build_methods(
                config, tau_ver, tau_delta
            )
            run_results = run_on_dataset(
                'FG-NET', partition, loaded,
                impostors, methods, calculator,
                seed=run_seed
            )

            for method_name, metrics in \
                    run_results.items():
                fgnet_run_metrics[method_name].append(
                    metrics
                )

        # Aggregate FG-NET results
        fgnet_aggregated = {}
        for method_name, run_metrics_list in \
                fgnet_run_metrics.items():
            agg = calculator.aggregate(run_metrics_list)
            fgnet_aggregated[method_name] = {
                'auc_mean': agg.auc_mean,
                'auc_std': agg.auc_std,
                'eer_mean': agg.eer_mean,
                'eer_std': agg.eer_std,
                'tar_at_far1_mean': agg.tar_at_far1_mean,
                'tar_at_far1_std': agg.tar_at_far1_std,
                'total_updates_mean': (
                    agg.total_updates_mean
                ),
                'total_updates_std': agg.total_updates_std
            }

        all_results['fgnet'] = fgnet_aggregated

    else:
        logger.warning(
            f"FG-NET root not found at {fgnet_root}. "
            f"Skipping FG-NET evaluation."
        )

    # --------------------------------------------------
    # Save results and LaTeX tables
    # --------------------------------------------------
    cross_path = results_dir / 'cross_dataset_results.json'
    with open(cross_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(
        f"Cross-dataset results saved to {cross_path}"
    )

    # Generate LaTeX tables per dataset
    method_display = {
        'static': 'Static Enrollment',
        'ols': 'Naive OLS Expansion',
        'replay': 'Replay Dual Memory',
        'buffer': 'Fixed Buffer Averaging',
        'cocov': 'COCOV'
    }

    for dataset_name, dataset_results in \
            all_results.items():
        latex_path = (
            results_dir /
            f'{dataset_name}_results_table.tex'
        )
        with open(latex_path, 'w') as f:
            f.write(
                f"% Auto-generated {dataset_name.upper()} "
                f"results table\n"
                "\\begin{table}[ht]\n"
                "\\centering\n"
                f"\\caption{{Verification performance on "
                f"{dataset_name.upper()} under continuous "
                f"verification.}}\n"
                f"\\label{{tab:{dataset_name}-results}}\n"
                "\\begin{tabular}{lcccc}\n"
                "\\hline\n"
                "\\textbf{Method} & "
                "\\textbf{AUC} & "
                "\\textbf{EER} & "
                "\\textbf{TAR@FAR=1\\%} & "
                "\\textbf{Updates} \\\\\n"
                "\\hline\n"
            )
            for method_name, metrics in \
                    dataset_results.items():
                display = method_display.get(
                    method_name, method_name
                )
                row = (
                    f"{display} & "
                    f"{metrics['auc_mean']:.4f} "
                    f"$\\pm$ {metrics['auc_std']:.4f} & "
                    f"{metrics['eer_mean']:.4f} "
                    f"$\\pm$ {metrics['eer_std']:.4f} & "
                    f"{metrics['tar_at_far1_mean']:.4f} "
                    f"$\\pm$ "
                    f"{metrics['tar_at_far1_std']:.4f} & "
                    f"{metrics['total_updates_mean']:.0f} "
                    f"$\\pm$ "
                    f"{metrics['total_updates_std']:.0f}"
                    f" \\\\"
                )
                f.write(row + '\n')
            f.write(
                "\\hline\n"
                "\\end{tabular}\n"
                "\\end{table}\n"
            )
        logger.info(
            f"LaTeX table saved to {latex_path}"
        )

    logger.info("=== Cross-dataset evaluation complete ===")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Run cross-dataset evaluation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='/opt/code/cocov/config/config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    run_cross_dataset(args.config)
