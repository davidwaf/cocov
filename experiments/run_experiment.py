"""
experiments/run_experiment.py
-----------------------------
Main experiment runner for continuous face verification
evaluation.

Executes all five evaluated methods under both static and
continuous verification protocols across multiple independent
identity batches. Aggregates results and saves tables, figures,
and per-run logs for subsequent analysis.

Execution order per run:
    1. Select non-overlapping identity batch
    2. Build enrollment/probe partitions
    3. Extract or load cached embeddings
    4. Construct interleaved evaluation stream
    5. Enrol all identities for each method
    6. Process stream events sequentially
    7. Compute and record metrics
    8. Save run results

After all runs:
    9. Aggregate metrics across runs
    10. Save aggregated results and LaTeX tables

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm
import yaml

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import VGGFace2Dataset
from data.embeddings import EmbeddingCache
from data.stream import VerificationStream
from models.encoder import FaceEncoder
from methods.static import StaticEnrollment
from methods.ols import NaiveOLSExpansion
from methods.replay import ReplayDualMemory
from methods.buffer import FixedBufferAveraging
from methods.cocov import COCOV
from verification.metrics import MetricsCalculator
from calibration.calibrate import ThresholdCalibrator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/data/logs/experiment.log')
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_methods(config: dict, tau_ver: float, tau_delta: float) -> dict:
    proto_cfg = config['prototype']
    baseline_cfg = config['baselines']

    return {
        'static': StaticEnrollment(
            verification_threshold=tau_ver
        ),
        'ols': NaiveOLSExpansion(
            verification_threshold=tau_ver
        ),
        'replay': ReplayDualMemory(
            verification_threshold=tau_ver,
            buffer_size=baseline_cfg['replay']['buffer_size'],
            consolidation_interval=baseline_cfg['replay'][
                'consolidation_interval'
            ]
        ),
        'buffer': FixedBufferAveraging(
            verification_threshold=tau_ver,
            buffer_size=baseline_cfg['buffer']['buffer_size']
        ),
        'cocov': COCOV(
            verification_threshold=tau_ver,
            drift_threshold=tau_delta,
            assign_threshold=proto_cfg['assign_threshold'],
            new_threshold=proto_cfg['new_threshold'],
            merge_threshold=proto_cfg['merge_threshold'],
            momentum=proto_cfg['momentum'],
            max_prototypes=proto_cfg['max_prototypes'],
            simulate_reviewer=True
        )
    }


def run_single_experiment(
        method,
        stream: VerificationStream,
        loaded_embeddings: dict
) -> dict:
    method.reset()

    for identity_id, data in loaded_embeddings.items():
        method.enroll(identity_id, data['enrollment'])

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
            claimed_identity_id=event.claimed_identity_id,
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

    return {
        'similarities': similarities,
        'labels': labels,
        'drifts': drifts,
        'update_counts': method.get_update_counts(),
        'n_escalated': n_escalated
    }


def run_all_experiments(config_path: str) -> None:
    config = load_config(config_path)

    logger.info("=== COCOV Experimental Evaluation ===")
    logger.info(f"Config: {config_path}")

    encoder = FaceEncoder(
        device=config['encoder']['device']
    )

    ds = VGGFace2Dataset(
        root=config['paths']['vggface2_root'],
        min_images=config['dataset']['vggface2'][
            'min_images_per_identity'
        ]
    )

    logger.info(f"Dataset: {ds.n_eligible} eligible identities")

    eval_ids = ds.select_identities(
        n_identities=config['dataset']['vggface2'][
            'n_identities'
        ],
        seed=config['dataset']['vggface2']['random_seed']
    )

    logger.info(
        f"Selected fixed evaluation pool: {len(eval_ids)} identities"
    )

    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    cal_results_path = results_dir / 'calibration_results.json'

    if cal_results_path.exists():
        logger.info("Loading existing calibration results...")
        with open(cal_results_path, 'r') as f:
            cal_results = json.load(f)
    else:
        logger.info("Running calibration...")
        calibrator = ThresholdCalibrator(
            cache_dir=config['paths']['embeddings_dir'],
            results_dir=config['paths']['results_dir']
        )

        # Merge operating point into sweep config
        sweep_cfg = config['calibration']['sweep'].copy()
        sweep_cfg['operating_point'] = config[
            'calibration'
        ].get('calibration_operating_point', 'tar_far1')

        cal_results = calibrator.calibrate(
            dataset=ds,
            encoder=encoder,
            n_calibration_identities=config['calibration'][
                'n_calibration_identities'
            ],
            calibration_seed=config['calibration'][
                'calibration_seed'
            ],
            evaluation_identity_ids=eval_ids,
            enrollment_size=config['dataset']['vggface2'][
                'enrollment_size'
            ],
            sweep_config=sweep_cfg
        )

    tau_ver = cal_results['tau_ver_optimal']
    tau_delta = cal_results['tau_delta_optimal']

    logger.info(
        f"Using tau_ver={tau_ver:.4f}, "
        f"tau_delta={tau_delta:.4f}"
    )

    calculator = MetricsCalculator()

    all_run_metrics = {
        method_name: []
        for method_name in [
            'static', 'ols', 'replay', 'buffer', 'cocov'
        ]
    }

    n_runs = config['experiment']['n_runs']
    seed_base = config['experiment']['random_seed_base']

    batch_size = config['dataset']['vggface2'].get(
        'batch_size',
        len(eval_ids)
    )

    required_identities = n_runs * batch_size

    if required_identities > len(eval_ids):
        raise ValueError(
            "Not enough selected identities for requested batching. "
            f"Requested {required_identities} identities "
            f"({n_runs} runs x {batch_size} batch size), "
            f"but only {len(eval_ids)} identities were selected."
        )

    logger.info(
        f"Batching enabled: {n_runs} runs x {batch_size} identities "
        f"= {required_identities} unique identities"
    )

    used_identity_ids = set()

    for run_idx in range(n_runs):
        run_seed = seed_base + run_idx

        start_idx = run_idx * batch_size
        end_idx = start_idx + batch_size
        batch_eval_ids = eval_ids[start_idx:end_idx]

        overlap = used_identity_ids.intersection(batch_eval_ids)
        if overlap:
            raise RuntimeError(
                f"Identity overlap detected in run {run_idx + 1}: "
                f"{len(overlap)} overlapping identities."
            )

        used_identity_ids.update(batch_eval_ids)

        logger.info(
            f"\n--- Run {run_idx + 1}/{n_runs} "
            f"(seed={run_seed}) ---"
        )

        logger.info(
            f"Identity batch: {start_idx}:{end_idx} "
            f"({len(batch_eval_ids)} unique identities)"
        )

        partition = ds.build_partition(
            identity_ids=batch_eval_ids,
            enrollment_size=config['dataset']['vggface2'][
                'enrollment_size'
            ],
            impostor_ratio=config['experiment'][
                'impostor_ratio'
            ],
            seed=run_seed
        )

        cache = EmbeddingCache(
            cache_dir=config['paths']['embeddings_dir'],
            encoder=encoder,
            dataset_name='vggface2'
        )

        cache.extract_and_cache(
            partition,
            batch_size=64
        )

        loaded = cache.load_partition_embeddings(partition)

        impostors = cache.load_impostor_embeddings(
            partition,
            loaded
        )

        stream = VerificationStream(seed=run_seed)
        stream.build(partition, loaded, impostors)

        logger.info(
            f"Stream: {stream.stream_stats()}"
        )

        methods = build_methods(config, tau_ver, tau_delta)

        run_results = {}

        for method_name, method in methods.items():
            logger.info(
                f"Running {method.method_name}..."
            )

            raw = run_single_experiment(
                method,
                stream,
                loaded
            )

            run_results[method_name] = raw

            metrics = calculator.compute_run_metrics(
                similarities=raw['similarities'],
                labels=raw['labels'],
                drift_values=raw['drifts'],
                update_counts=raw['update_counts'],
                n_escalated=raw['n_escalated']
            )

            all_run_metrics[method_name].append(metrics)

            logger.info(
                f"  AUC={metrics.auc:.4f} "
                f"EER={metrics.eer:.4f} "
                f"TAR@1%={metrics.tar_at_far1:.4f} "
                f"Updates={metrics.total_updates}"
            )

        run_output = results_dir / f'run_{run_idx:02d}'
        run_output.mkdir(parents=True, exist_ok=True)

        for method_name, raw in run_results.items():
            run_file = run_output / f'{method_name}.json'

            with open(run_file, 'w') as f:
                json.dump({
                    'method': method_name,
                    'run': run_idx,
                    'seed': run_seed,
                    'identity_batch_start': start_idx,
                    'identity_batch_end': end_idx,
                    'n_identities': len(batch_eval_ids),
                    'update_counts': raw['update_counts'],
                    'n_escalated': raw['n_escalated'],
                    'n_events': len(raw['similarities'])
                }, f, indent=2)

    logger.info(
        f"Completed {n_runs} batches using "
        f"{len(used_identity_ids)} unique identities."
    )

    logger.info("\n=== Aggregating results ===")

    aggregated = {}
    latex_rows = []

    method_display_names = {
        'static': 'Static Enrollment',
        'ols': 'Naive OLS Expansion',
        'replay': 'Replay Dual Memory',
        'buffer': 'Fixed Buffer Averaging',
        'cocov': 'COCOV'
    }

    for method_name, run_metrics_list in all_run_metrics.items():
        agg = calculator.aggregate(run_metrics_list)

        aggregated[method_name] = {
            'auc_mean': agg.auc_mean,
            'auc_std': agg.auc_std,
            'eer_mean': agg.eer_mean,
            'eer_std': agg.eer_std,
            'tar_at_far1_mean': agg.tar_at_far1_mean,
            'tar_at_far1_std': agg.tar_at_far1_std,
            'total_updates_mean': agg.total_updates_mean,
            'total_updates_std': agg.total_updates_std,
            'drift_mean': agg.drift_mean,
            'drift_std': agg.drift_std,
            'escalation_rate_mean': agg.escalation_rate_mean,
            'n_runs': agg.n_runs,
            'identity_batch_size': batch_size,
            'unique_identities_total': len(used_identity_ids)
        }

        display_name = method_display_names[method_name]
        latex_row = calculator.format_table_row(
            display_name,
            agg
        )
        latex_rows.append(latex_row)

        logger.info(
            f"{display_name}: "
            f"AUC={agg.auc_mean:.4f}±{agg.auc_std:.4f} "
            f"EER={agg.eer_mean:.4f}±{agg.eer_std:.4f} "
            f"TAR@1%={agg.tar_at_far1_mean:.4f}"
            f"±{agg.tar_at_far1_std:.4f}"
        )

    agg_path = results_dir / 'aggregated_results.json'

    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    logger.info(f"Aggregated results saved to {agg_path}")

    latex_path = results_dir / 'results_table.tex'

    with open(latex_path, 'w') as f:
        f.write(
            "% Auto-generated results table\n"
            "% Chapter 6, Table: Verification performance\n"
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\caption{Verification performance across "
            "baselines under continuous verification.}\n"
            "\\label{tab:comparative-results}\n"
            "\\begin{tabular}{lcccc}\n"
            "\\hline\n"
            "\\textbf{Method} & "
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

    logger.info(f"LaTeX table saved to {latex_path}")
    logger.info("=== Evaluation complete ===")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run COCOV experimental evaluation'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='/opt/code/cocov/config/config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()
    run_all_experiments(args.config)