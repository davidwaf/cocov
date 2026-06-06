"""
scripts/extract_embeddings.py
-----------------------------
Extract and cache face embeddings for one encoder and all three datasets.

Run once per backbone before running experiments. Embeddings are stored under:
    {embeddings_dir}/{encoder_name}/{dataset_name}/{identity_id}/embeddings.npy

Usage
-----
    python scripts/extract_embeddings.py --encoder facenet
    python scripts/extract_embeddings.py --encoder arcface_r100 --device cuda
    python scripts/extract_embeddings.py --encoder adaface --force
    python scripts/extract_embeddings.py --encoder vitb_arcface --datasets vggface2 fgnet

Arguments
---------
--encoder  : one of facenet | arcface_r50 | arcface_r100 | adaface | vitb_arcface
--config   : path to config.yaml (default: config/config.yaml)
--device   : cuda or cpu (overrides config)
--force    : re-extract even if cache exists
--datasets : subset of datasets to extract (default: all three)

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder import get_encoder, list_encoders
from data.dataset import VGGFace2Dataset, FGNETDataset
from data.embeddings import EmbeddingCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract face embeddings for COCOV"
    )
    p.add_argument(
        "--encoder", required=True, choices=list_encoders(),
        help="Encoder backbone to use for extraction"
    )
    p.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config.yaml"
    )
    p.add_argument(
        "--device", default=None,
        help="Override device: cuda or cpu"
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-extract even if cache exists"
    )
    p.add_argument(
        "--datasets", nargs="+",
        default=["vggface2", "fgnet"],
        choices=["vggface2", "fgnet"],
        help="Datasets to extract (default: vggface2 fgnet)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Forward optional weight paths from config to environment variables
    if cfg['encoder'].get('adaface_weights'):
        os.environ['ADAFACE_WEIGHTS'] = cfg['encoder']['adaface_weights']
    if cfg['encoder'].get('vitarcface_weights'):
        os.environ['VITARCFACE_WEIGHTS'] = cfg['encoder']['vitarcface_weights']

    device = args.device or cfg['encoder']['device']
    encoder = get_encoder(args.encoder, device=device)
    logger.info(f"Encoder: {encoder.info}")

    cache_root = Path(cfg['paths']['embeddings_dir']) / args.encoder

    for ds_name in args.datasets:
        logger.info(f"=== {ds_name} ===")

        if ds_name == "vggface2":
            dc = cfg['dataset']['vggface2']
            dataset = VGGFace2Dataset(
                root=cfg['paths']['vggface2_root'],
                min_images=dc['min_images_per_identity']
            )
            ids = dataset.select_identities(
                n_identities=dc['n_identities'], seed=dc['random_seed']
            )
            partition = dataset.build_partition(
                identity_ids=ids,
                enrollment_size=dc['enrollment_size'],
                impostor_ratio=cfg['experiment']['impostor_ratio'],
                seed=dc['random_seed']
            )
        elif ds_name == "fgnet":
            dc = cfg['dataset']['fgnet']
            dataset = FGNETDataset(root=cfg['paths']['fgnet_root'])
            partition = dataset.build_partition(
                seed=dc['random_seed']
            )

        cache = EmbeddingCache(
            cache_dir=cache_root,
            encoder=encoder,
            dataset_name=ds_name
        )
        cache.extract_and_cache(
            partition=partition,
            batch_size=64,
            force=args.force
        )
        stats = cache.cache_stats()
        logger.info(
            f"{ds_name}: {stats['n_cached_identities']} identities, "
            f"{stats['disk_usage_mb']:.1f} MB"
        )

    logger.info("All extractions complete.")


if __name__ == "__main__":
    main()
