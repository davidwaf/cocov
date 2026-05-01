"""
embeddings.py
-------------
Embedding extraction and caching for continuous face verification
experiments.

Extracts 512-dimensional l2-normalised embeddings from face images
using the FaceEncoder and stores them on disk. During experiments,
embeddings are loaded from cache rather than recomputed, ensuring:

    - All methods operate on identical numerical inputs
    - Extraction cost does not contribute to runtime measurements
    - Experiments are reproducible across runs

Cache structure on disk:
    embeddings_dir/
        {dataset_name}/
            {identity_id}/
                embeddings.npy   # shape (N, 512) float32
                paths.json       # ordered list of source paths

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm
from models.encoder import FaceEncoder
from data.dataset import DatasetPartition, IdentityPartition

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Manages extraction and disk caching of face embeddings.

    Embeddings are extracted once per dataset and stored as
    numpy arrays. Subsequent experiment runs load from cache,
    making repeated experiments fast and consistent.

    Parameters
    ----------
    cache_dir : str or Path
        Root directory for embedding cache storage.
    encoder : FaceEncoder
        Encoder instance used for extraction.
    dataset_name : str
        Name of the dataset being cached.
    """

    def __init__(self,
                 cache_dir: str | Path,
                 encoder: FaceEncoder,
                 dataset_name: str):
        self.cache_dir = Path(cache_dir) / dataset_name
        self.encoder = encoder
        self.dataset_name = dataset_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"EmbeddingCache initialised: "
            f"{self.cache_dir}"
        )

    def extract_and_cache(
        self,
        partition: DatasetPartition,
        batch_size: int = 64,
        force: bool = False
    ) -> None:
        """
        Extract embeddings for all images in a partition
        and save to disk.

        Skips identities that are already cached unless
        force=True. Includes both enrollment and probe
        images for each identity.

        Parameters
        ----------
        partition : DatasetPartition
            Dataset partition containing image paths.
        batch_size : int
            Images per GPU batch during extraction.
        force : bool
            If True, re-extract even if cache exists.
        """
        identities = partition.identity_partitions
        logger.info(
            f"Extracting embeddings for "
            f"{len(identities)} identities..."
        )

        skipped = 0
        extracted = 0

        for identity_id, ip in tqdm(
            identities.items(),
            desc="Extracting embeddings"
        ):
            identity_cache = self.cache_dir / identity_id

            if identity_cache.exists() and not force:
                skipped += 1
                continue

            identity_cache.mkdir(parents=True, exist_ok=True)

            # All images: enrollment + probe
            all_paths = ip.enrollment_paths + ip.probe_paths

            try:
                embeddings = self.encoder.encode_batch(
                    all_paths, batch_size=batch_size
                )
            except RuntimeError as e:
                logger.error(
                    f"Failed to extract embeddings for "
                    f"{identity_id}: {e}"
                )
                continue

            # Save embeddings as float32
            np.save(
                str(identity_cache / "embeddings.npy"),
                embeddings.numpy().astype(np.float32)
            )

            # Save ordered path list
            with open(
                identity_cache / "paths.json", 'w'
            ) as f:
                json.dump(all_paths, f, indent=2)

            extracted += 1

        logger.info(
            f"Extraction complete: {extracted} extracted, "
            f"{skipped} loaded from cache."
        )

    def load_embeddings(
        self,
        identity_id: str
    ) -> tuple[np.ndarray, list[str]]:
        """
        Load cached embeddings for a single identity.

        Parameters
        ----------
        identity_id : str
            Identity whose embeddings to load.

        Returns
        -------
        tuple of (np.ndarray, list of str)
            embeddings: shape (N, 512) float32
            paths: ordered list of source image paths

        Raises
        ------
        FileNotFoundError
            If embeddings are not cached for this identity.
        """
        identity_cache = self.cache_dir / identity_id

        emb_path = identity_cache / "embeddings.npy"
        paths_path = identity_cache / "paths.json"

        if not emb_path.exists():
            raise FileNotFoundError(
                f"No cached embeddings for identity "
                f"'{identity_id}'. Run extract_and_cache() first."
            )

        embeddings = np.load(str(emb_path))

        with open(paths_path, 'r') as f:
            paths = json.load(f)

        return embeddings, paths

    def load_partition_embeddings(
        self,
        partition: DatasetPartition
    ) -> dict[str, dict]:
        """
        Load cached embeddings for all identities in a partition.

        Returns embeddings split into enrollment and probe sets,
        consistent with the partition definition.

        Parameters
        ----------
        partition : DatasetPartition
            Partition defining enrollment/probe splits.

        Returns
        -------
        dict
            Maps identity_id to:
                {
                    'enrollment': np.ndarray (k, 512),
                    'probes': np.ndarray (n, 512),
                    'enrollment_paths': list of str,
                    'probe_paths': list of str
                }
        """
        result = {}

        for identity_id, ip in \
                partition.identity_partitions.items():
            embeddings, paths = self.load_embeddings(
                identity_id
            )

            k = len(ip.enrollment_paths)

            result[identity_id] = {
                'enrollment': embeddings[:k],
                'probes': embeddings[k:],
                'enrollment_paths': ip.enrollment_paths,
                'probe_paths': ip.probe_paths
            }

        return result

    def load_impostor_embeddings(
        self,
        partition: DatasetPartition,
        loaded_embeddings: dict[str, dict]
    ) -> list[dict]:
        """
        Resolve impostor trial embeddings from cached data.

        For each impostor trial, retrieves the probe embedding
        from the already-loaded identity embeddings.

        Parameters
        ----------
        partition : DatasetPartition
            Partition containing impostor trial definitions.
        loaded_embeddings : dict
            Already-loaded embeddings from
            load_partition_embeddings().

        Returns
        -------
        list of dict
            Each entry contains:
                {
                    'embedding': np.ndarray (512,),
                    'claimed_identity_id': str,
                    'true_identity_id': str,
                    'probe_path': str
                }
        """
        impostor_data = []

        for trial in partition.impostor_trials:
            true_id = trial.true_identity_id
            probe_path = trial.probe_path

            if true_id not in loaded_embeddings:
                continue

            id_data = loaded_embeddings[true_id]
            all_paths = (
                id_data['enrollment_paths'] +
                id_data['probe_paths']
            )
            all_embeddings = np.vstack([
                id_data['enrollment'],
                id_data['probes']
            ])

            if probe_path in all_paths:
                idx = all_paths.index(probe_path)
                embedding = all_embeddings[idx]
            else:
                logger.warning(
                    f"Impostor probe path not found in "
                    f"cache for {true_id}. Skipping."
                )
                continue

            impostor_data.append({
                'embedding': embedding,
                'claimed_identity_id': trial.claimed_identity_id,
                'true_identity_id': true_id,
                'probe_path': probe_path
            })

        return impostor_data

    def is_cached(self, identity_id: str) -> bool:
        """
        Check whether embeddings are cached for an identity.

        Parameters
        ----------
        identity_id : str
            Identity to check.

        Returns
        -------
        bool
            True if cached embeddings exist.
        """
        return (
            self.cache_dir / identity_id / "embeddings.npy"
        ).exists()

    def cache_stats(self) -> dict:
        """
        Return statistics about the current cache state.

        Returns
        -------
        dict
            Number of cached identities and total
            disk usage in MB.
        """
        cached = list(self.cache_dir.iterdir())
        n_cached = sum(1 for p in cached if p.is_dir())

        total_bytes = sum(
            f.stat().st_size
            for f in self.cache_dir.rglob("*")
            if f.is_file()
        )

        return {
            "n_cached_identities": n_cached,
            "cache_dir": str(self.cache_dir),
            "disk_usage_mb": round(
                total_bytes / (1024 ** 2), 2
            )
        }
