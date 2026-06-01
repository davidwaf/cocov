"""
dataset.py
----------
Dataset loading, identity selection, and partition construction
for continuous face verification experiments.

Handles three datasets:
    - VGGFace2: primary evaluation, filename-based ordering
    - CACD: cross-dataset, year-based temporal ordering
    - FG-NET: diagnostic, age-label-based ordering

For each dataset, this module:
    - Discovers available identities and image paths
    - Applies minimum image count filtering
    - Selects identity subsets using fixed random seeds
    - Partitions each identity into enrollment and probe sets
    - Constructs impostor trial pairings

All file I/O and partitioning is performed here.
Embedding extraction is handled separately in embeddings.py.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import os
import re
import json
import random
import logging
import unicodedata
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IdentityPartition:
    """
    Enrollment and probe partition for a single identity.

    Attributes
    ----------
    identity_id : str
        Unique identity identifier.
    enrollment_paths : list of str
        Ordered image paths used for enrollment initialisation.
    probe_paths : list of str
        Ordered image paths used as verification probes.
    """
    identity_id: str
    enrollment_paths: list[str]
    probe_paths: list[str]


@dataclass
class ImpostorTrial:
    """
    A single impostor verification trial.

    Attributes
    ----------
    probe_path : str
        Path to the impostor probe image.
    claimed_identity_id : str
        Identity against which the impostor is verified.
    true_identity_id : str
        True identity of the probe image.
    """
    probe_path: str
    claimed_identity_id: str
    true_identity_id: str


@dataclass
class DatasetPartition:
    """
    Complete partition for a single experimental run.

    Attributes
    ----------
    identity_partitions : dict
        Maps identity_id to IdentityPartition.
    impostor_trials : list of ImpostorTrial
        All impostor trials for this run.
    dataset_name : str
        Name of the source dataset.
    run_seed : int
        Random seed used for this partition.
    n_identities : int
        Number of identities in this partition.
    """
    identity_partitions: dict[str, IdentityPartition]
    impostor_trials: list[ImpostorTrial]
    dataset_name: str
    run_seed: int
    n_identities: int = 0

    def __post_init__(self):
        self.n_identities = len(self.identity_partitions)


class VGGFace2Dataset:
    """
    Dataset loader for VGGFace2-HQ.

    Images are pre-aligned at 512x512. Within-identity ordering
    is constructed using filename ordering within each identity
    directory, as acquisition timestamps are not available.

    Parameters
    ----------
    root : str or Path
        Path to VGGFace2 root directory containing identity
        subdirectories (n000002, n000003, ...).
    min_images : int
        Minimum number of images required for an identity to
        be eligible for selection. Default 20.
    """

    def __init__(self,
                 root: str | Path,
                 min_images: int = 20):
        self.root = Path(root)
        self.min_images = min_images
        self.dataset_name = "vggface2"

        if not self.root.exists():
            raise FileNotFoundError(
                f"VGGFace2 root not found: {self.root}"
            )

        self._identity_paths = self._discover_identities()
        logger.info(
            f"VGGFace2: {len(self._identity_paths)} identities "
            f"with >= {min_images} images."
        )

    def _discover_identities(self) -> dict[str, list[str]]:
        """
        Discover all eligible identities and their ordered
        image paths.

        Returns
        -------
        dict
            Maps identity_id to sorted list of image paths.
        """
        identity_paths = {}

        for identity_dir in sorted(self.root.iterdir()):
            if not identity_dir.is_dir():
                continue

            images = sorted([
                str(p) for p in identity_dir.iterdir()
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            ])

            if len(images) >= self.min_images:
                identity_paths[identity_dir.name] = images

        return identity_paths

    def select_identities(self,
                          n_identities: int,
                          seed: int) -> list[str]:
        """
        Select a random subset of eligible identities.

        Parameters
        ----------
        n_identities : int
            Number of identities to select.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        list of str
            Selected identity IDs.
        """
        available = list(self._identity_paths.keys())
        if n_identities > len(available):
            logger.warning(
                f"Requested {n_identities} identities but only "
                f"{len(available)} are eligible. "
                f"Using all {len(available)}."
            )
            n_identities = len(available)

        rng = random.Random(seed)
        selected = rng.sample(available, n_identities)
        return sorted(selected)

    def build_partition(
        self,
        identity_ids: list[str],
        enrollment_size: int,
        impostor_ratio: float,
        seed: int
    ) -> DatasetPartition:
        """
        Build enrollment/probe partitions and impostor trials.

        Parameters
        ----------
        identity_ids : list of str
            Identity IDs to include in this partition.
        enrollment_size : int
            Number of images per identity used for enrollment.
        impostor_ratio : float
            Number of impostor trials per genuine probe.
        seed : int
            Random seed for impostor sampling.

        Returns
        -------
        DatasetPartition
            Complete partition for one experimental run.
        """
        rng = random.Random(seed)
        partitions = {}

        for identity_id in identity_ids:
            images = self._identity_paths[identity_id]
            enrollment = images[:enrollment_size]
            probes = images[enrollment_size:]

            if not probes:
                logger.warning(
                    f"Identity {identity_id} has no probe "
                    f"images after enrollment. Skipping."
                )
                continue

            partitions[identity_id] = IdentityPartition(
                identity_id=identity_id,
                enrollment_paths=enrollment,
                probe_paths=probes
            )

        # Build impostor trials
        impostor_trials = self._build_impostor_trials(
            partitions, impostor_ratio, rng
        )

        return DatasetPartition(
            identity_partitions=partitions,
            impostor_trials=impostor_trials,
            dataset_name=self.dataset_name,
            run_seed=seed
        )

    def _build_impostor_trials(
        self,
        partitions: dict[str, IdentityPartition],
        impostor_ratio: float,
        rng: random.Random
    ) -> list[ImpostorTrial]:
        """
        Construct impostor trials by pairing probe images
        from one identity against the claimed identity of
        another.

        For each genuine probe, one impostor probe is sampled
        uniformly at random from a non-matching identity.
        Impostor probes are drawn from probe sets only.

        Parameters
        ----------
        partitions : dict
            Identity partitions for all enrolled identities.
        impostor_ratio : float
            Number of impostor trials per genuine probe.
        rng : random.Random
            Seeded random number generator.

        Returns
        -------
        list of ImpostorTrial
            All impostor trials for this partition.
        """
        identity_ids = list(partitions.keys())
        impostor_trials = []

        for identity_id, partition in partitions.items():
            other_ids = [
                i for i in identity_ids
                if i != identity_id
            ]

            n_impostors = max(
                1,
                int(len(partition.probe_paths) * impostor_ratio)
            )

            for _ in range(n_impostors):
                # Sample a non-matching identity
                impostor_id = rng.choice(other_ids)
                impostor_probe = rng.choice(
                    partitions[impostor_id].probe_paths
                )

                impostor_trials.append(ImpostorTrial(
                    probe_path=impostor_probe,
                    claimed_identity_id=identity_id,
                    true_identity_id=impostor_id
                ))

        return impostor_trials

    @property
    def n_eligible(self) -> int:
        """Number of identities meeting the minimum image count."""
        return len(self._identity_paths)

    @property
    def identity_image_counts(self) -> dict[str, int]:
        """Image count per eligible identity."""
        return {
            k: len(v)
            for k, v in self._identity_paths.items()
        }


class FGNETDataset:
    """
    Dataset loader for FG-NET Aging Database.

    Images are ordered strictly by age label extracted from
    filenames. FG-NET filenames encode subject ID and age
    in the format: {subject_id}A{age}{suffix}.jpg

    Example: 001A02.jpg = Subject 001, Age 02

    All 82 identities are included. No subset sampling is
    applied.

    Parameters
    ----------
    root : str or Path
        Path to FG-NET images directory.
    enrollment_size : int
        Number of youngest images used for enrollment.
    """

    def __init__(self,
                 root: str | Path,
                 enrollment_size: int = 5):
        self.root = Path(root)
        self.enrollment_size = enrollment_size
        self.dataset_name = "fgnet"

        if not self.root.exists():
            raise FileNotFoundError(
                f"FG-NET root not found: {self.root}"
            )

        self._identity_paths = self._discover_identities()
        logger.info(
            f"FG-NET: {len(self._identity_paths)} identities "
            f"loaded."
        )

    def _parse_age(self, filename: str) -> int:
        """
        Extract age from FG-NET filename.

        Parameters
        ----------
        filename : str
            Image filename, e.g. '001A02.jpg'.

        Returns
        -------
        int
            Age extracted from filename.
        """
        match = re.search(r'A(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0

    def _discover_identities(self) -> dict[str, list[str]]:
        """
        Discover all identities and their age-ordered image paths.

        Returns
        -------
        dict
            Maps subject_id to age-ordered list of image paths.
        """
        identity_images = {}

        for img_path in self.root.iterdir():
            if img_path.suffix.lower() not in {
                '.jpg', '.jpeg', '.png'
            }:
                continue

            # Extract subject ID (first 3 characters)
            subject_id = img_path.name[:3]
            if subject_id not in identity_images:
                identity_images[subject_id] = []
            identity_images[subject_id].append(img_path)

        # Sort each identity by age
        identity_paths = {}
        for subject_id, paths in identity_images.items():
            sorted_paths = sorted(
                paths,
                key=lambda p: self._parse_age(p.name)
            )
            if len(sorted_paths) > self.enrollment_size:
                identity_paths[subject_id] = [
                    str(p) for p in sorted_paths
                ]

        return identity_paths

    def build_partition(self, seed: int) -> DatasetPartition:
        """
        Build partition for all FG-NET identities.

        Parameters
        ----------
        seed : int
            Random seed for impostor sampling.

        Returns
        -------
        DatasetPartition
            Complete partition for FG-NET evaluation.
        """
        rng = random.Random(seed)
        partitions = {}

        for identity_id, images in self._identity_paths.items():
            enrollment = images[:self.enrollment_size]
            probes = images[self.enrollment_size:]

            partitions[identity_id] = IdentityPartition(
                identity_id=identity_id,
                enrollment_paths=enrollment,
                probe_paths=probes
            )

        impostor_trials = self._build_impostor_trials(
            partitions, rng
        )

        return DatasetPartition(
            identity_partitions=partitions,
            impostor_trials=impostor_trials,
            dataset_name=self.dataset_name,
            run_seed=seed
        )

    def _build_impostor_trials(
        self,
        partitions: dict[str, IdentityPartition],
        rng: random.Random
    ) -> list[ImpostorTrial]:
        """
        Construct impostor trials for FG-NET.

        One impostor trial per genuine probe, sampled from
        non-matching identities.
        """
        identity_ids = list(partitions.keys())
        impostor_trials = []

        for identity_id, partition in partitions.items():
            other_ids = [
                i for i in identity_ids
                if i != identity_id
            ]
            for _ in partition.probe_paths:
                impostor_id = rng.choice(other_ids)
                impostor_probe = rng.choice(
                    partitions[impostor_id].probe_paths
                )
                impostor_trials.append(ImpostorTrial(
                    probe_path=impostor_probe,
                    claimed_identity_id=identity_id,
                    true_identity_id=impostor_id
                ))

        return impostor_trials

    @property
    def n_identities(self) -> int:
        """Number of identities with sufficient images."""
        return len(self._identity_paths)


class CACDDataset:
    """
    Dataset loader for Cross-Age Celebrity Dataset (CACD).

    Images are ordered by year of acquisition. CACD filenames
    encode the acquisition year in the format:
    {year}_{identity_name}_{index}.jpg

    Example: 2004_Adrien_Brody_0001.jpg

    Parameters
    ----------
    root : str or Path
        Path to CACD image directory.
    min_images : int
        Minimum images per identity. Default 20.
    """

    def __init__(self,
                 root: str | Path,
                 min_images: int = 20):
        self.root = Path(root)
        self.min_images = min_images
        self.dataset_name = "cacd"

        if not self.root.exists():
            raise FileNotFoundError(
                f"CACD root not found: {self.root}"
            )

        self._identity_paths = self._discover_identities()
        logger.info(
            f"CACD: {len(self._identity_paths)} identities "
            f"with >= {min_images} images."
        )

    def _parse_year(self, filename: str) -> int:
        """
        Extract acquisition year from CACD filename.

        Parameters
        ----------
        filename : str
            Image filename, e.g. '2004_Adrien_Brody_0001.jpg'.

        Returns
        -------
        int
            Year extracted from filename.
        """
        match = re.match(r'^(\d{4})_', filename)
        if match:
            return int(match.group(1))
        return 0

    def _parse_identity(self, filename: str) -> str:
        """
        Extract identity name from CACD filename.

        Parameters
        ----------
        filename : str
            Image filename.

        Returns
        -------
        str
            Identity name extracted from filename,
            NFC normalised for consistent comparison.
        """
        parts = filename.split('_')
        if len(parts) >= 3:
            # Format: year_firstname_lastname_index.jpg
            name = '_'.join(parts[1:-1])
        else:
            name = filename
        return unicodedata.normalize('NFC', name)

    def _discover_identities(self) -> dict[str, list[str]]:
        """
        Discover identities and year-ordered image paths.

        CACD structure: each identity has its own
        subdirectory containing images named as:
        {age}_{First}_{Last}_{index}.jpg

        Returns
        -------
        dict
            Maps identity_name to year-ordered image paths.
        """
        identity_paths = {}

        # Each subdirectory is one identity
        for identity_dir in sorted(self.root.iterdir()):
            if not identity_dir.is_dir():
                continue

            # NFC normalise to handle special characters
            # consistently across filesystems
            identity_name = unicodedata.normalize(
                'NFC', identity_dir.name
            )
            images = []

            for img_path in identity_dir.iterdir():
                if img_path.suffix.lower() not in {
                    '.jpg', '.jpeg', '.png'
                }:
                    continue
                images.append(img_path)

            if len(images) < self.min_images:
                continue

            # Sort by age (first token in filename)
            sorted_images = sorted(
                images,
                key=lambda p: (
                    self._parse_year(p.name), p.name
                )
            )
            identity_paths[identity_name] = [
                str(p) for p in sorted_images
            ]

        return identity_paths

    def build_partition(
        self,
        enrollment_size: int,
        impostor_ratio: float,
        seed: int
    ) -> DatasetPartition:
        """
        Build partition for all eligible CACD identities.

        Parameters
        ----------
        enrollment_size : int
            Number of earliest images used for enrollment.
        impostor_ratio : float
            Impostor trials per genuine probe.
        seed : int
            Random seed for impostor sampling.

        Returns
        -------
        DatasetPartition
            Complete partition for CACD evaluation.
        """
        rng = random.Random(seed)
        partitions = {}

        for identity_id, images in self._identity_paths.items():
            enrollment = images[:enrollment_size]
            probes = images[enrollment_size:]

            if not probes:
                continue

            partitions[identity_id] = IdentityPartition(
                identity_id=identity_id,
                enrollment_paths=enrollment,
                probe_paths=probes
            )

        impostor_trials = self._build_impostor_trials(
            partitions, impostor_ratio, rng
        )

        return DatasetPartition(
            identity_partitions=partitions,
            impostor_trials=impostor_trials,
            dataset_name=self.dataset_name,
            run_seed=seed
        )

    def _build_impostor_trials(
        self,
        partitions: dict[str, IdentityPartition],
        impostor_ratio: float,
        rng: random.Random
    ) -> list[ImpostorTrial]:
        """Construct impostor trials for CACD."""
        identity_ids = list(partitions.keys())
        impostor_trials = []

        for identity_id, partition in partitions.items():
            other_ids = [
                i for i in identity_ids
                if i != identity_id
            ]
            n_impostors = max(
                1,
                int(len(partition.probe_paths) * impostor_ratio)
            )
            for _ in range(n_impostors):
                impostor_id = rng.choice(other_ids)
                impostor_probe = rng.choice(
                    partitions[impostor_id].probe_paths
                )
                impostor_trials.append(ImpostorTrial(
                    probe_path=impostor_probe,
                    claimed_identity_id=identity_id,
                    true_identity_id=impostor_id
                ))

        return impostor_trials

    @property
    def n_eligible(self) -> int:
        """Number of eligible identities."""
        return len(self._identity_paths)
