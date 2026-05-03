"""
tests/test_dataset.py
---------------------
Unit tests for dataset loading, identity selection,
enrollment partitioning, impostor construction, and
temporal stream ordering.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import sys
import pytest
import random
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import (
    VGGFace2Dataset,
    FGNETDataset,
    CACDDataset,
    IdentityPartition,
    DatasetPartition,
    ImpostorTrial
)

# ----------------------------------------------------------
# Fixtures
# ----------------------------------------------------------

VGGFACE2_ROOT = (
    "/opt/data/vggface2/"
    "VGGface2_None_norm_512_true_bygfpgan"
)
FGNET_ROOT = "/opt/data/fgnet/FGNET/images"
CACD_ROOT = (
    "/opt/data/cacd/archive/cacd_split/cacd_split"
)

VGGFACE2_AVAILABLE = Path(VGGFACE2_ROOT).exists()
FGNET_AVAILABLE = Path(FGNET_ROOT).exists()
CACD_AVAILABLE = Path(CACD_ROOT).exists()


@pytest.fixture(scope="module")
def vggface2():
    if not VGGFACE2_AVAILABLE:
        pytest.skip("VGGFace2 dataset not available.")
    return VGGFace2Dataset(
        root=VGGFACE2_ROOT,
        min_images=20
    )


@pytest.fixture(scope="module")
def fgnet():
    if not FGNET_AVAILABLE:
        pytest.skip("FG-NET dataset not available.")
    return FGNETDataset(
        root=FGNET_ROOT,
        enrollment_size=5
    )


@pytest.fixture(scope="module")
def cacd():
    if not CACD_AVAILABLE:
        pytest.skip("CACD dataset not available.")
    return CACDDataset(
        root=CACD_ROOT,
        min_images=20
    )


# ----------------------------------------------------------
# VGGFace2 Tests
# ----------------------------------------------------------

class TestVGGFace2Dataset:

    def test_eligible_identities(self, vggface2):
        """Dataset should have eligible identities."""
        assert vggface2.n_eligible > 0

    def test_minimum_image_count(self, vggface2):
        """All eligible identities meet minimum threshold."""
        for count in vggface2.identity_image_counts.values():
            assert count >= 20

    def test_deterministic_selection(self, vggface2):
        """Same seed produces same identity selection."""
        selected_1 = vggface2.select_identities(
            n_identities=100, seed=42
        )
        selected_2 = vggface2.select_identities(
            n_identities=100, seed=42
        )
        assert selected_1 == selected_2

    def test_different_seeds_differ(self, vggface2):
        """Different seeds produce different selections."""
        selected_1 = vggface2.select_identities(
            n_identities=100, seed=42
        )
        selected_2 = vggface2.select_identities(
            n_identities=100, seed=99
        )
        assert selected_1 != selected_2

    def test_selection_count(self, vggface2):
        """Selection returns requested number of identities."""
        n = 50
        selected = vggface2.select_identities(
            n_identities=n, seed=42
        )
        assert len(selected) == n

    def test_selection_within_eligible(self, vggface2):
        """Selected identities are all eligible."""
        selected = vggface2.select_identities(
            n_identities=50, seed=42
        )
        eligible = set(
            vggface2.identity_image_counts.keys()
        )
        for id_ in selected:
            assert id_ in eligible

    def test_partition_disjoint(self, vggface2):
        """Enrollment and probe sets are disjoint."""
        selected = vggface2.select_identities(10, seed=42)
        partition = vggface2.build_partition(
            identity_ids=selected,
            enrollment_size=5,
            impostor_ratio=1.0,
            seed=42
        )
        for ip in partition.identity_partitions.values():
            enrollment_set = set(ip.enrollment_paths)
            probe_set = set(ip.probe_paths)
            assert len(
                enrollment_set & probe_set
            ) == 0, (
                "Enrollment and probe sets must be disjoint."
            )

    def test_enrollment_size(self, vggface2):
        """Enrollment set has correct size."""
        k = 5
        selected = vggface2.select_identities(10, seed=42)
        partition = vggface2.build_partition(
            identity_ids=selected,
            enrollment_size=k,
            impostor_ratio=1.0,
            seed=42
        )
        for ip in partition.identity_partitions.values():
            assert len(ip.enrollment_paths) == k

    def test_impostor_from_different_identity(self, vggface2):
        """Impostor probes come from non-matching identities."""
        selected = vggface2.select_identities(10, seed=42)
        partition = vggface2.build_partition(
            identity_ids=selected,
            enrollment_size=5,
            impostor_ratio=1.0,
            seed=42
        )
        for trial in partition.impostor_trials:
            assert (
                trial.claimed_identity_id !=
                trial.true_identity_id
            )

    def test_partition_identity_count(self, vggface2):
        """Partition contains requested identities."""
        n = 10
        selected = vggface2.select_identities(n, seed=42)
        partition = vggface2.build_partition(
            identity_ids=selected,
            enrollment_size=5,
            impostor_ratio=1.0,
            seed=42
        )
        assert partition.n_identities == n

    def test_image_paths_exist(self, vggface2):
        """Image paths in partitions point to real files."""
        selected = vggface2.select_identities(3, seed=42)
        partition = vggface2.build_partition(
            identity_ids=selected,
            enrollment_size=5,
            impostor_ratio=1.0,
            seed=42
        )
        for ip in partition.identity_partitions.values():
            for path in ip.enrollment_paths:
                assert Path(path).exists(), (
                    f"Enrollment path not found: {path}"
                )
            for path in ip.probe_paths[:3]:
                assert Path(path).exists(), (
                    f"Probe path not found: {path}"
                )


# ----------------------------------------------------------
# FG-NET Tests
# ----------------------------------------------------------

class TestFGNETDataset:

    def test_identity_count(self, fgnet):
        """FG-NET should have 82 identities."""
        assert fgnet.n_identities == 82

    def test_age_ordering(self, fgnet):
        """Images are ordered by age label."""
        partition = fgnet.build_partition(seed=42)

        def get_age(path):
            match = re.search(
                r'A(\d+)', path, re.IGNORECASE
            )
            return int(match.group(1)) if match else 0

        for ip in partition.identity_partitions.values():
            all_paths = (
                ip.enrollment_paths + ip.probe_paths
            )
            ages = [get_age(p) for p in all_paths]
            assert ages == sorted(ages), (
                f"Age ordering violated for "
                f"{ip.identity_id}: {ages}"
            )

    def test_enrollment_from_youngest(self, fgnet):
        """Enrollment uses youngest images."""
        partition = fgnet.build_partition(seed=42)

        def get_age(path):
            match = re.search(
                r'A(\d+)', path, re.IGNORECASE
            )
            return int(match.group(1)) if match else 0

        for ip in partition.identity_partitions.values():
            if not ip.probe_paths:
                continue
            max_enroll_age = max(
                get_age(p) for p in ip.enrollment_paths
            )
            min_probe_age = min(
                get_age(p) for p in ip.probe_paths
            )
            assert max_enroll_age <= min_probe_age, (
                f"Enrollment should use younger images "
                f"than probes for {ip.identity_id}"
            )

    def test_impostor_from_different_identity(self, fgnet):
        """FG-NET impostor probes are non-matching."""
        partition = fgnet.build_partition(seed=42)
        for trial in partition.impostor_trials:
            assert (
                trial.claimed_identity_id !=
                trial.true_identity_id
            )

    def test_deterministic_partition(self, fgnet):
        """Same seed produces identical partition."""
        p1 = fgnet.build_partition(seed=42)
        p2 = fgnet.build_partition(seed=42)
        ids1 = sorted(p1.identity_partitions.keys())
        ids2 = sorted(p2.identity_partitions.keys())
        assert ids1 == ids2


# ----------------------------------------------------------
# CACD Tests
# ----------------------------------------------------------

class TestCACDDataset:

    def test_eligible_identities(self, cacd):
        """CACD should have 2000 eligible identities."""
        assert cacd.n_eligible == 2000

    def test_year_ordering(self, cacd):
        """Images are ordered by acquisition year."""
        partition = cacd.build_partition(
            enrollment_size=5,
            impostor_ratio=1.0,
            seed=42
        )

        def get_year(path):
            match = re.match(
                r'(\d+)_',
                Path(path).name
            )
            return int(match.group(1)) if match else 0

        # Check a sample of identities
        identities = list(
            partition.identity_partitions.values()
        )[:10]

        for ip in identities:
            all_paths = (
                ip.enrollment_paths + ip.probe_paths
            )
            years = [get_year(p) for p in all_paths]
            assert years == sorted(years), (
                f"Year ordering violated for "
                f"{ip.identity_id}: {years[:5]}"
            )

    def test_enrollment_size(self, cacd):
        """CACD enrollment has correct size."""
        k = 5
        partition = cacd.build_partition(
            enrollment_size=k,
            impostor_ratio=1.0,
            seed=42
        )
        for ip in partition.identity_partitions.values():
            assert len(ip.enrollment_paths) == k

    def test_impostor_non_matching(self, cacd):
        """CACD impostor probes are non-matching."""
        partition = cacd.build_partition(
            enrollment_size=5,
            impostor_ratio=1.0,
            seed=42
        )
        for trial in partition.impostor_trials[:100]:
            assert (
                trial.claimed_identity_id !=
                trial.true_identity_id
            )
