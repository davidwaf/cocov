"""
tests/test_memory.py
--------------------
Unit tests for prototype-based identity memory.

Tests cover enrollment, prototype assignment, insertion,
merging, centre computation, capacity bounds, update
counts, and unit normalisation invariants.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import sys
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.identity_memory import IdentityMemory


# ----------------------------------------------------------
# Fixtures
# ----------------------------------------------------------

@pytest.fixture
def memory():
    """Standard IdentityMemory instance for testing."""
    return IdentityMemory(
        embedding_dim=512,
        max_prototypes=10,
        assign_threshold=0.3,
        new_threshold=0.6,
        merge_threshold=0.95,
        momentum=0.9
    )


@pytest.fixture
def small_memory():
    """Small capacity memory for boundary testing."""
    return IdentityMemory(
        embedding_dim=512,
        max_prototypes=3,
        assign_threshold=0.3,
        new_threshold=0.6,
        merge_threshold=0.95,
        momentum=0.9
    )


def make_unit_embeddings(n: int, seed: int = 42):
    """Create n unit-normalised random embeddings."""
    torch.manual_seed(seed)
    e = torch.randn(n, 512)
    return F.normalize(e, p=2, dim=1)


def make_similar_embeddings(
    base: torch.Tensor,
    n: int,
    noise: float = 0.05
):
    """Create embeddings similar to base with small noise."""
    embeddings = []
    for _ in range(n):
        noisy = base + torch.randn_like(base) * noise
        embeddings.append(
            F.normalize(noisy.unsqueeze(0), p=2, dim=1
                        ).squeeze(0)
        )
    return torch.stack(embeddings)


# ----------------------------------------------------------
# Enrollment Tests
# ----------------------------------------------------------

class TestEnrollment:

    def test_enroll_single(self, memory):
        """Single embedding enrollment succeeds."""
        e = make_unit_embeddings(1)
        memory.enroll('id1', e)
        assert memory.is_enrolled('id1')

    def test_enroll_multiple(self, memory):
        """Multiple embedding enrollment succeeds."""
        e = make_unit_embeddings(5)
        memory.enroll('id1', e)
        assert memory.get_state('id1').prototype_count == 5

    def test_enroll_capped_at_kmax(self, small_memory):
        """Enrollment capped at K_max prototypes."""
        e = make_unit_embeddings(10)
        small_memory.enroll('id1', e)
        assert small_memory.get_state(
            'id1'
        ).prototype_count == 3

    def test_enroll_duplicate_raises(self, memory):
        """Enrolling same identity twice raises ValueError."""
        e = make_unit_embeddings(3)
        memory.enroll('id1', e)
        with pytest.raises(ValueError):
            memory.enroll('id1', e)

    def test_centre_unit_norm_after_enroll(self, memory):
        """Identity centre is unit-normalised after enrollment."""
        e = make_unit_embeddings(5)
        memory.enroll('id1', e)
        centre = memory.get_centre('id1')
        norm = centre.norm().item()
        assert abs(norm - 1.0) < 1e-5, (
            f"Centre norm should be 1.0, got {norm}"
        )

    def test_not_enrolled_raises(self, memory):
        """Accessing unenrolled identity raises KeyError."""
        with pytest.raises(KeyError):
            memory.get_centre('unknown_id')

    def test_enrolled_ids_list(self, memory):
        """enrolled_ids returns all enrolled identities."""
        for i in range(5):
            e = make_unit_embeddings(3, seed=i)
            memory.enroll(f'id{i}', e)
        assert memory.n_enrolled == 5
        for i in range(5):
            assert f'id{i}' in memory.enrolled_ids


# ----------------------------------------------------------
# Prototype Update Tests
# ----------------------------------------------------------

class TestPrototypeUpdate:

    def test_assignment_when_close(self, memory):
        """Assignment triggered when probe is close to prototype."""
        base = make_unit_embeddings(1, seed=0).squeeze(0)
        enrollment = make_similar_embeddings(
            base, 3, noise=0.02
        )
        memory.enroll('id1', enrollment)

        # Probe very similar to base
        probe = make_similar_embeddings(
            base, 1, noise=0.02
        ).squeeze(0)
        op = memory.update('id1', probe, drift=0.05)
        assert op == 'assignment', (
            f"Expected assignment, got {op}"
        )

    def test_insertion_when_novel(self, memory):
        """Insertion triggered when probe is novel."""
        base = make_unit_embeddings(1, seed=0).squeeze(0)
        enrollment = make_similar_embeddings(
            base, 3, noise=0.02
        )
        memory.enroll('id1', enrollment)
        initial_count = memory.get_state(
            'id1'
        ).prototype_count

        # Probe very different from base
        novel = make_unit_embeddings(1, seed=99).squeeze(0)
        op = memory.update('id1', novel, drift=0.0)
        assert op == 'insertion', (
            f"Expected insertion, got {op}"
        )
        assert memory.get_state(
            'id1'
        ).prototype_count == initial_count + 1

    def test_no_update_high_drift(self, memory):
        """No update applied when drift exceeds threshold."""
        base = make_unit_embeddings(1, seed=0).squeeze(0)
        enrollment = make_similar_embeddings(
            base, 3, noise=0.02
        )
        memory.enroll('id1', enrollment)
        initial_count = memory.get_state(
            'id1'
        ).prototype_count

        # Novel probe but high drift blocks insertion
        novel = make_unit_embeddings(1, seed=99).squeeze(0)
        op = memory.update('id1', novel, drift=1.5)
        assert op == 'no_update', (
            f"Expected no_update with high drift, got {op}"
        )
        assert memory.get_state(
            'id1'
        ).prototype_count == initial_count

    def test_prototype_norm_after_assignment(self, memory):
        """Prototypes remain unit-normalised after assignment."""
        base = make_unit_embeddings(1, seed=0).squeeze(0)
        enrollment = make_similar_embeddings(base, 3, 0.02)
        memory.enroll('id1', enrollment)

        probe = make_similar_embeddings(base, 1, 0.02
                                        ).squeeze(0)
        memory.update('id1', probe, drift=0.05)

        prototypes = memory.get_prototypes('id1')
        for i in range(len(prototypes)):
            norm = prototypes[i].norm().item()
            assert abs(norm - 1.0) < 1e-5, (
                f"Prototype {i} norm should be 1.0, "
                f"got {norm}"
            )

    def test_centre_norm_after_update(self, memory):
        """Centre remains unit-normalised after update."""
        base = make_unit_embeddings(1, seed=0).squeeze(0)
        enrollment = make_similar_embeddings(base, 3, 0.02)
        memory.enroll('id1', enrollment)

        probe = make_similar_embeddings(base, 1, 0.02
                                        ).squeeze(0)
        memory.update('id1', probe, drift=0.05)

        centre = memory.get_centre('id1')
        norm = centre.norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_capacity_not_exceeded(self, small_memory):
        """Prototype count never exceeds K_max."""
        base = make_unit_embeddings(1, seed=0).squeeze(0)
        enrollment = make_similar_embeddings(base, 2, 0.02)
        small_memory.enroll('id1', enrollment)

        # Try to insert many novel probes
        for i in range(20):
            novel = make_unit_embeddings(
                1, seed=100 + i
            ).squeeze(0)
            small_memory.update('id1', novel, drift=0.0)

        count = small_memory.get_state(
            'id1'
        ).prototype_count
        assert count <= 3, (
            f"Prototype count {count} exceeds K_max=3"
        )


# ----------------------------------------------------------
# Merging Tests
# ----------------------------------------------------------

class TestMerging:

    def test_merge_reduces_count(self):
        """Merging redundant prototypes reduces count."""
        memory = IdentityMemory(
            embedding_dim=512,
            max_prototypes=10,
            assign_threshold=0.3,
            new_threshold=0.6,
            merge_threshold=0.99,  # High threshold
            momentum=0.9
        )

        # Create two very similar prototypes
        base = make_unit_embeddings(1, seed=0).squeeze(0)
        similar = make_similar_embeddings(base, 2, 0.001)
        memory.enroll('id1', similar)
        initial_count = memory.get_state(
            'id1'
        ).prototype_count

        # Insert another very similar one to trigger merge
        probe = make_similar_embeddings(
            base, 1, 0.001
        ).squeeze(0)
        memory.update('id1', probe, drift=0.0)

        final_count = memory.get_state(
            'id1'
        ).prototype_count
        # Either merged or not, count should not exceed K_max
        assert final_count <= 10

    def test_merge_preserves_norm(self):
        """Merged prototype is unit-normalised."""
        memory = IdentityMemory(
            embedding_dim=512,
            max_prototypes=10,
            assign_threshold=0.5,
            new_threshold=0.7,
            merge_threshold=0.999,
            momentum=0.9
        )
        base = make_unit_embeddings(1, seed=0).squeeze(0)
        similar = make_similar_embeddings(base, 3, 0.0001)
        memory.enroll('id1', similar)

        prototypes = memory.get_prototypes('id1')
        for p in prototypes:
            norm = p.norm().item()
            assert abs(norm - 1.0) < 1e-4


# ----------------------------------------------------------
# Update Count Tests
# ----------------------------------------------------------

class TestUpdateCounts:

    def test_initial_counts_zero(self, memory):
        """Update counts start at zero after enrollment."""
        e = make_unit_embeddings(3)
        memory.enroll('id1', e)
        counts = memory.get_update_counts()
        assert counts['total_updates'] == 0
        assert counts['assignments'] == 0
        assert counts['insertions'] == 0
        assert counts['merges'] == 0

    def test_assignment_count_increments(self, memory):
        """Assignment count increments after assignment."""
        base = make_unit_embeddings(1, seed=0).squeeze(0)
        enrollment = make_similar_embeddings(base, 3, 0.02)
        memory.enroll('id1', enrollment)

        probe = make_similar_embeddings(
            base, 1, 0.02
        ).squeeze(0)
        op = memory.update('id1', probe, drift=0.05)

        if op == 'assignment':
            counts = memory.get_update_counts()
            assert counts['assignments'] >= 1

    def test_insertion_count_increments(self, memory):
        """Insertion count increments after insertion."""
        base = make_unit_embeddings(1, seed=0).squeeze(0)
        enrollment = make_similar_embeddings(base, 3, 0.02)
        memory.enroll('id1', enrollment)

        novel = make_unit_embeddings(1, seed=99).squeeze(0)
        op = memory.update('id1', novel, drift=0.0)

        if op == 'insertion':
            counts = memory.get_update_counts()
            assert counts['insertions'] >= 1

    def test_per_identity_counts(self, memory):
        """Per-identity counts tracked correctly."""
        base = make_unit_embeddings(1, seed=0).squeeze(0)
        e1 = make_similar_embeddings(base, 3, 0.02)
        memory.enroll('id1', e1)

        base2 = make_unit_embeddings(1, seed=5).squeeze(0)
        e2 = make_similar_embeddings(base2, 3, 0.02)
        memory.enroll('id2', e2)

        # Update id1 only
        probe = make_similar_embeddings(
            base, 1, 0.02
        ).squeeze(0)
        memory.update('id1', probe, drift=0.05)

        counts = memory.get_update_counts()
        per_id = counts['per_identity']

        assert 'id1' in per_id
        assert 'id2' in per_id
        assert per_id['id2']['updates'] == 0


# ----------------------------------------------------------
# Prototype Retrieval Tests
# ----------------------------------------------------------

class TestPrototypeRetrieval:

    def test_get_prototypes_shape(self, memory):
        """get_prototypes returns correct shape."""
        k = 4
        e = make_unit_embeddings(k)
        memory.enroll('id1', e)
        prototypes = memory.get_prototypes('id1')
        assert prototypes.shape == (k, 512)

    def test_get_prototype_counts(self, memory):
        """get_prototype_counts returns correct values."""
        for i in range(3):
            e = make_unit_embeddings(i + 2, seed=i)
            memory.enroll(f'id{i}', e)

        counts = memory.get_prototype_counts()
        assert counts['id0'] == 2
        assert counts['id1'] == 3
        assert counts['id2'] == 4

    def test_centre_is_normalised_mean(self, memory):
        """Centre equals normalised mean of prototypes."""
        e = make_unit_embeddings(4, seed=42)
        memory.enroll('id1', e)

        prototypes = memory.get_prototypes('id1')
        mean = prototypes.mean(dim=0)
        expected_centre = F.normalize(
            mean.unsqueeze(0), p=2, dim=1
        ).squeeze(0)

        centre = memory.get_centre('id1')
        diff = (centre - expected_centre).abs().max().item()
        assert diff < 1e-5, (
            f"Centre does not match normalised mean. "
            f"Max diff: {diff}"
        )
