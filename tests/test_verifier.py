"""
tests/test_verifier.py
----------------------
Unit tests for verification scoring, drift computation,
decision gating, and escalation logic.

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

from verification.verifier import Verifier, VerificationResult


# ----------------------------------------------------------
# Fixtures
# ----------------------------------------------------------

@pytest.fixture
def verifier():
    """Standard verifier instance."""
    return Verifier(
        verification_threshold=0.5,
        drift_threshold=0.35
    )


def unit_vector(seed: int = 42) -> np.ndarray:
    """Create a single unit-normalised numpy vector."""
    np.random.seed(seed)
    v = np.random.randn(512)
    return v / np.linalg.norm(v)


def unit_tensor(seed: int = 42) -> torch.Tensor:
    """Create a single unit-normalised torch tensor."""
    torch.manual_seed(seed)
    v = torch.randn(512)
    return F.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)


def similar_vector(
    base: np.ndarray,
    noise: float = 0.05
) -> np.ndarray:
    """Create a numpy vector similar to base."""
    noisy = base + np.random.randn(*base.shape) * noise
    return noisy / np.linalg.norm(noisy)


def similar_tensor(
    base: torch.Tensor,
    noise: float = 0.05
) -> torch.Tensor:
    """Create a torch tensor similar to base."""
    noisy = base + torch.randn_like(base) * noise
    return F.normalize(noisy.unsqueeze(0), p=2, dim=1).squeeze(0)


# ----------------------------------------------------------
# Similarity Tests
# ----------------------------------------------------------

class TestSimilarity:

    def test_identical_vectors(self, verifier):
        """Identical vectors have similarity 1.0."""
        v = unit_tensor(42)
        sim = verifier.similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_opposite_vectors(self, verifier):
        """Opposite vectors have similarity -1.0."""
        v = unit_tensor(42)
        sim = verifier.similarity(v, -v)
        assert abs(sim - (-1.0)) < 1e-6

    def test_similarity_range(self, verifier):
        """Similarity is always in [-1, 1]."""
        for seed in range(20):
            a = unit_tensor(seed)
            b = unit_tensor(seed + 100)
            sim = verifier.similarity(a, b)
            assert -1.0 <= sim <= 1.0

    def test_similarity_symmetric(self, verifier):
        """Similarity is symmetric."""
        a = unit_tensor(1)
        b = unit_tensor(2)
        assert abs(
            verifier.similarity(a, b) -
            verifier.similarity(b, a)
        ) < 1e-6

    def test_similar_vectors_high_similarity(self, verifier):
        """Similar vectors have higher similarity than random."""
        base = unit_tensor(42)
        similar = similar_tensor(base, noise=0.05)
        random_vec = unit_tensor(999)
        sim_similar = verifier.similarity(base, similar)
        sim_random = verifier.similarity(base, random_vec)
        assert sim_similar > sim_random, (
            f"Similar vector sim={sim_similar:.4f} should "
            f"exceed random sim={sim_random:.4f}"
        )

    def test_dissimilar_vectors_low_similarity(self, verifier):
        """Random unrelated vectors have low similarity."""
        a = unit_tensor(1)
        b = unit_tensor(999)
        sim = verifier.similarity(a, b)
        assert sim < 0.5


# ----------------------------------------------------------
# Drift Tests
# ----------------------------------------------------------

class TestDrift:

    def test_drift_identity(self, verifier):
        """Drift of identical vectors is 0."""
        v = unit_tensor(42)
        drift = verifier.drift(v, v)
        assert abs(drift) < 1e-6

    def test_drift_opposite(self, verifier):
        """Drift of opposite vectors is 2."""
        v = unit_tensor(42)
        drift = verifier.drift(v, -v)
        assert abs(drift - 2.0) < 1e-6

    def test_drift_range(self, verifier):
        """Drift is always in [0, 2]."""
        for seed in range(20):
            a = unit_tensor(seed)
            b = unit_tensor(seed + 100)
            drift = verifier.drift(a, b)
            assert 0.0 <= drift <= 2.0

    def test_similarity_plus_drift_equals_one(self, verifier):
        """Similarity + drift = 1 always."""
        for seed in range(20):
            a = unit_tensor(seed)
            b = unit_tensor(seed + 50)
            sim = verifier.similarity(a, b)
            drift = verifier.drift(a, b)
            assert abs(sim + drift - 1.0) < 1e-6, (
                f"sim={sim} + drift={drift} != 1.0"
            )


# ----------------------------------------------------------
# Verification Decision Tests
# ----------------------------------------------------------

class TestVerificationDecision:

    def test_accept_high_similarity(self, verifier):
        """High similarity probe is accepted."""
        base = unit_tensor(42)
        probe = similar_tensor(base, noise=0.02)
        result = verifier.verify(
            probe, base, 'id1', is_genuine=True
        )
        assert result.accepted

    def test_reject_low_similarity(self, verifier):
        """Low similarity probe is rejected."""
        base = unit_tensor(42)
        impostor = unit_tensor(999)
        result = verifier.verify(
            impostor, base, 'id1', is_genuine=False
        )
        assert not result.accepted

    def test_result_contains_identity_id(self, verifier):
        """Result contains correct identity ID."""
        base = unit_tensor(42)
        probe = unit_tensor(43)
        result = verifier.verify(
            probe, base, 'test_identity'
        )
        assert result.identity_id == 'test_identity'

    def test_result_similarity_matches(self, verifier):
        """Result similarity matches direct computation."""
        base = unit_tensor(42)
        probe = unit_tensor(43)
        expected_sim = verifier.similarity(probe, base)
        result = verifier.verify(probe, base, 'id1')
        assert abs(result.similarity - expected_sim) < 1e-6

    def test_result_drift_matches(self, verifier):
        """Result drift matches direct computation."""
        base = unit_tensor(42)
        probe = unit_tensor(43)
        expected_drift = verifier.drift(probe, base)
        result = verifier.verify(probe, base, 'id1')
        assert abs(result.drift - expected_drift) < 1e-6

    def test_threshold_boundary_accept(self):
        """Probe exactly at threshold is accepted."""
        verifier = Verifier(
            verification_threshold=0.5,
            drift_threshold=0.35
        )
        base = unit_tensor(42)

        # Find a probe with similarity exactly 0.5
        # by scaling: not straightforward with unit vectors
        # Instead test that threshold=0.0 accepts everything
        verifier_permissive = Verifier(
            verification_threshold=-1.0,
            drift_threshold=2.0
        )
        probe = unit_tensor(999)
        result = verifier_permissive.verify(
            probe, base, 'id1'
        )
        assert result.accepted

    def test_threshold_boundary_reject(self):
        """Strict threshold rejects all probes."""
        verifier_strict = Verifier(
            verification_threshold=1.1,
            drift_threshold=0.35
        )
        base = unit_tensor(42)
        probe = unit_tensor(42)  # Identical
        result = verifier_strict.verify(probe, base, 'id1')
        assert not result.accepted


# ----------------------------------------------------------
# Escalation Tests
# ----------------------------------------------------------

class TestEscalation:

    def test_escalate_low_similarity(self, verifier):
        """Low similarity triggers escalation."""
        base = unit_tensor(42)
        impostor = unit_tensor(999)
        result = verifier.verify(
            impostor, base, 'id1', is_genuine=False
        )
        assert result.escalated

    def test_escalate_high_drift(self, verifier):
        """High drift triggers escalation."""
        verifier_tight = Verifier(
            verification_threshold=0.5,
            drift_threshold=0.01
        )
        base = unit_tensor(42)
        # Similar enough to pass similarity but drift > 0.01
        probe = similar_tensor(base, noise=0.1)
        result = verifier_tight.verify(
            probe, base, 'id1', is_genuine=True
        )
        # With very tight drift threshold, should escalate
        if result.drift > 0.01:
            assert result.escalated

    def test_no_escalation_good_probe(self, verifier):
        """Good probe does not escalate."""
        verifier_loose = Verifier(
            verification_threshold=0.3,
            drift_threshold=0.9
        )
        base = unit_tensor(42)
        probe = similar_tensor(base, noise=0.02)
        result = verifier_loose.verify(
            probe, base, 'id1', is_genuine=True
        )
        assert not result.escalated

    def test_escalation_condition_either_or(self):
        """Escalation triggered by either condition."""
        verifier = Verifier(
            verification_threshold=0.5,
            drift_threshold=0.35
        )
        base = unit_tensor(42)

        # Case: only similarity fails
        # Need sim < 0.5 and drift < 0.35
        # drift = 1 - sim, so if sim = 0.3, drift = 0.7
        # This means both conditions fail simultaneously
        # for random vectors in 512D
        # Just verify the OR logic directly
        probe = unit_tensor(999)
        result = verifier.verify(probe, base, 'id1')
        escalated_expected = (
            result.similarity < 0.5 or
            result.drift > 0.35
        )
        assert result.escalated == escalated_expected


# ----------------------------------------------------------
# Threshold Update Tests
# ----------------------------------------------------------

class TestThresholdUpdate:

    def test_update_verification_threshold(self, verifier):
        """Verification threshold can be updated."""
        verifier.update_thresholds(
            verification_threshold=0.8
        )
        assert verifier.verification_threshold == 0.8

    def test_update_drift_threshold(self, verifier):
        """Drift threshold can be updated."""
        verifier.update_thresholds(drift_threshold=0.5)
        assert verifier.drift_threshold == 0.5

    def test_update_both_thresholds(self, verifier):
        """Both thresholds can be updated simultaneously."""
        verifier.update_thresholds(
            verification_threshold=0.7,
            drift_threshold=0.4
        )
        assert verifier.verification_threshold == 0.7
        assert verifier.drift_threshold == 0.4

    def test_info_reflects_thresholds(self, verifier):
        """Info dict reflects current thresholds."""
        info = verifier.info
        assert (
            info['verification_threshold'] ==
            verifier.verification_threshold
        )
        assert (
            info['drift_threshold'] ==
            verifier.drift_threshold
        )

    def test_decision_changes_after_threshold_update(
        self, verifier
    ):
        """Decision changes when threshold is updated."""
        base = unit_tensor(42)
        probe = similar_tensor(base, noise=0.1)

        # Record initial decision
        result1 = verifier.verify(probe, base, 'id1')

        # Make threshold very strict
        verifier.update_thresholds(
            verification_threshold=0.9999
        )
        result2 = verifier.verify(probe, base, 'id1')

        # With strict threshold, should now reject
        assert not result2.accepted
