"""
methods/ols.py
--------------
Naive memory expansion baseline using Ordinary Least Squares.

All observed embeddings are accumulated without constraint.
At each verification event, the identity representation is
updated by reconstructing the incoming embedding using OLS
regression over the accumulated embedding set.

This baseline represents the limiting case of unconstrained
incremental updating and characterises the consequences of
unregulated memory expansion on verification performance.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import numpy as np
import logging
from methods.base import BaseVerificationMethod, MethodResult

logger = logging.getLogger(__name__)


class NaiveOLSExpansion(BaseVerificationMethod):
    """
    Unbounded memory expansion with OLS-based representation.

    Accumulates all observed embeddings per identity.
    The identity centre is the normalised mean of all
    accumulated embeddings, updated after every observation.

    Parameters
    ----------
    verification_threshold : float
        Cosine similarity threshold for acceptance.
    """

    def __init__(
        self,
        verification_threshold: float = 0.5
    ):
        super().__init__(
            method_name="Naive OLS Expansion",
            verification_threshold=verification_threshold
        )
        # Maps identity_id -> list of accumulated embeddings
        self._memory: dict[str, list[np.ndarray]] = {}
        # Maps identity_id -> current centre
        self._centres: dict[str, np.ndarray] = {}
        self._total_updates: int = 0

    def enroll(
        self,
        identity_id: str,
        enrollment_embeddings: np.ndarray
    ) -> None:
        """
        Enrol identity. Enrollment embeddings seed the memory.

        Parameters
        ----------
        identity_id : str
            Unique identity identifier.
        enrollment_embeddings : np.ndarray
            Enrollment embeddings of shape (k, 512).
        """
        self._memory[identity_id] = list(enrollment_embeddings)
        mean = enrollment_embeddings.mean(axis=0)
        self._centres[identity_id] = self._normalise(mean)

        logger.debug(
            f"OLS enrolled '{identity_id}' with "
            f"{len(enrollment_embeddings)} embeddings."
        )

    def verify_and_update(
        self,
        embedding: np.ndarray,
        claimed_identity_id: str,
        is_genuine: bool,
        sequence_position: int,
        identity_sequence_position: int
    ) -> MethodResult:
        """
        Verify probe and accumulate embedding unconditionally.

        The new embedding is added to memory regardless of
        the verification outcome. The centre is recomputed
        from all accumulated embeddings.

        Parameters
        ----------
        embedding : np.ndarray
            Unit-normalised probe embedding of shape (512,).
        claimed_identity_id : str
            Identity claimed by this probe.
        is_genuine : bool
            Ground truth label.
        sequence_position : int
            Position in the combined evaluation stream.
        identity_sequence_position : int
            Position within the identity probe sequence.

        Returns
        -------
        MethodResult
            Verification result with update applied.
        """
        if claimed_identity_id not in self._centres:
            logger.warning(
                f"Identity '{claimed_identity_id}' not enrolled."
            )
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

        centre = self._centres[claimed_identity_id]
        similarity = self._compute_similarity(embedding, centre)
        drift = self._compute_drift(embedding, centre)
        accepted = similarity >= self.verification_threshold

        # Accumulate and recompute centre
        self._memory[claimed_identity_id].append(embedding)
        all_embeddings = np.stack(
            self._memory[claimed_identity_id]
        )
        new_mean = all_embeddings.mean(axis=0)
        self._centres[claimed_identity_id] = \
            self._normalise(new_mean)
        self._total_updates += 1

        result = MethodResult(
            identity_id=claimed_identity_id,
            similarity=similarity,
            drift=drift,
            accepted=accepted,
            is_genuine=is_genuine,
            escalated=False,
            update_performed='ols_update',
            sequence_position=sequence_position,
            identity_sequence_position=identity_sequence_position
        )

        self._results.append(result)
        return result

    def get_update_counts(self) -> dict:
        """
        Return update counts for OLS expansion.

        Returns
        -------
        dict
            Total updates and memory sizes per identity.
        """
        memory_sizes = {
            k: len(v) for k, v in self._memory.items()
        }
        return {
            'total_updates': self._total_updates,
            'assignments': 0,
            'insertions': 0,
            'merges': 0,
            'memory_sizes': memory_sizes
        }

    def _reset_state(self) -> None:
        """Clear memory and centres."""
        self._memory = {}
        self._centres = {}
        self._total_updates = 0
