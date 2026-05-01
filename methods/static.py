"""
methods/static.py
-----------------
Static enrollment baseline for continuous face verification.

Each identity is represented by a single fixed reference
computed from the enrollment embeddings. No updates are
applied at any point during the evaluation stream.

This baseline establishes the performance achievable without
any form of adaptation and serves as the primary reference
point against which update strategies are assessed.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import numpy as np
import logging
from methods.base import BaseVerificationMethod, MethodResult

logger = logging.getLogger(__name__)


class StaticEnrollment(BaseVerificationMethod):
    """
    Non-adaptive verification using a fixed enrollment reference.

    The identity representation is the normalised mean of the
    enrollment embeddings, computed once at enrollment and
    unchanged for the duration of the evaluation.

    Parameters
    ----------
    verification_threshold : float
        Cosine similarity threshold for acceptance (tau_ver).
    """

    def __init__(
        self,
        verification_threshold: float = 0.5
    ):
        super().__init__(
            method_name="Static Enrollment",
            verification_threshold=verification_threshold
        )
        # Maps identity_id -> centre vector (512,)
        self._centres: dict[str, np.ndarray] = {}

    def enroll(
        self,
        identity_id: str,
        enrollment_embeddings: np.ndarray
    ) -> None:
        """
        Enrol identity using normalised mean of enrollment
        embeddings.

        Parameters
        ----------
        identity_id : str
            Unique identity identifier.
        enrollment_embeddings : np.ndarray
            Enrollment embeddings of shape (k, 512).
        """
        mean = enrollment_embeddings.mean(axis=0)
        self._centres[identity_id] = self._normalise(mean)

        logger.debug(
            f"Static enrolled '{identity_id}' from "
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
        Verify probe against fixed enrollment reference.
        No update is applied.

        Parameters
        ----------
        embedding : np.ndarray
            Unit-normalised probe embedding of shape (512,).
        claimed_identity_id : str
            Identity claimed by this probe.
        is_genuine : bool
            Ground truth label for metric computation.
        sequence_position : int
            Position in the combined evaluation stream.
        identity_sequence_position : int
            Position within the identity's probe sequence.

        Returns
        -------
        MethodResult
            Verification result. update_performed is always
            'none' for this baseline.
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

        result = MethodResult(
            identity_id=claimed_identity_id,
            similarity=similarity,
            drift=drift,
            accepted=accepted,
            is_genuine=is_genuine,
            escalated=False,
            update_performed='none',
            sequence_position=sequence_position,
            identity_sequence_position=identity_sequence_position
        )

        self._results.append(result)
        return result

    def get_update_counts(self) -> dict:
        """
        Return update counts. Static baseline applies no updates.

        Returns
        -------
        dict
            All counts are zero by definition.
        """
        return {
            'total_updates': 0,
            'assignments': 0,
            'insertions': 0,
            'merges': 0
        }

    def _reset_state(self) -> None:
        """Clear all enrolled identity centres."""
        self._centres = {}
