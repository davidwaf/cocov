"""
methods/buffer.py
-----------------
Fixed-size buffer averaging baseline for continuous face
verification.

Maintains a bounded buffer of the most recent m embeddings
per identity. When a new embedding arrives it replaces the
oldest entry if the buffer is at capacity. The identity
representation is the normalised mean of current buffer
contents, updated at every observation.

This baseline combines bounded memory with continuous uniform
averaging, providing a compact stable representation that
adapts incrementally without drift detection or selective
updating.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import numpy as np
import logging
from collections import deque
from methods.base import BaseVerificationMethod, MethodResult

logger = logging.getLogger(__name__)


class FixedBufferAveraging(BaseVerificationMethod):
    """
    Fixed-size buffer averaging baseline.

    Parameters
    ----------
    verification_threshold : float
        Cosine similarity threshold for acceptance.
    buffer_size : int
        Maximum number of embeddings retained per identity.
    """

    def __init__(
        self,
        verification_threshold: float = 0.5,
        buffer_size: int = 10
    ):
        super().__init__(
            method_name="Fixed Buffer Averaging",
            verification_threshold=verification_threshold
        )
        self.buffer_size = buffer_size
        # Maps identity_id -> deque of embeddings
        self._buffers: dict[str, deque] = {}
        # Maps identity_id -> current centre
        self._centres: dict[str, np.ndarray] = {}
        self._total_updates: int = 0

    def enroll(
        self,
        identity_id: str,
        enrollment_embeddings: np.ndarray
    ) -> None:
        """
        Enrol identity. Enrollment embeddings seed the buffer.

        Parameters
        ----------
        identity_id : str
            Unique identity identifier.
        enrollment_embeddings : np.ndarray
            Enrollment embeddings of shape (k, 512).
        """
        buffer = deque(maxlen=self.buffer_size)
        for emb in enrollment_embeddings:
            buffer.append(emb)

        self._buffers[identity_id] = buffer
        self._centres[identity_id] = self._compute_centre(
            buffer
        )

        logger.debug(
            f"Buffer enrolled '{identity_id}' with "
            f"{len(buffer)} embeddings."
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
        Verify probe and update buffer unconditionally.

        New embedding is added to the buffer, displacing the
        oldest if at capacity. Centre is recomputed from
        current buffer contents.

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
            Verification result with buffer update applied.
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

        # Update buffer and recompute centre
        self._buffers[claimed_identity_id].append(embedding)
        self._centres[claimed_identity_id] = \
            self._compute_centre(
                self._buffers[claimed_identity_id]
            )
        self._total_updates += 1

        result = MethodResult(
            identity_id=claimed_identity_id,
            similarity=similarity,
            drift=drift,
            accepted=accepted,
            is_genuine=is_genuine,
            escalated=False,
            update_performed='buffer_update',
            sequence_position=sequence_position,
            identity_sequence_position=identity_sequence_position
        )

        self._results.append(result)
        return result

    def _compute_centre(
        self,
        buffer: deque
    ) -> np.ndarray:
        """
        Compute normalised mean of buffer contents.

        Parameters
        ----------
        buffer : deque
            Current buffer of embeddings.

        Returns
        -------
        np.ndarray
            Normalised mean of shape (512,).
        """
        stacked = np.stack(list(buffer))
        mean = stacked.mean(axis=0)
        return self._normalise(mean)

    def get_update_counts(self) -> dict:
        """
        Return update counts for buffer averaging.

        Returns
        -------
        dict
            Total buffer updates performed.
        """
        return {
            'total_updates': self._total_updates,
            'assignments': 0,
            'insertions': 0,
            'merges': 0
        }

    def _reset_state(self) -> None:
        """Clear buffers and centres."""
        self._buffers = {}
        self._centres = {}
        self._total_updates = 0
