"""
methods/replay.py
-----------------
Replay-based dual memory baseline for continuous face
verification.

Inspired by dual-memory consolidation architectures
(FearNet, Kemker & Kanan 2017). Incoming embeddings are
held in a short-term buffer of fixed capacity. At fixed
intervals, embeddings from the short-term buffer are
consolidated into a long-term store used for verification.
The short-term buffer is cleared after each consolidation.

Updates occur in discrete bursts aligned with the
consolidation schedule rather than continuously.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import numpy as np
import logging
from collections import deque
from methods.base import BaseVerificationMethod, MethodResult

logger = logging.getLogger(__name__)


class ReplayDualMemory(BaseVerificationMethod):
    """
    Dual-memory baseline with periodic consolidation.

    Parameters
    ----------
    verification_threshold : float
        Cosine similarity threshold for acceptance.
    buffer_size : int
        Maximum capacity of the short-term buffer per identity.
    consolidation_interval : int
        Number of observations between consolidation events.
    """

    def __init__(
        self,
        verification_threshold: float = 0.5,
        buffer_size: int = 10,
        consolidation_interval: int = 5
    ):
        super().__init__(
            method_name="Replay Dual Memory",
            verification_threshold=verification_threshold
        )
        self.buffer_size = buffer_size
        self.consolidation_interval = consolidation_interval

        # Long-term store: identity_id -> centre (512,)
        self._lt_centres: dict[str, np.ndarray] = {}
        # Long-term memory: identity_id -> list of embeddings
        self._lt_memory: dict[str, list[np.ndarray]] = {}
        # Short-term buffer: identity_id -> deque
        self._st_buffer: dict[str, deque] = {}
        # Observation counter per identity
        self._obs_count: dict[str, int] = {}
        self._total_updates: int = 0
        self._consolidations: int = 0

    def enroll(
        self,
        identity_id: str,
        enrollment_embeddings: np.ndarray
    ) -> None:
        """
        Enrol identity. Enrollment seeds the long-term store.

        Parameters
        ----------
        identity_id : str
            Unique identity identifier.
        enrollment_embeddings : np.ndarray
            Enrollment embeddings of shape (k, 512).
        """
        self._lt_memory[identity_id] = list(
            enrollment_embeddings
        )
        mean = enrollment_embeddings.mean(axis=0)
        self._lt_centres[identity_id] = self._normalise(mean)
        self._st_buffer[identity_id] = deque(
            maxlen=self.buffer_size
        )
        self._obs_count[identity_id] = 0

        logger.debug(
            f"Replay enrolled '{identity_id}'."
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
        Verify against long-term centre. Buffer incoming
        embedding and consolidate at fixed intervals.

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
            Verification result with update operation noted.
        """
        if claimed_identity_id not in self._lt_centres:
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

        centre = self._lt_centres[claimed_identity_id]
        similarity = self._compute_similarity(embedding, centre)
        drift = self._compute_drift(embedding, centre)
        accepted = similarity >= self.verification_threshold

        # Add to short-term buffer
        self._st_buffer[claimed_identity_id].append(embedding)
        self._obs_count[claimed_identity_id] += 1

        update_op = 'buffered'

        # Consolidate at fixed intervals
        if (self._obs_count[claimed_identity_id] %
                self.consolidation_interval == 0):
            self._consolidate(claimed_identity_id)
            update_op = 'consolidation'
            self._consolidations += 1

        self._total_updates += 1

        result = MethodResult(
            identity_id=claimed_identity_id,
            similarity=similarity,
            drift=drift,
            accepted=accepted,
            is_genuine=is_genuine,
            escalated=False,
            update_performed=update_op,
            sequence_position=sequence_position,
            identity_sequence_position=identity_sequence_position
        )

        self._results.append(result)
        return result

    def _consolidate(self, identity_id: str) -> None:
        """
        Merge short-term buffer into long-term store.

        Transfers all embeddings from the short-term buffer
        to long-term memory and recomputes the centre.
        Clears the short-term buffer after consolidation.

        Parameters
        ----------
        identity_id : str
            Identity to consolidate.
        """
        st_embeddings = list(
            self._st_buffer[identity_id]
        )
        if not st_embeddings:
            return

        self._lt_memory[identity_id].extend(st_embeddings)
        self._st_buffer[identity_id].clear()

        all_embeddings = np.stack(
            self._lt_memory[identity_id]
        )
        new_mean = all_embeddings.mean(axis=0)
        self._lt_centres[identity_id] = \
            self._normalise(new_mean)

        logger.debug(
            f"Consolidated '{identity_id}': "
            f"{len(self._lt_memory[identity_id])} "
            f"embeddings in long-term store."
        )

    def get_update_counts(self) -> dict:
        """
        Return update counts for replay dual memory.

        Returns
        -------
        dict
            Total observations buffered and consolidations
            performed.
        """
        return {
            'total_updates': self._total_updates,
            'consolidations': self._consolidations,
            'assignments': 0,
            'insertions': 0,
            'merges': 0
        }

    def _reset_state(self) -> None:
        """Clear all memory structures."""
        self._lt_centres = {}
        self._lt_memory = {}
        self._st_buffer = {}
        self._obs_count = {}
        self._total_updates = 0
        self._consolidations = 0
