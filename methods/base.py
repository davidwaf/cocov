"""
methods/base.py
---------------
Abstract base class for all verification methods.

Defines the interface that all methods --- static enrollment,
naive OLS expansion, replay-based dual memory, fixed-size
buffer averaging, and COCOV --- must implement.

Enforcing a common interface guarantees that the experiment
runner treats all methods identically, eliminating the
possibility of implementation bias in the comparison.

All methods receive the same stream events in the same order
and are evaluated under the same verification protocol.
The only difference between methods is how they implement
enroll(), verify(), and update().

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MethodResult:
    """
    Result of a single verification event from any method.

    Attributes
    ----------
    identity_id : str
        Claimed identity.
    similarity : float
        Cosine similarity score.
    drift : float
        Cosine distance from identity centre.
    accepted : bool
        Verification decision.
    is_genuine : bool
        Ground truth label.
    escalated : bool
        Whether reviewer escalation was triggered.
    update_performed : str
        Update operation applied after this event.
        One of: 'assignment', 'insertion', 'no_update',
        'buffer_update', 'ols_update', 'consolidation',
        'none' (static baseline).
    sequence_position : int
        Position in the evaluation stream.
    identity_sequence_position : int
        Position within the identity's probe sequence.
    """
    identity_id: str
    similarity: float
    drift: float
    accepted: bool
    is_genuine: bool
    escalated: bool = False
    update_performed: str = 'none'
    sequence_position: int = 0
    identity_sequence_position: int = 0


class BaseVerificationMethod(ABC):
    """
    Abstract base class for all verification methods.

    All methods must implement enroll(), verify_and_update(),
    and get_update_counts(). The experiment runner calls
    these methods identically for all evaluated approaches.

    Parameters
    ----------
    method_name : str
        Human-readable name for this method.
        Used in logging and results reporting.
    verification_threshold : float
        Cosine similarity threshold for acceptance.
    """

    def __init__(
        self,
        method_name: str,
        verification_threshold: float = 0.5
    ):
        self.method_name = method_name
        self.verification_threshold = verification_threshold
        self._results: list[MethodResult] = []

        logger.info(
            f"Method '{method_name}' initialised with "
            f"tau_ver={verification_threshold}"
        )

    @abstractmethod
    def enroll(
        self,
        identity_id: str,
        enrollment_embeddings: np.ndarray
    ) -> None:
        """
        Enrol an identity using enrollment embeddings.

        Called once per identity before the evaluation stream
        begins. Initialises the identity representation that
        will be used for verification and updating.

        Parameters
        ----------
        identity_id : str
            Unique identifier for the identity.
        enrollment_embeddings : np.ndarray
            Enrollment embeddings of shape (k, 512).
            All embeddings are unit-normalised.
        """
        pass

    @abstractmethod
    def verify_and_update(
        self,
        embedding: np.ndarray,
        claimed_identity_id: str,
        is_genuine: bool,
        sequence_position: int,
        identity_sequence_position: int
    ) -> MethodResult:
        """
        Perform verification and apply update if applicable.

        Called for every event in the evaluation stream.
        Computes a verification decision and applies any
        method-specific update to the identity representation.

        The update uses the ground-truth label only where
        the method explicitly uses supervision (COCOV reviewer
        simulation). All other methods update without labels.

        Parameters
        ----------
        embedding : np.ndarray
            Unit-normalised probe embedding of shape (512,).
        claimed_identity_id : str
            Identity claimed by this probe.
        is_genuine : bool
            Ground truth label. Used for metric computation
            and, where applicable, supervised updates.
        sequence_position : int
            Position of this event in the combined stream.
        identity_sequence_position : int
            Position of this probe within its identity's
            probe sequence.

        Returns
        -------
        MethodResult
            Verification result including decision, scores,
            and update operation performed.
        """
        pass

    @abstractmethod
    def get_update_counts(self) -> dict:
        """
        Return update operation counts for this method.

        Returns
        -------
        dict
            Method-specific update count statistics.
            Must include 'total_updates' key at minimum.
        """
        pass

    def reset(self) -> None:
        """
        Reset method state for a new experimental run.

        Clears all enrolled identities and results.
        Called between runs to ensure independence.
        """
        self._results = []
        self._reset_state()
        logger.debug(
            f"Method '{self.method_name}' reset."
        )

    @abstractmethod
    def _reset_state(self) -> None:
        """
        Reset method-specific internal state.
        Called by reset(). Subclasses implement this
        to clear identity representations and counters.
        """
        pass

    def get_results(self) -> list[MethodResult]:
        """
        Return all verification results from this run.

        Returns
        -------
        list of MethodResult
            All results in stream order.
        """
        return self._results

    def _compute_similarity(
        self,
        embedding: np.ndarray,
        centre: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between embedding and centre.

        Both must be unit-normalised. Cosine similarity
        reduces to inner product under unit normalisation.

        Parameters
        ----------
        embedding : np.ndarray
            Unit-normalised probe embedding (512,).
        centre : np.ndarray
            Unit-normalised identity centre (512,).

        Returns
        -------
        float
            Cosine similarity in [-1, 1].
        """
        return float(np.dot(embedding, centre))

    def _compute_drift(
        self,
        embedding: np.ndarray,
        centre: np.ndarray
    ) -> float:
        """
        Compute cosine distance between embedding and centre.

        Parameters
        ----------
        embedding : np.ndarray
            Unit-normalised probe embedding (512,).
        centre : np.ndarray
            Unit-normalised identity centre (512,).

        Returns
        -------
        float
            Cosine distance in [0, 2].
        """
        return 1.0 - self._compute_similarity(
            embedding, centre
        )

    def _normalise(self, vector: np.ndarray) -> np.ndarray:
        """
        L2-normalise a vector.

        Parameters
        ----------
        vector : np.ndarray
            Input vector of shape (d,) or (n, d).

        Returns
        -------
        np.ndarray
            Unit-normalised vector of same shape.
        """
        if vector.ndim == 1:
            norm = np.linalg.norm(vector)
            return vector / norm if norm > 0 else vector
        norms = np.linalg.norm(vector, axis=1, keepdims=True)
        return np.where(norms > 0, vector / norms, vector)

    @property
    def n_results(self) -> int:
        """Number of verification events processed."""
        return len(self._results)
