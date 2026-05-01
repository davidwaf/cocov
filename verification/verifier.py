"""
verifier.py
-----------
Verification scoring and decision logic for continuous face
verification.

Implements cosine similarity scoring, drift computation, and
the binary verification decision as defined in Chapter 3 of
the accompanying thesis.

Verification is formulated as a claimed-identity problem:
given a probe embedding and a claimed identity, determine
whether the probe corresponds to that identity by comparing
against the stored identity centre.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import torch
import torch.nn.functional as F
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """
    Result of a single verification event.

    Attributes
    ----------
    identity_id : str
        Claimed identity.
    similarity : float
        Cosine similarity between probe and identity centre.
        Range [-1, 1]. Higher is more similar.
    drift : float
        Cosine distance between probe and identity centre.
        Range [0, 2]. Lower indicates better alignment.
    accepted : bool
        Whether the claimed identity was accepted.
    escalated : bool
        Whether the observation was escalated for review.
        True when similarity < tau_ver or drift > tau_delta.
    is_genuine : Optional[bool]
        Ground truth label. True if probe belongs to claimed
        identity. None if not provided.
    """
    identity_id: str
    similarity: float
    drift: float
    accepted: bool
    escalated: bool
    is_genuine: bool = None


class Verifier:
    """
    Cosine similarity based verifier for continuous verification.

    Computes similarity and drift between probe embeddings and
    stored identity centres, applies fixed thresholds to produce
    binary verification decisions, and determines whether
    reviewer escalation is required.

    Parameters
    ----------
    verification_threshold : float
        Minimum cosine similarity for automatic acceptance
        (tau_ver). Probe is accepted if similarity >= threshold.
    drift_threshold : float
        Maximum cosine distance for automatic update eligibility
        (tau_delta). Escalation triggered if drift > threshold.
    """

    def __init__(
        self,
        verification_threshold: float = 0.5,
        drift_threshold: float = 0.35
    ):
        self.verification_threshold = verification_threshold
        self.drift_threshold = drift_threshold

        logger.info(
            f"Verifier initialised: "
            f"tau_ver={verification_threshold}, "
            f"tau_delta={drift_threshold}"
        )

    def similarity(self,
                   probe: torch.Tensor,
                   centre: torch.Tensor) -> float:
        """
        Compute cosine similarity between probe and identity centre.

        Both vectors must be unit-normalised. Under this condition
        cosine similarity reduces to the inner product.

        Parameters
        ----------
        probe : torch.Tensor
            Unit-normalised probe embedding of shape (512,).
        centre : torch.Tensor
            Unit-normalised identity centre of shape (512,).

        Returns
        -------
        float
            Cosine similarity in range [-1, 1].
        """
        return torch.dot(probe, centre).item()

    def drift(self,
              probe: torch.Tensor,
              centre: torch.Tensor) -> float:
        """
        Compute cosine distance between probe and identity centre.

        Defined as 1 - cosine_similarity. Range [0, 2].
        Lower values indicate stronger alignment with the
        current identity representation.

        Parameters
        ----------
        probe : torch.Tensor
            Unit-normalised probe embedding of shape (512,).
        centre : torch.Tensor
            Unit-normalised identity centre of shape (512,).

        Returns
        -------
        float
            Cosine distance in range [0, 2].
        """
        return 1.0 - self.similarity(probe, centre)

    def verify(self,
               probe: torch.Tensor,
               centre: torch.Tensor,
               identity_id: str,
               is_genuine: bool = None) -> VerificationResult:
        """
        Perform a single verification event.

        Computes similarity and drift, applies thresholds to
        determine acceptance and escalation, and returns a
        structured result.

        Escalation is triggered when:
            similarity < tau_ver  OR  drift > tau_delta

        This is consistent with the reviewer escalation condition
        defined in Chapter 3 Equation (3.8) of the thesis.

        Parameters
        ----------
        probe : torch.Tensor
            Unit-normalised probe embedding of shape (512,).
        centre : torch.Tensor
            Unit-normalised identity centre of shape (512,).
        identity_id : str
            Claimed identity identifier.
        is_genuine : bool, optional
            Ground truth label for metric computation.
            None if ground truth is not available.

        Returns
        -------
        VerificationResult
            Structured result containing similarity, drift,
            acceptance decision, and escalation flag.
        """
        sim = self.similarity(probe, centre)
        drft = self.drift(probe, centre)

        accepted = sim >= self.verification_threshold
        escalated = (
            sim < self.verification_threshold or
            drft > self.drift_threshold
        )

        return VerificationResult(
            identity_id=identity_id,
            similarity=sim,
            drift=drft,
            accepted=accepted,
            escalated=escalated,
            is_genuine=is_genuine
        )

    def update_thresholds(self,
                          verification_threshold: float = None,
                          drift_threshold: float = None) -> None:
        """
        Update verification or drift thresholds.

        Used during calibration only. Thresholds are fixed
        for the duration of each evaluation run.

        Parameters
        ----------
        verification_threshold : float, optional
            New verification threshold value.
        drift_threshold : float, optional
            New drift threshold value.
        """
        if verification_threshold is not None:
            self.verification_threshold = verification_threshold
            logger.debug(
                f"tau_ver updated to {verification_threshold}"
            )
        if drift_threshold is not None:
            self.drift_threshold = drift_threshold
            logger.debug(
                f"tau_delta updated to {drift_threshold}"
            )

    @property
    def info(self) -> dict:
        """
        Return verifier configuration as a dictionary.
        Used for logging and reproducibility records.
        """
        return {
            "verification_threshold": self.verification_threshold,
            "drift_threshold": self.drift_threshold
        }
