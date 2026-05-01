"""
methods/cocov.py
----------------
Continuous Collaborative Verification (COCOV) framework.

Implements drift-aware, selective prototype-based identity
memory updates with conditional reviewer interaction.

Identity representations are updated only when incoming
observations deviate meaningfully from existing identity
structure, as determined by drift and distance thresholds.
Reviewer escalation is triggered when automatic acceptance
cannot be established reliably.

In the experimental setting, reviewer input is simulated
using ground-truth identity labels supplied by the dataset.
This provides an upper bound on collaborative performance
under ideal reviewer conditions. The reviewer web application
(webapp/main.py) implements real human reviewer interaction
for deployment scenarios.

The update mechanism follows the assignment, insertion,
merging, and bounded memory rules defined in Chapter 3
of the accompanying thesis.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import numpy as np
import logging
from methods.base import BaseVerificationMethod, MethodResult
from models.identity_memory import IdentityMemory
from verification.verifier import Verifier
import torch

logger = logging.getLogger(__name__)


class COCOV(BaseVerificationMethod):
    """
    Continuous Collaborative Verification framework.

    Maintains a bounded prototype set per identity and applies
    updates selectively based on drift and distance thresholds.
    Reviewer escalation is triggered when verification score
    falls below tau_ver or drift exceeds tau_delta.

    In the experimental setting, reviewer responses are
    simulated using ground-truth labels. Escalated observations
    confirmed by the reviewer are incorporated under the same
    bounded update rules as automatically accepted observations.

    Parameters
    ----------
    verification_threshold : float
        Minimum cosine similarity for automatic acceptance
        (tau_ver).
    drift_threshold : float
        Maximum cosine distance for update eligibility
        (tau_delta). Escalation triggered if drift exceeds
        this value.
    assign_threshold : float
        Maximum cosine distance for prototype assignment
        (rho_assign).
    new_threshold : float
        Minimum cosine distance for prototype insertion
        (rho_new).
    merge_threshold : float
        Minimum cosine similarity for prototype merging
        (rho_merge).
    momentum : float
        Momentum parameter for prototype assignment (gamma).
    max_prototypes : int
        Maximum prototypes per identity (K_max).
    simulate_reviewer : bool
        If True, reviewer responses are simulated using
        ground-truth labels. If False, escalated observations
        are not incorporated into memory pending real reviewer
        input via the web application.
    """

    def __init__(
        self,
        verification_threshold: float = 0.5,
        drift_threshold: float = 0.35,
        assign_threshold: float = 0.3,
        new_threshold: float = 0.6,
        merge_threshold: float = 0.95,
        momentum: float = 0.9,
        max_prototypes: int = 10,
        simulate_reviewer: bool = True
    ):
        super().__init__(
            method_name="COCOV",
            verification_threshold=verification_threshold
        )
        self.drift_threshold = drift_threshold
        self.simulate_reviewer = simulate_reviewer

        # Prototype memory store
        self._memory = IdentityMemory(
            embedding_dim=512,
            max_prototypes=max_prototypes,
            assign_threshold=assign_threshold,
            new_threshold=new_threshold,
            merge_threshold=merge_threshold,
            momentum=momentum
        )

        # Verifier for similarity and drift computation
        self._verifier = Verifier(
            verification_threshold=verification_threshold,
            drift_threshold=drift_threshold
        )

        # Escalation tracking
        self._n_escalated: int = 0
        self._n_reviewer_confirmed: int = 0
        self._n_reviewer_rejected: int = 0

        logger.info(
            f"COCOV initialised: "
            f"tau_ver={verification_threshold}, "
            f"tau_delta={drift_threshold}, "
            f"simulate_reviewer={simulate_reviewer}"
        )

    def enroll(
        self,
        identity_id: str,
        enrollment_embeddings: np.ndarray
    ) -> None:
        """
        Enrol identity using prototype memory.

        Enrollment embeddings initialise the prototype set.
        Up to K_max enrollment embeddings are stored as
        initial prototypes. The identity centre is computed
        from all active prototypes.

        Parameters
        ----------
        identity_id : str
            Unique identity identifier.
        enrollment_embeddings : np.ndarray
            Enrollment embeddings of shape (k, 512).
        """
        embeddings_tensor = torch.from_numpy(
            enrollment_embeddings.astype(np.float32)
        )
        self._memory.enroll(identity_id, embeddings_tensor)

        logger.debug(
            f"COCOV enrolled '{identity_id}' with "
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
        Verify probe and selectively update prototype memory.

        Processing follows the five-step cycle defined in
        Chapter 3 Section 3.9:
            1. Retrieve identity centre
            2. Compute similarity and drift
            3. Apply decision gating
            4. Handle escalation if triggered
            5. Update prototype memory if accepted

        Update is applied only when:
            - Observation is automatically accepted, OR
            - Observation is escalated and reviewer confirms

        Parameters
        ----------
        embedding : np.ndarray
            Unit-normalised probe embedding of shape (512,).
        claimed_identity_id : str
            Identity claimed by this probe.
        is_genuine : bool
            Ground truth label. Used for metric computation
            and reviewer simulation.
        sequence_position : int
            Position in the combined evaluation stream.
        identity_sequence_position : int
            Position within the identity probe sequence.

        Returns
        -------
        MethodResult
            Verification result including similarity, drift,
            decision, escalation flag, and update operation.
        """
        if not self._memory.is_enrolled(claimed_identity_id):
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

        # Retrieve current identity centre
        centre_tensor = self._memory.get_centre(
            claimed_identity_id
        )
        centre = centre_tensor.numpy()

        # Compute similarity and drift
        similarity = self._compute_similarity(embedding, centre)
        drift = self._compute_drift(embedding, centre)
        accepted = similarity >= self.verification_threshold

        # Determine escalation
        escalated = (
            similarity < self.verification_threshold or
            drift > self.drift_threshold
        )

        update_op = 'no_update'
        effective_identity = claimed_identity_id

        if not escalated:
            # Automatic acceptance: apply update
            update_op = self._apply_update(
                claimed_identity_id, embedding, drift
            )

        elif escalated and self.simulate_reviewer:
            # Reviewer simulation using ground-truth label
            self._n_escalated += 1
            reviewer_confirmed = is_genuine

            if reviewer_confirmed:
                # Reviewer confirms claimed identity
                self._n_reviewer_confirmed += 1
                # Apply update with reviewer_confirmed=True
                # to bypass drift insertion gate
                update_op = self._apply_update(
                    claimed_identity_id, embedding, drift,
                    reviewer_confirmed=True
                )
            else:
                # Reviewer rejects: no memory update
                self._n_reviewer_rejected += 1
                update_op = 'reviewer_rejected'

        elif escalated and not self.simulate_reviewer:
            # Real reviewer mode: escalate without updating
            # Update will be applied when reviewer responds
            # via the web application
            self._n_escalated += 1
            update_op = 'escalated_pending'

        result = MethodResult(
            identity_id=claimed_identity_id,
            similarity=similarity,
            drift=drift,
            accepted=accepted,
            is_genuine=is_genuine,
            escalated=escalated,
            update_performed=update_op,
            sequence_position=sequence_position,
            identity_sequence_position=identity_sequence_position
        )

        self._results.append(result)
        return result

    def _apply_update(
        self,
        identity_id: str,
        embedding: np.ndarray,
        drift: float,
        reviewer_confirmed: bool = False
    ) -> str:
        """
        Apply prototype update for an accepted observation.

        Converts numpy embedding to tensor and delegates to
        IdentityMemory.update(). When reviewer_confirmed is
        True, the drift gate for insertion is bypassed since
        the reviewer has already validated the identity.
        Automatic updates remain subject to the full drift
        constraint.

        Parameters
        ----------
        identity_id : str
            Identity whose memory to update.
        embedding : np.ndarray
            Unit-normalised probe embedding of shape (512,).
        drift : float
            Cosine distance from identity centre.
        reviewer_confirmed : bool
            If True, bypass drift insertion gate.
            The assignment threshold still applies.

        Returns
        -------
        str
            Operation performed: 'assignment', 'insertion',
            or 'no_update'.
        """
        embedding_tensor = torch.from_numpy(
            embedding.astype(np.float32)
        )
        # When reviewer confirms, pass drift=0 to bypass
        # the drift insertion gate in IdentityMemory.update()
        effective_drift = 0.0 if reviewer_confirmed else drift
        return self._memory.update(
            identity_id, embedding_tensor, effective_drift
        )

    def get_update_counts(self) -> dict:
        """
        Return update counts from prototype memory.

        Returns
        -------
        dict
            Total updates, assignments, insertions, merges,
            escalations, and reviewer outcomes.
        """
        memory_counts = self._memory.get_update_counts()
        return {
            'total_updates': memory_counts['total_updates'],
            'assignments': memory_counts['assignments'],
            'insertions': memory_counts['insertions'],
            'merges': memory_counts['merges'],
            'n_escalated': self._n_escalated,
            'n_reviewer_confirmed': self._n_reviewer_confirmed,
            'n_reviewer_rejected': self._n_reviewer_rejected,
            'per_identity': memory_counts['per_identity']
        }

    def get_prototype_counts(self) -> dict:
        """
        Return current prototype count per identity.

        Returns
        -------
        dict
            Maps identity_id to prototype count.
        """
        return self._memory.get_prototype_counts()

    def _reset_state(self) -> None:
        """Reset prototype memory and escalation counters."""
        self._memory = IdentityMemory(
            embedding_dim=512,
            max_prototypes=self._memory.max_prototypes,
            assign_threshold=self._memory.assign_threshold,
            new_threshold=self._memory.new_threshold,
            merge_threshold=self._memory.merge_threshold,
            momentum=self._memory.momentum
        )
        self._verifier = Verifier(
            verification_threshold=self.verification_threshold,
            drift_threshold=self.drift_threshold
        )
        self._n_escalated = 0
        self._n_reviewer_confirmed = 0
        self._n_reviewer_rejected = 0
