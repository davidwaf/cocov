"""
identity_memory.py
------------------
Prototype-based identity memory for continuous face verification.

Each enrolled identity is represented by a bounded set of
unit-normalised prototype vectors in the embedding space.
Prototypes are updated incrementally through assignment,
insertion, and merging operations governed by fixed thresholds,
as defined in Chapter 3 of the accompanying thesis.

The identity centre --- the normalised mean of all current
prototypes --- serves as the reference for verification scoring
and drift estimation. Individual prototypes are retained to
preserve distinct appearance modes within an identity.

Memory is bounded: the number of prototypes per identity never
exceeds K_max. All prototype vectors remain unit-normalised
at all times.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import torch
import torch.nn.functional as F
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IdentityState:
    """
    State container for a single enrolled identity.

    Attributes
    ----------
    identity_id : str
        Unique identifier for this identity.
    prototypes : torch.Tensor
        Matrix of shape (K_max, 512) storing prototype vectors.
        Active rows are those with index < prototype_count.
    prototype_count : int
        Number of currently active prototypes.
    centre : torch.Tensor
        Normalised mean of active prototypes. Shape (512,).
        Recomputed after every prototype modification.
    update_count : int
        Total number of update operations applied to this identity.
    assignment_count : int
        Number of assignment operations applied.
    insertion_count : int
        Number of insertion operations applied.
    merge_count : int
        Number of merge operations applied.
    """
    identity_id: str
    prototypes: torch.Tensor
    prototype_count: int
    centre: torch.Tensor
    update_count: int = 0
    assignment_count: int = 0
    insertion_count: int = 0
    merge_count: int = 0


class IdentityMemory:
    """
    Manages prototype-based identity memory for all enrolled identities.

    Implements the prototype assignment, insertion, merging, and
    centre recomputation operations defined in Chapter 3 of the
    thesis. All operations maintain the unit-normalisation invariant
    and the K_max capacity bound.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the embedding space. Default 512.
    max_prototypes : int
        Maximum number of prototypes per identity (K_max).
    assign_threshold : float
        Maximum cosine distance for prototype assignment (rho_assign).
        Incoming embedding assigned to nearest prototype if
        d_min <= assign_threshold.
    new_threshold : float
        Minimum cosine distance for prototype insertion (rho_new).
        New prototype inserted if d_min > new_threshold and
        drift condition is satisfied.
    merge_threshold : float
        Minimum cosine similarity for prototype merging (rho_merge).
        Prototype pairs with similarity >= merge_threshold are merged.
    momentum : float
        Momentum parameter for prototype assignment update (gamma).
        Controls stability-plasticity trade-off.
        Higher values retain existing prototype structure.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        max_prototypes: int = 10,
        assign_threshold: float = 0.3,
        new_threshold: float = 0.6,
        merge_threshold: float = 0.95,
        momentum: float = 0.9
    ):
        self.embedding_dim = embedding_dim
        self.max_prototypes = max_prototypes
        self.assign_threshold = assign_threshold
        self.new_threshold = new_threshold
        self.merge_threshold = merge_threshold
        self.momentum = momentum

        # Registry of enrolled identities
        # Maps identity_id -> IdentityState
        self._registry: dict[str, IdentityState] = {}

        logger.info(
            f"IdentityMemory initialised: "
            f"K_max={max_prototypes}, "
            f"rho_assign={assign_threshold}, "
            f"rho_new={new_threshold}, "
            f"rho_merge={merge_threshold}, "
            f"gamma={momentum}"
        )

    # ----------------------------------------------------------
    # Enrollment
    # ----------------------------------------------------------

    def enroll(self,
               identity_id: str,
               embeddings: torch.Tensor) -> None:
        """
        Enrol a new identity using one or more enrollment embeddings.

        The initial prototype set is constructed from the provided
        embeddings. If multiple embeddings are provided, they are
        stored as individual prototypes up to K_max, and the
        identity centre is computed from all active prototypes.

        Parameters
        ----------
        identity_id : str
            Unique identifier for the identity to enrol.
        embeddings : torch.Tensor
            Enrollment embeddings of shape (k, 512) or (512,).
            All embeddings must be unit-normalised.

        Raises
        ------
        ValueError
            If identity_id is already enrolled.
        """
        if identity_id in self._registry:
            raise ValueError(
                f"Identity '{identity_id}' is already enrolled. "
                f"Use update() to modify an existing identity."
            )

        # Ensure 2D
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        # Initialise prototype matrix
        prototypes = torch.zeros(
            self.max_prototypes, self.embedding_dim
        )

        # Fill up to K_max prototypes
        n = min(len(embeddings), self.max_prototypes)
        prototypes[:n] = F.normalize(embeddings[:n], p=2, dim=1)

        centre = self._compute_centre(prototypes, n)

        self._registry[identity_id] = IdentityState(
            identity_id=identity_id,
            prototypes=prototypes,
            prototype_count=n,
            centre=centre
        )

        logger.debug(
            f"Enrolled '{identity_id}' with {n} prototype(s)."
        )

    # ----------------------------------------------------------
    # Access
    # ----------------------------------------------------------

    def get_centre(self, identity_id: str) -> torch.Tensor:
        """
        Return the current normalised centre for an identity.

        Parameters
        ----------
        identity_id : str
            Identity to query.

        Returns
        -------
        torch.Tensor
            Normalised centre vector of shape (512,).

        Raises
        ------
        KeyError
            If identity_id is not enrolled.
        """
        self._check_enrolled(identity_id)
        return self._registry[identity_id].centre

    def get_prototypes(self,
                       identity_id: str) -> torch.Tensor:
        """
        Return active prototypes for an identity.

        Parameters
        ----------
        identity_id : str
            Identity to query.

        Returns
        -------
        torch.Tensor
            Active prototype vectors of shape (k, 512),
            where k is the current prototype count.
        """
        self._check_enrolled(identity_id)
        state = self._registry[identity_id]
        return state.prototypes[:state.prototype_count]

    def get_state(self,
                  identity_id: str) -> IdentityState:
        """
        Return the full state object for an identity.

        Parameters
        ----------
        identity_id : str
            Identity to query.

        Returns
        -------
        IdentityState
            Full state including prototypes, centre, and
            update counts.
        """
        self._check_enrolled(identity_id)
        return self._registry[identity_id]

    def is_enrolled(self, identity_id: str) -> bool:
        """Return True if identity_id is currently enrolled."""
        return identity_id in self._registry

    @property
    def enrolled_ids(self) -> list[str]:
        """Return list of all enrolled identity IDs."""
        return list(self._registry.keys())

    @property
    def n_enrolled(self) -> int:
        """Return number of enrolled identities."""
        return len(self._registry)

    # ----------------------------------------------------------
    # Update Operations
    # ----------------------------------------------------------

    def update(self,
               identity_id: str,
               embedding: torch.Tensor,
               drift: float) -> str:
        """
        Update identity memory with a verified embedding.

        Applies assignment, insertion, or no update depending
        on the distance between the embedding and existing
        prototypes, and the drift value relative to thresholds.

        Assignment is preferred when the embedding is close to
        an existing prototype. Insertion occurs when the embedding
        is sufficiently novel and drift is within bounds. If the
        prototype count is at capacity, insertion is redirected
        to assignment regardless of distance.

        Merging is applied after every assignment or insertion
        to consolidate redundant prototypes.

        Parameters
        ----------
        identity_id : str
            Identity whose memory is to be updated.
        embedding : torch.Tensor
            Unit-normalised probe embedding of shape (512,).
        drift : float
            Cosine distance between embedding and identity centre.
            Used to gate insertion.

        Returns
        -------
        str
            Operation performed: 'assignment', 'insertion',
            or 'no_update'.
        """
        self._check_enrolled(identity_id)
        state = self._registry[identity_id]
        embedding = F.normalize(embedding.unsqueeze(0),
                                p=2, dim=1).squeeze(0)

        # Compute distances to all active prototypes
        active = state.prototypes[:state.prototype_count]
        distances = 1.0 - torch.mv(active, embedding)
        d_min, k_star = distances.min(0)
        d_min = d_min.item()
        k_star = k_star.item()

        operation = "no_update"

        if d_min <= self.assign_threshold:
            # Assignment: refine nearest prototype
            self._assign(state, k_star, embedding)
            operation = "assignment"
            state.assignment_count += 1

        elif (d_min > self.new_threshold and
              drift <= self.assign_threshold and
              state.prototype_count < self.max_prototypes):
            # Insertion: add new prototype
            self._insert(state, embedding)
            operation = "insertion"
            state.insertion_count += 1

        elif (d_min > self.new_threshold and
              drift <= self.assign_threshold and
              state.prototype_count >= self.max_prototypes):
            # Capacity reached: redirect to assignment
            self._assign(state, k_star, embedding)
            operation = "assignment"
            state.assignment_count += 1

        if operation != "no_update":
            # Merge redundant prototypes
            merged = self._merge(state)
            state.merge_count += merged
            # Recompute centre
            state.centre = self._compute_centre(
                state.prototypes,
                state.prototype_count
            )
            state.update_count += 1

        return operation

    # ----------------------------------------------------------
    # Private Operations
    # ----------------------------------------------------------

    def _assign(self,
                state: IdentityState,
                k_star: int,
                embedding: torch.Tensor) -> None:
        """
        Refine prototype k_star using exponential moving average.

        Implements Equation (3.6) from the thesis:
            w_new = normalise(gamma * w_old + (1 - gamma) * z)

        Parameters
        ----------
        state : IdentityState
            Identity state to modify.
        k_star : int
            Index of the prototype to refine.
        embedding : torch.Tensor
            Incoming unit-normalised embedding of shape (512,).
        """
        w_old = state.prototypes[k_star]
        w_new = (self.momentum * w_old +
                 (1 - self.momentum) * embedding)
        state.prototypes[k_star] = F.normalize(
            w_new.unsqueeze(0), p=2, dim=1
        ).squeeze(0)

    def _insert(self,
                state: IdentityState,
                embedding: torch.Tensor) -> None:
        """
        Insert a new prototype into the identity state.

        Parameters
        ----------
        state : IdentityState
            Identity state to modify.
        embedding : torch.Tensor
            Unit-normalised embedding to insert as shape (512,).
        """
        idx = state.prototype_count
        state.prototypes[idx] = embedding
        state.prototype_count += 1

    def _merge(self, state: IdentityState) -> int:
        """
        Merge prototype pairs whose cosine similarity exceeds
        rho_merge.

        Pairs are replaced by their normalised mean. The process
        repeats until no redundant pairs remain.

        Parameters
        ----------
        state : IdentityState
            Identity state to consolidate.

        Returns
        -------
        int
            Number of merge operations performed.
        """
        n_merged = 0

        merged = True
        while merged and state.prototype_count > 1:
            merged = False
            active = state.prototypes[:state.prototype_count]
            sim = torch.mm(active, active.t())

            for i in range(state.prototype_count):
                for j in range(i + 1,
                               state.prototype_count):
                    if sim[i, j].item() >= self.merge_threshold:
                        # Replace i with normalised mean
                        mean = (active[i] + active[j]) / 2.0
                        state.prototypes[i] = F.normalize(
                            mean.unsqueeze(0), p=2, dim=1
                        ).squeeze(0)
                        # Remove j by shifting remaining down
                        state.prototypes[j:-1] = \
                            state.prototypes[j + 1:].clone()
                        state.prototypes[-1] = 0.0
                        state.prototype_count -= 1
                        n_merged += 1
                        merged = True
                        break
                if merged:
                    break

        return n_merged

    def _compute_centre(self,
                        prototypes: torch.Tensor,
                        count: int) -> torch.Tensor:
        """
        Compute the normalised mean of active prototypes.

        Implements Equations (3.4) and (3.5) from the thesis.

        Parameters
        ----------
        prototypes : torch.Tensor
            Full prototype matrix of shape (K_max, 512).
        count : int
            Number of active prototypes.

        Returns
        -------
        torch.Tensor
            Normalised centre vector of shape (512,).
        """
        active = prototypes[:count]
        mean = active.mean(dim=0)
        return F.normalize(mean.unsqueeze(0),
                           p=2, dim=1).squeeze(0)

    def _check_enrolled(self, identity_id: str) -> None:
        """
        Raise KeyError if identity_id is not enrolled.

        Parameters
        ----------
        identity_id : str
            Identity to check.
        """
        if identity_id not in self._registry:
            raise KeyError(
                f"Identity '{identity_id}' is not enrolled."
            )

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    def get_update_counts(self) -> dict:
        """
        Return update operation counts across all identities.

        Returns
        -------
        dict
            Dictionary with keys: total_updates, assignments,
            insertions, merges, per_identity.
        """
        total = assignments = insertions = merges = 0
        per_identity = {}

        for id_, state in self._registry.items():
            total += state.update_count
            assignments += state.assignment_count
            insertions += state.insertion_count
            merges += state.merge_count
            per_identity[id_] = {
                "updates": state.update_count,
                "assignments": state.assignment_count,
                "insertions": state.insertion_count,
                "merges": state.merge_count,
                "prototype_count": state.prototype_count
            }

        return {
            "total_updates": total,
            "assignments": assignments,
            "insertions": insertions,
            "merges": merges,
            "per_identity": per_identity
        }

    def get_prototype_counts(self) -> dict:
        """
        Return current prototype count per identity.

        Returns
        -------
        dict
            Maps identity_id to current prototype count.
        """
        return {
            id_: state.prototype_count
            for id_, state in self._registry.items()
        }
