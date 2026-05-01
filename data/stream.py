"""
stream.py
---------
Sequential evaluation stream construction for continuous face
verification experiments.

Combines probe embeddings from multiple identities into a single
interleaved evaluation stream, preserving within-identity ordering
while randomising the order across identities.

This reflects continuous operation in which a verification system
processes claims from multiple enrolled identities in an order
that is not structured by identity.

Stream events are of two types:
    - Genuine: probe belongs to the claimed identity
    - Impostor: probe belongs to a different identity

Both types are interleaved throughout the stream. The same stream
is presented to all evaluated methods within a run, ensuring that
performance differences arise from update behaviour rather than
from stream ordering effects.

Author: David Wafula
Project: COCOV - Continuous Collaborative Verification
"""

import random
import numpy as np
import logging
from dataclasses import dataclass
from data.dataset import DatasetPartition

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """
    A single event in the evaluation stream.

    Attributes
    ----------
    embedding : np.ndarray
        Unit-normalised probe embedding of shape (512,).
    claimed_identity_id : str
        Identity claimed by this probe.
    true_identity_id : str
        True identity of this probe.
    is_genuine : bool
        True if probe belongs to claimed identity.
    sequence_position : int
        Position of this event in the combined stream.
    identity_sequence_position : int
        Position of this probe within its identity's
        probe sequence. Used for temporal analysis.
    image_path : str
        Source image path for audit and logging.
    """
    embedding: np.ndarray
    claimed_identity_id: str
    true_identity_id: str
    is_genuine: bool
    sequence_position: int
    identity_sequence_position: int
    image_path: str


class VerificationStream:
    """
    Constructs and manages the sequential evaluation stream.

    Genuine probes are drawn from each identity's ordered probe
    sequence. Impostor probes are interleaved at positions
    determined by a fixed random seed.

    Within-identity ordering is strictly preserved: for any
    identity i, probe at position t always appears before
    probe at position t+1 in the combined stream, regardless
    of interleaving with other identities.

    Parameters
    ----------
    seed : int
        Random seed governing stream interleaving.
        Held fixed across all methods within a run.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._events: list[StreamEvent] = []
        self._built = False

        logger.info(
            f"VerificationStream initialised with seed={seed}"
        )

    def build(
        self,
        partition: DatasetPartition,
        loaded_embeddings: dict[str, dict],
        impostor_embeddings: list[dict]
    ) -> None:
        """
        Build the interleaved evaluation stream.

        Constructs genuine events from each identity's ordered
        probe sequence, then interleaves impostor events at
        random positions while preserving within-identity order.

        Parameters
        ----------
        partition : DatasetPartition
            Partition defining probe sequences.
        loaded_embeddings : dict
            Cached embeddings keyed by identity_id.
            Each value contains 'probes' and 'probe_paths'.
        impostor_embeddings : list of dict
            Resolved impostor trial embeddings from
            EmbeddingCache.load_impostor_embeddings().
        """
        rng = random.Random(self.seed)
        genuine_events = []
        position_counter = [0]

        # Build genuine events preserving within-identity order
        # Use round-robin interleaving across identities
        identity_ids = list(loaded_embeddings.keys())
        identity_queues = {}

        for identity_id in identity_ids:
            data = loaded_embeddings[identity_id]
            probes = data['probes']
            probe_paths = data['probe_paths']

            events = []
            for seq_pos, (emb, path) in enumerate(
                zip(probes, probe_paths)
            ):
                events.append({
                    'embedding': emb,
                    'claimed_identity_id': identity_id,
                    'true_identity_id': identity_id,
                    'is_genuine': True,
                    'identity_sequence_position': seq_pos,
                    'image_path': path
                })
            identity_queues[identity_id] = events

        # Interleave identities while preserving within-id order
        # Randomly shuffle which identity contributes next
        all_genuine = []
        remaining = {
            k: list(v)
            for k, v in identity_queues.items()
            if v
        }

        while remaining:
            available = list(remaining.keys())
            rng.shuffle(available)
            chosen = available[0]
            all_genuine.append(remaining[chosen].pop(0))
            if not remaining[chosen]:
                del remaining[chosen]

        # Build impostor events
        all_impostor = []
        for imp in impostor_embeddings:
            all_impostor.append({
                'embedding': imp['embedding'],
                'claimed_identity_id': imp['claimed_identity_id'],
                'true_identity_id': imp['true_identity_id'],
                'is_genuine': False,
                'identity_sequence_position': -1,
                'image_path': imp['probe_path']
            })

        # Interleave impostors into genuine stream
        # Insert each impostor at a random position
        combined = list(all_genuine)
        for imp_event in all_impostor:
            insert_pos = rng.randint(0, len(combined))
            combined.insert(insert_pos, imp_event)

        # Assign final sequence positions and build StreamEvents
        self._events = []
        for seq_pos, event in enumerate(combined):
            self._events.append(StreamEvent(
                embedding=event['embedding'],
                claimed_identity_id=event[
                    'claimed_identity_id'
                ],
                true_identity_id=event['true_identity_id'],
                is_genuine=event['is_genuine'],
                sequence_position=seq_pos,
                identity_sequence_position=event[
                    'identity_sequence_position'
                ],
                image_path=event['image_path']
            ))

        self._built = True
        n_genuine = sum(1 for e in self._events if e.is_genuine)
        n_impostor = sum(
            1 for e in self._events if not e.is_genuine
        )
        logger.info(
            f"Stream built: {len(self._events)} total events "
            f"({n_genuine} genuine, {n_impostor} impostor)"
        )

    def __iter__(self):
        """Iterate over stream events in sequence order."""
        if not self._built:
            raise RuntimeError(
                "Stream not built. Call build() first."
            )
        return iter(self._events)

    def __len__(self) -> int:
        """Total number of events in the stream."""
        return len(self._events)

    @property
    def n_genuine(self) -> int:
        """Number of genuine events in the stream."""
        return sum(1 for e in self._events if e.is_genuine)

    @property
    def n_impostor(self) -> int:
        """Number of impostor events in the stream."""
        return sum(
            1 for e in self._events if not e.is_genuine
        )

    @property
    def identity_ids(self) -> list[str]:
        """
        Unique identity IDs appearing in the stream
        as claimed identities.
        """
        return list({
            e.claimed_identity_id for e in self._events
        })

    def genuine_events_for(
        self,
        identity_id: str
    ) -> list[StreamEvent]:
        """
        Return genuine events for a specific identity
        in sequence order.

        Parameters
        ----------
        identity_id : str
            Identity to filter by.

        Returns
        -------
        list of StreamEvent
            Genuine events for this identity, ordered by
            identity_sequence_position.
        """
        return sorted(
            [e for e in self._events
             if e.claimed_identity_id == identity_id
             and e.is_genuine],
            key=lambda e: e.identity_sequence_position
        )

    def stream_stats(self) -> dict:
        """
        Return summary statistics for the stream.

        Returns
        -------
        dict
            Stream statistics including event counts,
            identity count, and genuine/impostor ratio.
        """
        if not self._built:
            return {}

        n_total = len(self._events)
        n_genuine = self.n_genuine
        n_impostor = self.n_impostor

        return {
            "total_events": n_total,
            "n_genuine": n_genuine,
            "n_impostor": n_impostor,
            "genuine_ratio": round(
                n_genuine / n_total, 4
            ) if n_total > 0 else 0,
            "n_identities": len(self.identity_ids),
            "seed": self.seed
        }
