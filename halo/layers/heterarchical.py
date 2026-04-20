"""Heterarchical (non-hierarchical peer-to-peer) lateral mixing layer.

Units at the same processing level can influence each other via lateral
connections — forming a heterarchy rather than a strict hierarchy.  This
models horizontal cortico-cortical connections.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from halo.core.base import LayerBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["HeterarchicalLayer"]


class HeterarchicalLayer(LayerBase):
    """Lateral SDR mixing across peer cortical units.

    Usage
    -----
    >>> layer = HeterarchicalLayer()
    >>> layer.register_unit("unit_0")
    >>> layer.register_unit("unit_1")
    >>> layer.add_connection("unit_0", "unit_1")
    >>> mixed = layer.process(sdrs)
    """

    def __init__(self) -> None:
        # adjacency: from_id → list of to_ids
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._units: set[str] = set()

    def register_unit(self, unit_id: str) -> None:
        """Register *unit_id* as a node in the heterarchical graph."""
        self._units.add(unit_id)
        logger.debug("Registered unit %s", unit_id)

    def add_connection(self, from_id: str, to_id: str) -> None:
        """Add a directed lateral connection from *from_id* to *to_id*.

        Parameters
        ----------
        from_id:
            Source unit — its SDR will influence *to_id*.
        to_id:
            Target unit — receives lateral signal from *from_id*.
        """
        self._adjacency[from_id].append(to_id)
        logger.debug("Lateral connection %s → %s", from_id, to_id)

    def process(self, inputs: list[SDR]) -> list[SDR]:
        """Mix each SDR with lateral signals from its connected peers.

        For each SDR, the union of all incoming SDRs (from units that have a
        directed connection *to* this unit) is computed and OR-ed with the
        original SDR.

        Parameters
        ----------
        inputs:
            One SDR per registered unit.

        Returns
        -------
        list[SDR]
            Laterally modified SDRs in the same order as *inputs*.
        """
        # Build a lookup: unit_id → SDR
        sdr_map: dict[str, SDR] = {sdr.unit_id: sdr for sdr in inputs}

        # Build reverse adjacency: to_id → [from_ids]
        reverse: dict[str, list[str]] = defaultdict(list)
        for from_id, to_ids in self._adjacency.items():
            for to_id in to_ids:
                reverse[to_id].append(from_id)

        result: list[SDR] = []
        for sdr in inputs:
            incoming = [
                sdr_map[fid]
                for fid in reverse.get(sdr.unit_id, [])
                if fid in sdr_map
            ]
            mixed = sdr
            for lateral in incoming:
                mixed = mixed.union(lateral)
            result.append(mixed)

        return result

    def reset(self) -> None:
        """Clear all connections and registered units."""
        self._adjacency.clear()
        self._units.clear()
