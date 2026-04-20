"""CorticalUnit: SDR encoder with Hebbian learning.

Implements a simplified model of a cortical minicolumn ensemble.  Input is
projected through a random synaptic weight matrix; the top-k activations
(winner-take-all) form the SDR.  Weights are updated by a local Hebbian rule —
no gradient computation, no backpropagation.
"""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import CorticalConfig
from halo.core.base import CorticalUnitBase
from halo.core.sdr import SDR

logger = logging.getLogger(__name__)

__all__ = ["CorticalUnit"]


class CorticalUnit(CorticalUnitBase):
    """Cortical unit with winner-take-all encoding and Hebbian weight updates.

    Parameters
    ----------
    unit_id:
        Unique identifier string, e.g. ``"unit_0"``.
    config:
        CorticalConfig supplying n_columns, sparsity, learning_rate.
    rng:
        NumPy random generator for reproducible initialisation.
    input_dim:
        Dimensionality of the raw input vector.
    """

    def __init__(
        self,
        unit_id: str,
        config: CorticalConfig,
        rng: np.random.Generator,
        input_dim: int,
    ) -> None:
        self._unit_id = unit_id
        self._config = config
        self._rng = rng
        self._input_dim = input_dim
        self._step: int = 0
        self._init_weights()

    # ------------------------------------------------------------------
    # CorticalUnitBase interface
    # ------------------------------------------------------------------

    @property
    def unit_id(self) -> str:
        return self._unit_id

    def encode(self, input_data: np.ndarray) -> SDR:
        """Project *input_data* through synaptic weights and apply WTA.

        Parameters
        ----------
        input_data:
            Float array of shape ``(input_dim,)``.

        Returns
        -------
        SDR
            Sparse encoding with ``sparsity * n_columns`` active bits.
        """
        activations = self._synaptic_weights @ input_data  # (n_columns,)
        k = max(1, int(self._config.sparsity * self._config.n_columns))
        bits = self._winner_take_all(activations, k)
        self._step += 1
        return SDR(bits=bits, unit_id=self._unit_id, timestamp=self._step)

    def learn(self, sdr: SDR) -> None:
        """Hebbian weight update: strengthen active, weaken inactive columns.

        Active columns (bits==True) receive a positive delta; inactive
        columns receive a negative delta scaled by ``learning_rate``.

        This is a simplified Oja-style rule without explicit normalisation —
        sufficient for demonstration purposes.
        """
        delta = np.where(
            sdr.bits[:, np.newaxis],           # (n_columns, 1) broadcast
            self._config.learning_rate,
            -self._config.learning_rate * 0.1,  # mild decay for inactive
        )
        self._synaptic_weights += delta
        logger.debug(
            "%s learned from SDR with %d active columns",
            self._unit_id,
            int(sdr.bits.sum()),
        )

    def reset(self) -> None:
        """Reinitialise synaptic weights and step counter."""
        self._step = 0
        self._init_weights()
        logger.debug("%s reset", self._unit_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        self._synaptic_weights: np.ndarray = self._rng.uniform(
            low=0.0,
            high=1.0,
            size=(self._config.n_columns, self._input_dim),
        )

    def _winner_take_all(self, activations: np.ndarray, k: int) -> np.ndarray:
        """Return a boolean mask with exactly *k* True entries at the top-k positions.

        Parameters
        ----------
        activations:
            Float array of shape ``(n_columns,)``.
        k:
            Number of winners.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(n_columns,)``.
        """
        indices = np.argpartition(activations, -k)[-k:]
        mask = np.zeros(len(activations), dtype=bool)
        mask[indices] = True
        return mask
