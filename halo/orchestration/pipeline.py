"""HALOPipeline: orchestrates the full biologically inspired processing loop.

Step sequence per timestep
--------------------------
1. Each CorticalUnit encodes the raw input → list[SDR]  (Spatial Pooler)
2. Each CorticalUnit runs Temporal Memory on its column SDR → list[SDR] (cell-level)
3. HeterarchicalLayer caches current column SDRs for next-step lateral biases
4. Each CorticalUnit learns from its column SDR (SP permanences + TM AdaptSegments)
5. HeterarchicalLayer Hebbian lateral weight update
6. ThalamicLayer aggregates column SDRs weighted by reliability → broadcast signal
7. TRNGatingLayer filters per-unit column SDRs by population entropy → list[SDR]
8. ConsensusEngine produces final SDR from gated per-unit SDRs
9. Per-unit dopamine = prediction_accuracy * 2 - 1  (ties dopamine to TM quality)
10. ReliabilityModule updates each unit's score with its own dopamine signal
11. Increment step counter; return final SDR
"""

from __future__ import annotations

import logging

import numpy as np

from halo.config.schema import HALOConfig, ScalarEncoderConfig, CategoryEncoderConfig
from halo.consensus.engine import ConsensusEngine
from halo.core.sdr import SDR
from halo.encoders import EncoderBase, ScalarEncoder, CategoryEncoder
from halo.layers.heterarchical import HeterarchicalLayer
from halo.layers.thalamic import ThalamicLayer
from halo.layers.trn import TRNGatingLayer
from halo.models.cortical_unit import CorticalUnit
from halo.reliability.module import ReliabilityModule

logger = logging.getLogger(__name__)

__all__ = ["HALOPipeline"]


class HALOPipeline:
    """Full HALO processing pipeline.

    Parameters
    ----------
    config:
        Top-level :class:`~halo.config.schema.HALOConfig`.
    """

    def __init__(self, config: HALOConfig) -> None:
        self._config = config
        self._step: int = 0
        self._reliability_history: list[dict[str, float]] = []

        rng = np.random.default_rng(config.seed)

        # --- Cortical units ---
        self._unit_ids: list[str] = [f"unit_{i}" for i in range(config.n_units)]
        self._units: list[CorticalUnit] = [
            CorticalUnit(
                unit_id=uid,
                config=config.cortical,
                rng=np.random.default_rng(rng.integers(2**31)),
                input_dim=config.n_input_dim,
            )
            for uid in self._unit_ids
        ]

        # --- Heterarchical layer (all-to-all, learned lateral connections) ---
        self._heterarchical = HeterarchicalLayer(
            n_columns=config.cortical.n_columns,
            config=config.heterarchical,
            rng=np.random.default_rng(rng.integers(2**31)),
        )
        for uid in self._unit_ids:
            self._heterarchical.register_unit(uid)
        for i, from_id in enumerate(self._unit_ids):
            for j, to_id in enumerate(self._unit_ids):
                if i != j:
                    self._heterarchical.add_connection(from_id, to_id)

        # --- Thalamic relay ---
        self._thalamic = ThalamicLayer(config.thalamic)

        # --- TRN gating ---
        self._trn = TRNGatingLayer(config.trn)

        # --- Reliability ---
        self._reliability = ReliabilityModule(self._unit_ids, config.reliability)

        # --- Consensus ---
        self._consensus = ConsensusEngine(config.consensus)

        # --- Optional encoder ---
        self._encoder: EncoderBase | None = None
        if config.encoder is not None:
            if isinstance(config.encoder, ScalarEncoderConfig):
                self._encoder = ScalarEncoder(
                    n=config.encoder.n,
                    w=config.encoder.w,
                    min_val=config.encoder.min_val,
                    max_val=config.encoder.max_val,
                    periodic=config.encoder.periodic,
                )
            else:
                self._encoder = CategoryEncoder(
                    n=config.encoder.n,
                    w=config.encoder.w,
                    categories=config.encoder.categories,
                )

        logger.info(
            "HALOPipeline initialised: %d units, input_dim=%d",
            config.n_units,
            config.n_input_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, input_data: np.ndarray) -> SDR:
        """Execute one processing step.

        Parameters
        ----------
        input_data:
            Float array of shape ``(n_input_dim,)``.

        Returns
        -------
        SDR
            Consensus SDR for this timestep.
        """
        # 1. Compute lateral biases from previous step's SDRs (zero on step 0).
        lateral_biases = self._heterarchical.compute_biases()

        # 2. SP encode: each unit maps the input to a column-level SDR.
        raw_sdrs: list[SDR] = [
            unit.encode(input_data, lateral_bias=lateral_biases.get(unit.unit_id))
            for unit in self._units
        ]

        # 3. TM temporal_step: activate and predict cells from the column SDR.
        #    This must come BEFORE learn() so that _winner_cells / _prev_winner_cells
        #    are populated for _adapt_segments().
        for unit, sdr in zip(self._units, raw_sdrs):
            unit.temporal_step(sdr)

        # 4. Cache current column SDRs in heterarchical layer for next-step biases.
        self._heterarchical.update_sdrs(raw_sdrs)

        # 5. SP + TM learning: each unit updates permanences and adapts segments.
        for unit, sdr in zip(self._units, raw_sdrs):
            unit.learn(sdr)

        # 6. Hebbian lateral weight update.
        self._heterarchical.learn(raw_sdrs)

        # 7. Thalamic relay — aggregate per-unit column SDRs weighted by reliability.
        #    (Thalamic relay: Sherman & Guillery 2006)
        _thalamic_broadcast = self._thalamic.process(
            raw_sdrs, reliability=self._reliability
        )

        # 8. TRN gates per-unit column SDRs based on population entropy.
        #    (TRN-like selective inhibition: Crick 1984; Pinault 2004)
        gated = self._trn.process(raw_sdrs)

        # 9. Consensus over gated per-unit SDRs weighted by reliability scores.
        scores = self._reliability.all_scores()
        if gated:
            final_sdr = self._consensus.aggregate(gated, scores)
        else:
            # Fallback if all SDRs were suppressed by TRN.
            final_sdr = _thalamic_broadcast[0] if _thalamic_broadcast else SDR.empty(
                self._config.cortical.n_columns, "consensus", self._step
            )

        # 10. Per-unit dopamine = prediction_accuracy * 2 - 1 ∈ [-1, 1].
        #     Units that predicted every active column get +1; all-burst units get -1.
        #     Each unit receives its own signal — reliability diverges over time.
        #     (Dopamine-like reinforcement: Schultz et al. 1997)
        for unit in self._units:
            dopamine = unit.prediction_accuracy * 2.0 - 1.0
            self._reliability.update(unit.unit_id, dopamine)
            logger.debug(
                "Step %d unit %s: prediction_accuracy=%.3f dopamine=%.3f",
                self._step,
                unit.unit_id,
                unit.prediction_accuracy,
                dopamine,
            )
        self._reliability_history.append(self._reliability.all_scores())

        self._step += 1

        logger.debug(
            "Step %d: final SDR active=%d",
            self._step,
            int(final_sdr.bits.sum()),
        )
        return final_sdr

    def run(
        self,
        input_stream: list[np.ndarray | float | str] | None = None,
    ) -> list[SDR]:
        """Run the pipeline for ``config.max_steps`` steps.

        Parameters
        ----------
        input_stream:
            Optional list of inputs.  When an encoder is configured, items
            may be ``float`` (scalar encoder) or ``str`` (category encoder)
            and will be encoded automatically.  Raw ``np.ndarray`` items
            bypass the encoder.  If *None* or shorter than ``max_steps``,
            missing steps use random inputs (encoded if encoder is
            configured, Gaussian noise otherwise).

        Returns
        -------
        list[SDR]
            One consensus SDR per step.
        """
        rng = np.random.default_rng(self._config.seed + 1)
        outputs: list[SDR] = []
        for i in range(self._config.max_steps):
            if input_stream is not None and i < len(input_stream):
                raw = input_stream[i]
            else:
                raw = self._random_input(rng)

            inp = self._prepare_input(raw, step=i)
            outputs.append(self.step(inp))
        logger.info("Run complete: %d steps", self._config.max_steps)
        return outputs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_input(self, rng: np.random.Generator) -> np.ndarray | float | str:
        """Generate a random input appropriate for the configured encoder."""
        if self._encoder is None:
            return rng.standard_normal(self._config.n_input_dim)
        enc_cfg = self._config.encoder
        if isinstance(enc_cfg, ScalarEncoderConfig):
            return float(rng.uniform(enc_cfg.min_val, enc_cfg.max_val))
        # CategoryEncoder
        idx = int(rng.integers(len(enc_cfg.categories)))  # type: ignore[union-attr]
        return enc_cfg.categories[idx]  # type: ignore[union-attr]

    def _prepare_input(
        self, raw: np.ndarray | float | str, step: int
    ) -> np.ndarray:
        """Convert *raw* to a bool numpy array suitable for CorticalUnit.encode().

        If an encoder is configured and *raw* is a scalar or category string,
        encode it first.  Raw numpy arrays always bypass the encoder.
        """
        if self._encoder is not None and not isinstance(raw, np.ndarray):
            return self._encoder.encode(raw, unit_id="pipeline_encoder", timestamp=step).bits
        if isinstance(raw, np.ndarray):
            return raw
        # Fallback: should not happen in normal operation.
        return np.array(raw, dtype=float)

    def get_reliability_history(self) -> list[dict[str, float]]:
        """Return the per-step trust score snapshots collected during :meth:`run`."""
        return list(self._reliability_history)
