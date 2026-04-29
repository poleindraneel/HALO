"""HALOPipeline: orchestrates the full biologically inspired processing loop.

Step sequence per timestep
--------------------------
1. Each CorticalUnit encodes the raw input → list[SDR]
2. HeterarchicalLayer performs lateral mixing → list[SDR]
3. Each CorticalUnit learns from its (possibly mixed) SDR
4. ThalamicLayer aggregates mixed SDRs → broadcast signal (single SDR)
5. TRNGatingLayer filters per-unit mixed SDRs by population entropy → list[SDR]
6. ConsensusEngine produces final SDR from gated per-unit SDRs
7. Dopamine signal = overlap score between consecutive outputs
8. ReliabilityModule updates all unit scores
9. Increment step counter; return final SDR
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
from halo.utils.metrics import overlap_score

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

        # --- Heterarchical layer (all-to-all) ---
        self._heterarchical = HeterarchicalLayer()
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

        # Previous output for dopamine signal computation
        self._prev_output: SDR | None = None

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
        # 1. Encode
        raw_sdrs: list[SDR] = [unit.encode(input_data) for unit in self._units]

        # 2. Lateral mixing
        mixed_sdrs = self._heterarchical.process(raw_sdrs)

        # 3. Learn from mixed SDR
        for unit, sdr in zip(self._units, mixed_sdrs):
            unit.learn(sdr)

        # 4. Thalamic relay — aggregate mixed SDRs into a single broadcast signal.
        #    (Thalamic relay: Sherman & Guillery 2006)
        _thalamic_broadcast = self._thalamic.process(mixed_sdrs)  # [SDR(unit_id="thalamic")]

        # 5. TRN gates the per-unit mixed SDRs based on population entropy.
        #    (TRN-like selective inhibition: Crick 1984; Pinault 2004)
        #    Gating on per-unit SDRs preserves unit_id for downstream weighting.
        gated = self._trn.process(mixed_sdrs)

        # 6. Consensus over gated per-unit SDRs weighted by reliability scores.
        scores = self._reliability.all_scores()
        if gated:
            final_sdr = self._consensus.aggregate(gated, scores)
        else:
            # Fallback if all SDRs were suppressed by TRN.
            final_sdr = _thalamic_broadcast[0] if _thalamic_broadcast else SDR.empty(
                self._config.cortical.n_columns, "consensus", self._step
            )

        # 7. Dopamine signal: overlap with previous output
        if self._prev_output is not None and self._prev_output.n == final_sdr.n:
            dopamine = overlap_score(self._prev_output, final_sdr) * 2.0 - 1.0
        else:
            dopamine = 0.0

        # 8. Update reliability scores (broadcast same signal to all units)
        for uid in self._unit_ids:
            self._reliability.update(uid, dopamine)
        self._reliability_history.append(self._reliability.all_scores())

        self._prev_output = final_sdr
        self._step += 1

        logger.debug(
            "Step %d: final SDR active=%d, dopamine=%.4f",
            self._step,
            int(final_sdr.bits.sum()),
            dopamine,
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
