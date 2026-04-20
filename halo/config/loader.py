"""YAML → HALOConfig loader."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from halo.config.schema import (
    CorticalConfig,
    ConsensusConfig,
    HALOConfig,
    ReliabilityConfig,
    ThalamicConfig,
    TRNConfig,
)

logger = logging.getLogger(__name__)

__all__ = ["load_config"]


def load_config(path: str | Path) -> HALOConfig:
    """Parse *path* (YAML) and return a fully validated :class:`HALOConfig`.

    Parameters
    ----------
    path:
        Absolute or relative path to a YAML configuration file.

    Returns
    -------
    HALOConfig
        Nested, validated configuration object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If a required top-level key is missing.
    ValueError
        If any config value fails :class:`~halo.config.schema` validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.debug("Loading HALO config from %s", path)
    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    cortical = CorticalConfig(**raw["cortical"])
    thalamic = ThalamicConfig(**raw["thalamic"])
    trn = TRNConfig(**raw["trn"])
    reliability = ReliabilityConfig(**raw["reliability"])
    consensus = ConsensusConfig(**raw["consensus"])

    cfg = HALOConfig(
        n_units=int(raw["n_units"]),
        n_input_dim=int(raw["n_input_dim"]),
        cortical=cortical,
        thalamic=thalamic,
        trn=trn,
        reliability=reliability,
        consensus=consensus,
        max_steps=int(raw["max_steps"]),
        seed=int(raw["seed"]),
    )
    logger.info("Config loaded: %d units, %d steps", cfg.n_units, cfg.max_steps)
    return cfg
