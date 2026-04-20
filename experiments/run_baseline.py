"""Baseline experiment: run HALO pipeline with the default configuration.

Usage
-----
    python experiments/run_baseline.py
"""

import logging
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from halo.config.loader import load_config
from halo.orchestration.pipeline import HALOPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    config_path = Path(__file__).parent.parent / "configs" / "baseline.yaml"
    config = load_config(config_path)

    logger.info("Starting HALO baseline experiment")
    pipeline = HALOPipeline(config)
    pipeline.run()

    scores = pipeline._reliability.all_scores()
    logger.info("Final reliability scores:")
    for uid, score in sorted(scores.items()):
        logger.info("  %s: %.4f", uid, score)

    logger.info("Total steps completed: %d", config.max_steps)


if __name__ == "__main__":
    main()
