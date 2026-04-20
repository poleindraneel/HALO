"""Centralised logger factory for HALO modules."""

import logging
import sys

__all__ = ["get_logger"]

_HANDLER_ATTACHED: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a standard StreamHandler if none is configured.

    Calling :func:`logging.getLogger` directly is also fine inside modules;
    this helper ensures at least one handler is present so messages are not
    silently discarded.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log = logging.getLogger(name)
    # Only attach a handler once per root-name to avoid duplicate output.
    root_name = name.split(".")[0]
    if root_name not in _HANDLER_ATTACHED and not logging.root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )
        log.addHandler(handler)
        _HANDLER_ATTACHED.add(root_name)
    return log
