"""Input encoders for HALO.

Convert raw values (scalars, categories) into SDRs before they enter the
cortical pipeline.
"""

from halo.encoders.base import EncoderBase
from halo.encoders.category import CategoryEncoder
from halo.encoders.scalar import ScalarEncoder

__all__ = ["EncoderBase", "ScalarEncoder", "CategoryEncoder"]
