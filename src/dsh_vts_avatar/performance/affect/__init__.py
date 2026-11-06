"""Affect data model (Phase 1 of the emotion refactor).

Evidence sources describe a state; the state is one continuous object. Neither
writes to VTS on its own -- the mixer stays the only writer.
"""

from .log import AffectLog, read_log
from .presets import ANCHORS, LEGACY_LABELS, UNKNOWN_ANCHOR, anchor, is_known
from .schema import (
    NEUTRAL_LABELS,
    SOURCES,
    AffectEvidence,
    AffectState,
    derive_valence_arousal,
    normalize_labels,
)

__all__ = [
    "AffectEvidence",
    "AffectLog",
    "AffectState",
    "ANCHORS",
    "LEGACY_LABELS",
    "NEUTRAL_LABELS",
    "SOURCES",
    "UNKNOWN_ANCHOR",
    "anchor",
    "derive_valence_arousal",
    "is_known",
    "normalize_labels",
    "read_log",
]
