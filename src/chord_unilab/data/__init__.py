"""ChordReferenceLoader: direct ManoSharpaData parquet loading (no NPZ conversion).

Fidelity gate 2: output must equal flash_chord's ``Reference`` arrays
field-by-field. See README.md in this package for the field mapping, the
provenance chain, and the declared GAPs.
"""

from chord_unilab.data.loader import load_chord_reference
from chord_unilab.data.reference import (
    ChordReference,
    HandReference,
    ObjectReference,
    ReferenceMetadata,
)

__all__ = [
    "ChordReference",
    "HandReference",
    "ObjectReference",
    "ReferenceMetadata",
    "load_chord_reference",
]
