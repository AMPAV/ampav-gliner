"""Input adapters for AMPAV GLiNER named-entity extraction."""

from .entities import (
    extract_named_entities_from_file,
    extract_named_entities_from_transcript,
)


__all__ = [
    "extract_named_entities_from_file",
    "extract_named_entities_from_transcript",
]
