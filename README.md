# ampav-gliner

GLiNER tooling for the AMPAV environment.

This package provides a thin synchronous wrapper that extracts named entities
from plain text and returns an AMPAV `ToolOutput` containing `NamedEntities`.

Phase 1 intentionally does not handle AMPAV `Transcript` timestamp alignment or
long-transcript chunking.
