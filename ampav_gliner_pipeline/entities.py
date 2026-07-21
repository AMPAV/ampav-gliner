"""File and transcript adapters for the GLiNER text tool."""

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

from ampav.core.schema import NamedEntities, ToolOutput, Transcript
from ampav.core.text_chunking import words_to_text_units
from ampav.gliner import GlinerNamedEntityExtractor


def extract_named_entities_from_transcript(
    transcript: Transcript,
    labels: Sequence[str],
    *,
    extractor: GlinerNamedEntityExtractor | None = None,
    threshold: float | None = None,
    flat_ner: bool = True,
    multi_label: bool = False,
    batch_size: int | None = None,
    language: str | None = None,
    separator: str = " ",
    chunk_overlap_tokens: int = 0,
) -> ToolOutput:
    """Extract entities from canonical transcript words and align timestamps.

    The canonical text and chunk units are built together so offsets used for
    extraction and timestamp alignment share one coordinate system.

    Args:
        transcript: Source transcript; canonical text comes from its words.
        labels: Entity labels requested from GLiNER.
        extractor: Optional configured or preloaded extractor.
        threshold: Optional confidence threshold; ``None`` uses GLiNER's default.
        flat_ner: If true, prevent nested entity spans.
        multi_label: If true, allow more than one label for a span.
        batch_size: Optional GLiNER inference batch size.
        language: Optional language override for the output and entities.
        separator: Exact text inserted between rendered transcript words.
        chunk_overlap_tokens: Model-token context on each side of long chunks.
    """
    _validate_transcript(transcript)
    tool = extractor or GlinerNamedEntityExtractor()
    text, units = words_to_text_units(
        transcript.words,
        separator=separator,
        weight_fn=tool._model_token_weight,
    )
    output = tool._process_with_units(
        text,
        units,
        labels,
        threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
        batch_size=batch_size,
        language=language,
        chunk_overlap_tokens=chunk_overlap_tokens,
        languages=transcript.languages,
        media_duration=transcript.media_duration,
        extra_parameters={
            "text_source": "transcript_words",
            "text_separator": separator,
        },
    )
    named_entities = output.output
    if not isinstance(named_entities, NamedEntities):
        raise RuntimeError("GLiNER tool did not return NamedEntities")
    output.messages.extend(
        named_entities.align_timestamps(transcript.words, separator=separator)
    )
    return output


def extract_named_entities_from_file(
    source: str | PathLike[str],
    labels: Sequence[str],
    *,
    extractor: GlinerNamedEntityExtractor | None = None,
    encoding: str = "utf-8",
    threshold: float | None = None,
    flat_ner: bool = True,
    multi_label: bool = False,
    batch_size: int | None = None,
    language: str | None = None,
    chunk_overlap_tokens: int = 0,
) -> ToolOutput:
    """Read a caller-owned text file and pass its contents to GLiNER.

    Args:
        source: Path to the source text file; the file remains caller-owned.
        labels: Entity labels requested from GLiNER.
        extractor: Optional configured or preloaded extractor.
        encoding: Character encoding used to read the file.
        threshold: Optional confidence threshold; ``None`` uses GLiNER's default.
        flat_ner: If true, prevent nested entity spans.
        multi_label: If true, allow more than one label for a span.
        batch_size: Optional GLiNER inference batch size.
        language: Optional language assigned to the output and entities.
        chunk_overlap_tokens: Model-token context on each side of long chunks.
    """
    if isinstance(source, bytes) or not isinstance(source, (str, PathLike)):
        raise TypeError("source must be a string or path-like object")
    text = Path(source).read_text(encoding=encoding)
    tool = extractor or GlinerNamedEntityExtractor()
    return tool.process(
        text,
        labels,
        threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
        batch_size=batch_size,
        language=language,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )


def _validate_transcript(transcript: Transcript) -> None:
    """Reject transcripts that cannot produce meaningful canonical text."""
    if not isinstance(transcript, Transcript):
        raise TypeError("transcript must be a Transcript")
    if not transcript.words:
        raise ValueError("transcript must contain at least one word")
    for index, word in enumerate(transcript.words):
        if not word.word.strip():
            raise ValueError(f"transcript word {index} must not be empty")
