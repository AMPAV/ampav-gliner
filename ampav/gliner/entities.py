"""Named-entity extraction with GLiNER."""

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from time import time
from typing import Any

from ampav.core.schema import NamedEntities, NamedEntity, ToolOutput
from ampav.core.text_chunking import (
    TextChunk,
    TextUnit,
    chunk_text,
    dechunk_text_spans,
    text_to_units,
)

from ._version import DISTRIBUTION_NAME, __version__


DEFAULT_MODEL_ID = "urchade/gliner_small-v2.1"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlinerModelOptions:
    """Model-loading settings for the GLiNER wrapper.

    ``model_id`` and ``revision`` select model artifacts; ``cache_dir`` and
    ``local_files_only`` control Hugging Face cache access; ``map_location``
    selects the device used when GLiNER loads model weights.
    """

    model_id: str = DEFAULT_MODEL_ID
    cache_dir: str | Path | None = None
    local_files_only: bool = False
    revision: str | None = None
    map_location: str = "cpu"

    def to_parameters(self) -> dict[str, Any]:
        """Return JSON-friendly parameter values for `ToolOutput.parameters`."""
        return {
            "model_id": self.model_id,
            "cache_dir": None if self.cache_dir is None else str(self.cache_dir),
            "local_files_only": self.local_files_only,
            "revision": self.revision,
            "map_location": self.map_location,
        }

    def load_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for `GLiNER.from_pretrained`."""
        kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "map_location": self.map_location,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        if self.revision is not None:
            kwargs["revision"] = self.revision
        return kwargs


class GlinerNamedEntityExtractor:
    """Thin synchronous wrapper around GLiNER named-entity extraction."""

    distribution_name = DISTRIBUTION_NAME
    tool_name = "gliner"
    tool_version = __version__

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        map_location: str = "cpu",
        model: Any | None = None,
        include_tool_private: bool = False,
    ) -> None:
        """Configure a lazily loaded GLiNER extractor.

        Args:
            model_id: Hugging Face model ID or local model directory.
            cache_dir: Optional directory for downloaded model artifacts.
            local_files_only: If true, do not download missing model artifacts.
            revision: Optional model revision, branch, or commit.
            map_location: Device onto which GLiNER loads model weights.
            model: Optional preloaded compatible model, primarily for injection.
            include_tool_private: Include per-chunk native predictions for
                troubleshooting. Raw prediction offsets remain chunk-local.
        """
        self.model_options = GlinerModelOptions(
            model_id=model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            map_location=map_location,
        )
        self._model = model
        self.include_tool_private = include_tool_private

    @property
    def model(self) -> Any:
        """Return the loaded GLiNER model, loading it on first use."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        """Load the configured GLiNER model."""
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise RuntimeError("gliner is required to load a real GLiNER model") from exc
        return GLiNER.from_pretrained(
            self.model_options.model_id,
            **self.model_options.load_kwargs(),
        )

    def process(
        self,
        text: str,
        labels: Sequence[str],
        *,
        threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        batch_size: int | None = None,
        language: str | None = None,
        chunk_overlap_tokens: int = 0,
    ) -> ToolOutput:
        """Extract entities from text, chunking at the model's token limit.

        ``chunk_overlap_tokens`` adds best-effort model-token context on both
        sides of an owned chunk range. Overlapping predictions are assigned by
        their midpoint when the full-text result is reassembled.

        Args:
            text: Original source text. Final entity offsets refer to this text.
            labels: Non-empty, unique entity labels requested from GLiNER.
            threshold: Optional confidence threshold; ``None`` uses GLiNER's
                default.
            flat_ner: If true, prevent nested entity spans.
            multi_label: If true, allow more than one label for a span.
            batch_size: Optional GLiNER inference batch size.
            language: Optional language assigned to the output and each entity.
            chunk_overlap_tokens: Maximum model-token context added on each side
                of a chunk's owned range. It must leave positive owned capacity.
        """
        _validate_text(text)
        units = text_to_units(text, weight_fn=self._model_token_weight)
        return self._process_with_units(
            text,
            units,
            labels,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            batch_size=batch_size,
            language=language,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )

    def _process_with_units(
        self,
        text: str,
        units: Sequence[TextUnit],
        labels: Sequence[str],
        *,
        threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        batch_size: int | None = None,
        language: str | None = None,
        chunk_overlap_tokens: int = 0,
        languages: Sequence[str] | None = None,
        media_duration: float | None = None,
        extra_parameters: dict[str, Any] | None = None,
    ) -> ToolOutput:
        """Process text with prebuilt units supplied by an in-package adapter.

        This internal hook lets the transcript pipeline preserve the canonical
        text/unit ranges it builds from ``WordSegment`` objects. Public direct
        callers should use :meth:`process`.
        """
        _validate_text(text)
        clean_labels = _validate_labels(labels)
        max_tokens = self._model_max_tokens()
        chunks = chunk_text(
            text,
            units,
            max_weight=max_tokens,
            overlap_weight=chunk_overlap_tokens,
        )

        output_parameters: dict[str, Any] = {
            **self.model_options.to_parameters(),
            "labels": clean_labels,
            "threshold": threshold,
            "flat_ner": flat_ner,
            "multi_label": multi_label,
            "batch_size": batch_size,
            "language": language,
            "model_max_tokens": max_tokens,
            "chunk_overlap_tokens": chunk_overlap_tokens,
            "chunk_count": len(chunks),
        }
        if extra_parameters is not None:
            output_parameters.update(extra_parameters)

        logger.debug(
            "Processing %d GLiNER chunk(s) with a %d-token model limit",
            len(chunks),
            max_tokens,
        )

        output = ToolOutput(
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            parameters=output_parameters,
        )
        predict_kwargs: dict[str, Any] = {
            "flat_ner": flat_ner,
            "multi_label": multi_label,
        }
        if threshold is not None:
            predict_kwargs["threshold"] = threshold
        if batch_size is not None:
            predict_kwargs["batch_size"] = batch_size

        chunk_outputs: list[tuple[TextChunk, Sequence[NamedEntity]]] = []
        private_chunks: list[dict[str, Any]] = []
        output_languages = _output_languages(language, languages)
        entity_language = _entity_language(language, output_languages)
        output.start_time = time()
        for chunk_index, chunk in enumerate(chunks):
            logger.debug(
                "Processing GLiNER chunk %d/%d at source offsets %d:%d",
                chunk_index + 1,
                len(chunks),
                chunk.begin_offset,
                chunk.end_offset,
            )
            raw_predictions = self.model.predict_entities(
                chunk.text,
                clean_labels,
                **predict_kwargs,
            )
            chunk_entities = _gliner_predictions_to_named_entities(
                chunk.text,
                raw_predictions,
                language=entity_language,
            )
            chunk_outputs.append((chunk, chunk_entities.spans))
            if self.include_tool_private:
                # Raw offsets are chunk-local, so retain their source window.
                private_chunks.append(_private_chunk(chunk, raw_predictions))
        output.end_time = time()

        named_entities = NamedEntities(
            media_duration=media_duration,
            text=text,
            spans=dechunk_text_spans(text, chunk_outputs),
            languages=output_languages,
        )
        output.output = named_entities
        if self.include_tool_private:
            output.tool_private = {"gliner_chunks": private_chunks}
        return output

    def _model_max_tokens(self) -> int:
        """Return the loaded model's positive source-token limit."""
        max_tokens = getattr(getattr(self.model, "config", None), "max_len", None)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise RuntimeError("GLiNER model config.max_len must be a positive integer")
        return max_tokens

    def _model_token_weight(self, text: str) -> int:
        """Count the word-level tokens GLiNER uses for its ``max_len`` limit."""
        prepare_inputs = getattr(self.model, "prepare_inputs", None)
        if prepare_inputs is None:
            # Test doubles and alternate compatible models may expose only the
            # inference API. The shared unit builder's one-unit weight is the
            # conservative dependency-free fallback.
            return 1
        prepared = prepare_inputs([text])
        try:
            tokens = prepared[0][0]
        except (IndexError, TypeError) as exc:
            raise RuntimeError("GLiNER prepare_inputs returned an unexpected value") from exc
        if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
            raise RuntimeError("GLiNER prepare_inputs did not return a token sequence")
        # Core requires positive weights; punctuation-only source ranges may be
        # ignored by an alternate splitter but still need an indivisible unit.
        return max(1, len(tokens))


def _private_chunk(
    chunk: TextChunk,
    predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Describe one chunk and its chunk-local native predictions."""
    return {
        "begin_offset": chunk.begin_offset,
        "end_offset": chunk.end_offset,
        "owned_begin_offset": chunk.owned_begin_offset,
        "owned_end_offset": chunk.owned_end_offset,
        "predictions": predictions,
    }


def _gliner_predictions_to_named_entities(
    text: str,
    predictions: Sequence[dict[str, Any]],
    *,
    language: str | None = None,
) -> NamedEntities:
    """Convert chunk-local GLiNER dictionaries into AMPAV entities."""
    entities = [
        NamedEntity(
            text=str(prediction["text"]),
            entity_type=str(prediction["label"]),
            confidence=(
                None
                if prediction.get("score") is None
                else float(prediction["score"])
            ),
            begin_offset=(
                None if prediction.get("start") is None else int(prediction["start"])
            ),
            end_offset=(
                None if prediction.get("end") is None else int(prediction["end"])
            ),
            language=language,
        )
        for prediction in predictions
    ]
    return NamedEntities(
        text=text,
        spans=entities,
        languages=None if language is None else [language],
    )


def _validate_text(text: str) -> None:
    """Validate text input before calling GLiNER."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")


def _validate_labels(labels: Sequence[str]) -> list[str]:
    """Return stripped labels after rejecting duplicates and empty labels."""
    if isinstance(labels, str) or not isinstance(labels, Sequence):
        raise TypeError("labels must be a sequence of strings")

    clean_labels = []
    for label in labels:
        if not isinstance(label, str):
            raise TypeError("labels must contain only strings")
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("labels must not contain empty values")
        clean_labels.append(clean_label)

    if not clean_labels:
        raise ValueError("labels must contain at least one value")
    if len(set(clean_labels)) != len(clean_labels):
        raise ValueError("labels must not contain duplicate values")

    normalized_labels = [_normalize_label(label) for label in clean_labels]
    if len(set(normalized_labels)) != len(normalized_labels):
        raise ValueError("labels must not contain case or whitespace duplicate values")
    return clean_labels


def _output_languages(
    language: str | None,
    languages: Sequence[str] | None,
) -> list[str] | None:
    if language is not None:
        return [language]
    if languages is None:
        return None
    clean_languages = [item for item in languages if item]
    return clean_languages or None


def _entity_language(
    language: str | None,
    languages: list[str] | None,
) -> str | None:
    """Select a span language only when the source language is unambiguous."""
    if language is not None:
        return language
    if languages is not None and len(languages) == 1:
        return languages[0]
    return None


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).casefold()
