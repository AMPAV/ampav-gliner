"""Named-entity extraction with GLiNER."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from time import time
from typing import Any

from ampav.core.schema import NamedEntities, NamedEntity, ToolOutput, Transcript


DEFAULT_MODEL_ID = "urchade/gliner_small-v2.1"
TOOL_NAME = "gliner"
TOOL_VERSION = "0.0.1"


@dataclass(frozen=True)
class GlinerModelOptions:
    """Model-loading settings for the GLiNER wrapper."""

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

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        map_location: str = "cpu",
        model: Any | None = None,
    ) -> None:
        self.model_options = GlinerModelOptions(
            model_id=model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            map_location=map_location,
        )
        self._model = model

    @property
    def model(self) -> Any:
        """Return the loaded GLiNER model, loading it on first use."""
        if self._model is None:
            self._model = load_gliner_model(self.model_options)
        return self._model

    def extract(
        self,
        text: str,
        labels: Sequence[str],
        *,
        threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        batch_size: int | None = None,
        language: str | None = None,
        include_raw_output: bool = True,
    ) -> ToolOutput:
        """Extract named entities from plain text and return an AMPAV output."""
        validate_text(text)
        clean_labels = validate_labels(labels)
        return self._extract_text(
            text,
            clean_labels,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            batch_size=batch_size,
            language=language,
            include_raw_output=include_raw_output,
        )

    def extract_from_transcript(
        self,
        transcript: Transcript,
        labels: Sequence[str],
        *,
        threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        batch_size: int | None = None,
        language: str | None = None,
        separator: str = " ",
        include_raw_output: bool = True,
    ) -> ToolOutput:
        """Extract named entities from transcript words and align timestamps."""
        validate_transcript(transcript)
        text = transcript.words_to_text(separator=separator)
        validate_text(text)
        clean_labels = validate_labels(labels)

        output = self._extract_text(
            text,
            clean_labels,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            batch_size=batch_size,
            language=language,
            languages=transcript.languages,
            media_duration=transcript.media_duration,
            include_raw_output=include_raw_output,
            extra_parameters={
                "text_source": "transcript_words",
                "text_separator": separator,
            },
        )
        if isinstance(output.output, NamedEntities):
            output.messages.extend(output.output.align_timestamps(transcript.words, separator=separator))
        return output

    def _extract_text(
        self,
        text: str,
        labels: list[str],
        *,
        threshold: float | None,
        flat_ner: bool,
        multi_label: bool,
        batch_size: int | None,
        language: str | None,
        languages: Sequence[str] | None = None,
        media_duration: float | None = None,
        include_raw_output: bool,
        extra_parameters: dict[str, Any] | None = None,
    ) -> ToolOutput:
        """Run GLiNER against already-validated text."""
        output_parameters: dict[str, Any] = {
            **self.model_options.to_parameters(),
            "labels": labels,
            "threshold": threshold,
            "flat_ner": flat_ner,
            "multi_label": multi_label,
            "batch_size": batch_size,
            "language": language,
        }
        if extra_parameters is not None:
            output_parameters.update(extra_parameters)

        output = ToolOutput(
            tool_name=TOOL_NAME,
            tool_version=TOOL_VERSION,
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

        output.start_time = time()
        raw_predictions = self.model.predict_entities(text, labels, **predict_kwargs)
        output.end_time = time()

        output.output = gliner_predictions_to_named_entities(
            text,
            raw_predictions,
            language=language,
            languages=languages,
            media_duration=media_duration,
        )
        if include_raw_output:
            output.tool_private = {"gliner_predictions": raw_predictions}
        return output


def extract_named_entities(
    text: str,
    labels: Sequence[str],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    map_location: str = "cpu",
    threshold: float | None = None,
    flat_ner: bool = True,
    multi_label: bool = False,
    batch_size: int | None = None,
    language: str | None = None,
    model: Any | None = None,
    include_raw_output: bool = True,
) -> ToolOutput:
    """Extract named entities from plain text with a one-shot API."""
    extractor = GlinerNamedEntityExtractor(
        model_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=revision,
        map_location=map_location,
        model=model,
    )
    return extractor.extract(
        text,
        labels,
        threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
        batch_size=batch_size,
        language=language,
        include_raw_output=include_raw_output,
    )


def extract_named_entities_from_transcript(
    transcript: Transcript,
    labels: Sequence[str],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    map_location: str = "cpu",
    threshold: float | None = None,
    flat_ner: bool = True,
    multi_label: bool = False,
    batch_size: int | None = None,
    language: str | None = None,
    separator: str = " ",
    model: Any | None = None,
    include_raw_output: bool = True,
) -> ToolOutput:
    """Extract named entities from transcript words with a one-shot API."""
    extractor = GlinerNamedEntityExtractor(
        model_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        revision=revision,
        map_location=map_location,
        model=model,
    )
    return extractor.extract_from_transcript(
        transcript,
        labels,
        threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
        batch_size=batch_size,
        language=language,
        separator=separator,
        include_raw_output=include_raw_output,
    )


def load_gliner_model(options: GlinerModelOptions) -> Any:
    """Load a GLiNER model using the configured model-loading options."""
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError("gliner is required to load a real GLiNER model") from exc
    return GLiNER.from_pretrained(options.model_id, **options.load_kwargs())


def gliner_predictions_to_named_entities(
    text: str,
    predictions: Sequence[dict[str, Any]],
    *,
    language: str | None = None,
    languages: Sequence[str] | None = None,
    media_duration: float | None = None,
) -> NamedEntities:
    """Convert GLiNER prediction dictionaries into AMPAV `NamedEntities`."""
    output_languages = _output_languages(language, languages)
    entity_language = _entity_language(language, output_languages)
    entities = [
        NamedEntity(
            text=str(prediction["text"]),
            entity_type=str(prediction["label"]),
            confidence=None if prediction.get("score") is None else float(prediction["score"]),
            begin_offset=None if prediction.get("start") is None else int(prediction["start"]),
            end_offset=None if prediction.get("end") is None else int(prediction["end"]),
            language=entity_language,
        )
        for prediction in predictions
    ]
    return NamedEntities(
        media_duration=media_duration,
        text=text,
        spans=entities,
        languages=output_languages,
    )


def validate_text(text: str) -> None:
    """Validate text input before calling GLiNER."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")


def validate_labels(labels: Sequence[str]) -> list[str]:
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


def validate_transcript(transcript: Transcript) -> None:
    """Validate transcript input before building canonical text."""
    if not isinstance(transcript, Transcript):
        raise TypeError("transcript must be a Transcript")
    if not transcript.words:
        raise ValueError("transcript must contain at least one word")


def _output_languages(language: str | None, languages: Sequence[str] | None) -> list[str] | None:
    if language is not None:
        return [language]
    if languages is None:
        return None
    clean_languages = [item for item in languages if item]
    return clean_languages or None


def _entity_language(language: str | None, languages: list[str] | None) -> str | None:
    if language is not None:
        return language
    if languages is not None and len(languages) == 1:
        return languages[0]
    return None


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).casefold()
