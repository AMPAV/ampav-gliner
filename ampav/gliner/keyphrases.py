"""Preliminary keyphrase extraction with classic GLiNER."""

from __future__ import annotations

from typing import Any


DEFAULT_KEYPHRASE_MODEL_ID = "knowledgator/gliner-multitask-large-v0.5"
DEFAULT_KEYPHRASE_MODEL_REVISION = "7a95e168036db9ec6f914c0cc6b218edbd87f310"
DEFAULT_KEYPHRASE_PROMPT = (
    "Extract phrases that best represent the main subjects of the entire text; "
    "exclude generic noun phrases, pronouns, and incidental details."
)
DEFAULT_KEYPHRASE_LABEL = "key phrase"


class GlinerKeyPhraseExtractor:
    """Thin synchronous wrapper around tested GLiNER keyphrase behavior.

    The default model, prompt, label, and threshold are one tested preset, not
    model-independent keyphrase semantics. This preliminary wrapper processes
    one caller-supplied context window and returns unchanged native predictions
    with the prepared text needed to interpret prompt-relative offsets.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_KEYPHRASE_MODEL_ID,
        revision: str = DEFAULT_KEYPHRASE_MODEL_REVISION,
        device: str = "cpu",
        prompt: str | None = DEFAULT_KEYPHRASE_PROMPT,
        local_files_only: bool = False,
        model: Any | None = None,
    ) -> None:
        """Configure a lazily loaded GLiNER keyphrase extractor.

        Args:
            model_id: Hugging Face model ID used when ``model`` is not supplied.
            revision: Pinned model revision used when loading from Hugging Face.
            device: Device receiving the lazily loaded model.
            prompt: Instruction prepended to source text. Use ``None`` for
                unprompted label-driven extraction, such as ``key concept``.
            local_files_only: If true, require model artifacts in the local
                Hugging Face cache.
            model: Optional preloaded compatible GLiNER model.
        """
        if prompt is not None and not prompt.strip():
            raise ValueError("prompt must not be empty")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.prompt = prompt
        self.local_files_only = local_files_only
        self._model = model

    @property
    def model(self) -> Any:
        """Return the loaded GLiNER model, loading it on first use."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        """Load the configured classic GLiNER model and move it to the device."""
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise RuntimeError("gliner is required to extract keyphrases") from exc
        model = GLiNER.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        return model.to(self.device)

    def process(
        self,
        text: str,
        *,
        label: str = DEFAULT_KEYPHRASE_LABEL,
        threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Return native predictions and their exact inference context.

        Args:
            text: Source text that fits the selected model after any configured
                prompt is prepended.
            label: One native extraction label. Label variants are deliberately
                evaluated separately because combined labels change behavior.
            threshold: Native GLiNER span-selection threshold.

        Returns:
            A mapping containing ``native_predictions`` unchanged from GLiNER,
            the exact ``prepared_text``, and ``source_start``. Native offsets
            refer to ``prepared_text``; subtract ``source_start`` for spans
            wholly within the caller's source.
        """
        _validate_text(text)
        clean_label = _validate_label(label)
        if self.prompt is None:
            prepared_text = text
            source_start = 0
        else:
            prefix = f"{self.prompt} \n "
            prepared_text = prefix + text
            source_start = len(prefix)
        native_predictions = self.model.predict_entities(
            prepared_text,
            [clean_label],
            threshold=threshold,
            batch_size=1,
        )
        return {
            "prompt": self.prompt,
            "label": clean_label,
            "threshold": threshold,
            "prepared_text": prepared_text,
            "source_start": source_start,
            "native_predictions": native_predictions,
        }


def _validate_text(text: str) -> None:
    """Reject invalid text before loading or invoking the model."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")


def _validate_label(label: str) -> str:
    """Return one stripped native label after minimal validation."""
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    clean_label = label.strip()
    if not clean_label:
        raise ValueError("label must not be empty")
    return clean_label
