"""Preliminary extractive summarization with classic GLiNER."""

from __future__ import annotations

from typing import Any


DEFAULT_SUMMARIZATION_MODEL_ID = "knowledgator/gliner-multitask-large-v0.5"


class GlinerSummarizer:
    """Thin synchronous wrapper around GLiNER's native summarization pipeline.

    This preliminary wrapper processes one caller-supplied context window. It
    does not chunk long documents or aggregate independent chunk summaries.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_SUMMARIZATION_MODEL_ID,
        device: str = "cpu",
        prompt: str | None = None,
        model: Any | None = None,
    ) -> None:
        """Configure a lazily constructed native ``GLiNERSummarizer``.

        Args:
            model_id: Hugging Face model ID used when ``model`` is not supplied.
            device: Device passed to GLiNER's summarization pipeline.
            prompt: Optional summarization prompt replacing GLiNER's default.
            model: Optional preloaded classic GLiNER model. The native pipeline
                moves it to ``device``.
        """
        self.model_id = model_id
        self.device = device
        self.prompt = prompt
        self._model = model
        self._summarizer: Any | None = None

    @property
    def summarizer(self) -> Any:
        """Return the native summarizer, constructing it on first use."""
        if self._summarizer is None:
            self._summarizer = self._load_summarizer()
        return self._summarizer

    def _load_summarizer(self) -> Any:
        """Construct GLiNER's task-specific high-level summarizer."""
        try:
            from gliner.multitask import GLiNERSummarizer
        except ImportError as exc:
            raise RuntimeError(
                "gliner and its multitask dependencies are required to summarize text"
            ) from exc

        return GLiNERSummarizer(
            model_id=self.model_id,
            model=self._model,
            device=self.device,
            prompt=self.prompt,
        )

    def process(
        self,
        text: str,
        *,
        threshold: float = 0.25,
        batch_size: int = 8,
    ) -> list[str]:
        """Return GLiNER's native extractive summary result for one text.

        GLiNER returns a one-item list even for a single input string. Phase 1
        preserves that public result rather than introducing an AMPAV schema.

        Args:
            text: Source text that already fits the selected model's context
                window.
            threshold: Native GLiNER entity-selection threshold.
            batch_size: Native GLiNER inference batch size.

        Returns:
            GLiNER's one-item list containing its joined extractive summary.
        """
        _validate_text(text)
        return self.summarizer(
            text,
            labels=["summary"],
            threshold=threshold,
            batch_size=batch_size,
        )


def _validate_text(text: str) -> None:
    """Reject invalid text before loading or invoking the native model."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")
