"""Preliminary text classification with classic GLiNER."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


DEFAULT_CLASSIFICATION_MODEL_ID = "knowledgator/gliner-multitask-v1.0"


class GlinerTextClassifier:
    """Thin synchronous wrapper around GLiNER's native classifier.

    This preliminary wrapper classifies one caller-supplied context window. It
    does not chunk text, combine chunk classifications, or normalize labels to
    an AMPAV schema.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_CLASSIFICATION_MODEL_ID,
        device: str = "cpu",
        prompt: str | None = None,
        model: Any | None = None,
    ) -> None:
        """Configure a lazily constructed native ``GLiNERClassifier``.

        Args:
            model_id: Hugging Face model ID used when ``model`` is not supplied.
            device: Device passed to GLiNER's classification pipeline.
            prompt: Optional classification prompt template replacing GLiNER's
                default. It must contain the native ``{}`` class placeholder.
            model: Optional preloaded classic GLiNER model. The native pipeline
                moves it to ``device``.
        """
        if prompt is not None and "{}" not in prompt:
            raise ValueError("prompt must contain the '{}' class placeholder")
        self.model_id = model_id
        self.device = device
        self.prompt = prompt
        self._model = model
        self._classifier: Any | None = None

    @property
    def classifier(self) -> Any:
        """Return the native classifier, constructing it on first use."""
        if self._classifier is None:
            self._classifier = self._load_classifier()
        return self._classifier

    def _load_classifier(self) -> Any:
        """Construct GLiNER's task-specific high-level classifier."""
        try:
            from gliner.multitask import GLiNERClassifier
        except ImportError as exc:
            raise RuntimeError(
                "gliner and its multitask dependencies are required to classify text"
            ) from exc

        return GLiNERClassifier(
            model_id=self.model_id,
            model=self._model,
            device=self.device,
            prompt=self.prompt,
        )

    def process(
        self,
        text: str,
        classes: Sequence[str],
        *,
        multi_label: bool = False,
        threshold: float = 0.5,
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]:
        """Return GLiNER's native classification result for one text.

        GLiNER returns an outer batch list even for a single input string.
        Single-label mode applies the native helper's softmax post-processing;
        multi-label mode preserves its selected prediction scores.

        Args:
            text: Source text that fits the selected model after GLiNER adds its
                task prompt and candidate classes.
            classes: Explicit candidate class labels inserted into the native
                prompt.
            multi_label: Whether GLiNER may return multiple selected classes.
            threshold: Native GLiNER span-selection threshold.
            batch_size: Native GLiNER inference batch size.

        Returns:
            GLiNER's batch-shaped list of native ``label`` and ``score`` maps.
        """
        _validate_text(text)
        validated_classes = _validate_classes(classes)
        return self.classifier(
            text,
            labels=["match"],
            classes=validated_classes,
            multi_label=multi_label,
            threshold=threshold,
            batch_size=batch_size,
        )


def _validate_text(text: str) -> None:
    """Reject invalid text before loading or invoking the native model."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")


def _validate_classes(classes: Sequence[str]) -> list[str]:
    """Return a native-ready class list after minimal input validation."""
    if isinstance(classes, (str, bytes)) or not isinstance(classes, Sequence):
        raise TypeError("classes must be a sequence of strings")
    if not classes:
        raise ValueError("classes must not be empty")
    if any(not isinstance(label, str) for label in classes):
        raise TypeError("each class must be a string")
    if any(not label.strip() for label in classes):
        raise ValueError("class labels must not be empty")
    return list(classes)
