"""Tests for preliminary GLiNER keyphrase extraction."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from ampav.gliner import (
    DEFAULT_KEYPHRASE_LABEL,
    DEFAULT_KEYPHRASE_MODEL_ID,
    DEFAULT_KEYPHRASE_MODEL_REVISION,
    DEFAULT_KEYPHRASE_PROMPT,
    GlinerKeyPhraseExtractor,
)


class FakeGlinerModel:
    """Return a stable native result and retain inference calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.result = [
            {
                "text": "AMPAV",
                "label": "key phrase",
                "score": 0.75,
                "start": 10,
                "end": 15,
            }
        ]

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        **kwargs: object,
    ) -> list[dict[str, object]]:
        self.calls.append({"text": text, "labels": labels, "kwargs": kwargs})
        return self.result


class GlinerKeyPhraseExtractorTest(unittest.TestCase):
    def test_defaults_match_adopted_native_preset(self) -> None:
        model = FakeGlinerModel()
        extractor = GlinerKeyPhraseExtractor(model=model)

        result = extractor.process("AMPAV supports audiovisual metadata.")

        expected_prefix = f"{DEFAULT_KEYPHRASE_PROMPT} \n "
        self.assertEqual(extractor.model_id, DEFAULT_KEYPHRASE_MODEL_ID)
        self.assertEqual(extractor.revision, DEFAULT_KEYPHRASE_MODEL_REVISION)
        self.assertEqual(result["prompt"], DEFAULT_KEYPHRASE_PROMPT)
        self.assertEqual(result["label"], DEFAULT_KEYPHRASE_LABEL)
        self.assertEqual(result["threshold"], 0.25)
        self.assertEqual(result["source_start"], len(expected_prefix))
        self.assertEqual(
            result["prepared_text"],
            expected_prefix + "AMPAV supports audiovisual metadata.",
        )
        self.assertIs(result["native_predictions"], model.result)
        self.assertEqual(model.calls[0]["labels"], ["key phrase"])
        self.assertEqual(
            model.calls[0]["kwargs"],
            {"threshold": 0.25, "batch_size": 1},
        )

    def test_supports_unprompted_key_concept_configuration(self) -> None:
        model = FakeGlinerModel()
        extractor = GlinerKeyPhraseExtractor(model=model, prompt=None)

        result = extractor.process("Source text", label=" key concept ")

        self.assertIsNone(result["prompt"])
        self.assertEqual(result["label"], "key concept")
        self.assertEqual(result["prepared_text"], "Source text")
        self.assertEqual(result["source_start"], 0)
        self.assertEqual(model.calls[0]["text"], "Source text")
        self.assertEqual(model.calls[0]["labels"], ["key concept"])

    def test_lazy_load_forwards_pinned_model_settings(self) -> None:
        loaded_model = Mock()
        moved_model = object()
        loaded_model.to.return_value = moved_model
        with patch("gliner.GLiNER.from_pretrained", return_value=loaded_model) as load:
            extractor = GlinerKeyPhraseExtractor(
                model_id="test/model",
                revision="abc123",
                device="cuda",
                local_files_only=True,
            )

            self.assertIs(extractor.model, moved_model)

        load.assert_called_once_with(
            "test/model",
            revision="abc123",
            local_files_only=True,
        )
        loaded_model.to.assert_called_once_with("cuda")

    def test_rejects_invalid_text_label_and_prompt(self) -> None:
        model = FakeGlinerModel()
        extractor = GlinerKeyPhraseExtractor(model=model)

        with self.assertRaisesRegex(TypeError, "text must be a string"):
            extractor.process(None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "text must not be empty"):
            extractor.process("  ")
        with self.assertRaisesRegex(TypeError, "label must be a string"):
            extractor.process("text", label=None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "label must not be empty"):
            extractor.process("text", label="  ")
        with self.assertRaisesRegex(ValueError, "prompt must not be empty"):
            GlinerKeyPhraseExtractor(model=model, prompt="  ")


if __name__ == "__main__":
    unittest.main()
