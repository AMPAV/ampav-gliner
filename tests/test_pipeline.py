import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ampav.core.schema import NamedEntities, Transcript, WordSegment
from ampav.gliner import GlinerNamedEntityExtractor
from ampav_gliner_pipeline import (
    extract_named_entities_from_file,
    extract_named_entities_from_transcript,
)


class FakeGlinerModel:
    def __init__(self, predictions: list[dict], *, max_len: int = 384) -> None:
        self.predictions = predictions
        self.config = SimpleNamespace(max_len=max_len)
        self.calls: list[dict] = []
        self.prepared_texts: list[str] = []

    def prepare_inputs(self, texts: list[str]) -> tuple[list[list[str]], list, list]:
        self.prepared_texts.extend(texts)
        return [text.split() for text in texts], [], []

    def predict_entities(self, text: str, labels: list[str], **kwargs: object) -> list[dict]:
        self.calls.append({"text": text, "labels": labels, "kwargs": kwargs})
        return self.predictions


class TranscriptPipelineTest(unittest.TestCase):
    def test_transcript_adapter_builds_units_and_aligns_timestamps(self) -> None:
        transcript = Transcript(
            media_duration=12.3,
            text="This raw transcript text is not used for model input.",
            languages=["en"],
            words=[
                WordSegment(word="IU", suffix="'s", start_time=0.0, end_time=0.2),
                WordSegment(word="Media", start_time=0.3, end_time=0.5),
                WordSegment(word="School", start_time=0.6, end_time=0.8),
                WordSegment(word="invited", start_time=0.9, end_time=1.1),
                WordSegment(word="Maya", start_time=1.2, end_time=1.4),
                WordSegment(word="Chen", suffix=".", start_time=1.5, end_time=1.7),
            ],
        )
        model = FakeGlinerModel(
            [
                {
                    "start": 0,
                    "end": 2,
                    "text": "IU",
                    "label": "organization",
                    "score": 0.91,
                },
                {
                    "start": 26,
                    "end": 35,
                    "text": "Maya Chen",
                    "label": "person",
                    "score": 0.97,
                },
            ]
        )
        extractor = GlinerNamedEntityExtractor(model=model)

        result = extract_named_entities_from_transcript(
            transcript,
            ["organization", "person"],
            extractor=extractor,
        )

        self.assertEqual(model.calls[0]["text"], "IU's Media School invited Maya Chen.")
        self.assertEqual(
            model.prepared_texts,
            ["IU's ", "Media ", "School ", "invited ", "Maya ", "Chen."],
        )
        self.assertEqual(result.parameters["text_source"], "transcript_words")
        self.assertEqual(result.parameters["text_separator"], " ")
        self.assertEqual(result.messages, [])
        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        self.assertEqual(result.output.text, "IU's Media School invited Maya Chen.")
        self.assertEqual(result.output.media_duration, 12.3)
        self.assertEqual(result.output.languages, ["en"])
        self.assertEqual(
            [
                (entity.text, entity.start_time, entity.end_time, entity.language)
                for entity in result.output.spans
            ],
            [
                ("IU", 0.0, 0.2, "en"),
                ("Maya Chen", 1.2, 1.7, "en"),
            ],
        )

    def test_transcript_adapter_rejects_empty_words(self) -> None:
        extractor = GlinerNamedEntityExtractor(model=FakeGlinerModel([]))

        with self.assertRaises(ValueError):
            extract_named_entities_from_transcript(
                Transcript(),
                ["person"],
                extractor=extractor,
            )
        with self.assertRaises(ValueError):
            extract_named_entities_from_transcript(
                Transcript(words=[WordSegment(word="")]),
                ["person"],
                extractor=extractor,
            )

    def test_transcript_adapter_allows_punctuation_only_words(self) -> None:
        model = FakeGlinerModel([])
        extractor = GlinerNamedEntityExtractor(model=model)

        result = extract_named_entities_from_transcript(
            Transcript(words=[WordSegment(word=".")]),
            ["person"],
            extractor=extractor,
        )

        self.assertEqual(model.calls[0]["text"], ".")
        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        self.assertEqual(result.output.spans, [])


class FilePipelineTest(unittest.TestCase):
    def test_file_adapter_reads_caller_owned_utf8_text(self) -> None:
        predictions = [
            {
                "start": 0,
                "end": 9,
                "text": "Maya Chen",
                "label": "person",
                "score": 0.98,
            }
        ]
        model = FakeGlinerModel(predictions)
        extractor = GlinerNamedEntityExtractor(model=model)

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "transcript.txt"
            source.write_text("Maya Chen visited 東京.", encoding="utf-8")

            result = extract_named_entities_from_file(
                source,
                ["person"],
                extractor=extractor,
            )

            self.assertTrue(source.exists())

        self.assertEqual(model.calls[0]["text"], "Maya Chen visited 東京.")
        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        self.assertEqual(result.output.text, "Maya Chen visited 東京.")
        self.assertEqual(result.output.spans[0].begin_offset, 0)
        self.assertEqual(result.output.spans[0].end_offset, 9)


if __name__ == "__main__":
    unittest.main()
