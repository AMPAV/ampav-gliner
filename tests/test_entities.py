import os
import unittest
from importlib.metadata import version

from ampav.core.schema import NamedEntities, ToolOutput, Transcript, WordSegment

from ampav.gliner import (
    GlinerNamedEntityExtractor,
)


SAMPLE_TEXT = (
    "Maya Chen visited Indiana University. "
    "Maya Chen later met Rafael Ortiz."
)
REAL_MODEL_TEST_ENV = "AMPAV_GLINER_REAL_MODEL_TEST"
PROBE_THRESHOLD = 0.4

ProbeCase = tuple[str, str, list[str], list[tuple[str, str, int, int]]]
PROBE_CASES: list[ProbeCase] = [
    (
        "smoke",
        "Dr. Maya Chen visited Indiana University, then met Rafael Ortiz in Bloomington.",
        ["person", "organization", "location"],
        [
            ("Dr. Maya Chen", "person", 0, 13),
            ("Indiana University", "organization", 22, 40),
            ("Rafael Ortiz", "person", 51, 63),
            ("Bloomington", "location", 67, 78),
        ],
    ),
    (
        "honorific",
        "Dr. Smith met Prof. Anne-Marie ONeill at IU.",
        ["person", "organization"],
        [
            ("Dr. Smith", "person", 0, 9),
            ("Prof. Anne-Marie ONeill", "person", 14, 37),
            ("IU", "organization", 41, 43),
        ],
    ),
    (
        "possessive_apostrophe",
        "IU's Media School invited Maya Chen.",
        ["organization", "person"],
        [
            ("IU", "organization", 0, 2),
            ("Media School", "organization", 5, 17),
            ("Maya Chen", "person", 26, 35),
        ],
    ),
    (
        "quoted",
        'The panel quoted "Rafael Ortiz" during the session.',
        ["person"],
        [("Rafael Ortiz", "person", 18, 30)],
    ),
    (
        "parenthesized",
        "The award went to Maya Chen (Indiana University).",
        ["person", "organization"],
        [
            ("Maya Chen", "person", 18, 27),
            ("Indiana University", "organization", 29, 47),
        ],
    ),
    (
        "trailing_punctuation",
        "Rafael Ortiz, Maya Chen, and Indiana University attended.",
        ["person", "organization"],
        [
            ("Rafael Ortiz", "person", 0, 12),
            ("Maya Chen", "person", 14, 23),
            ("Indiana University", "organization", 29, 47),
        ],
    ),
    (
        "repeated",
        "Maya Chen thanked Maya Chen after the talk.",
        ["person"],
        [
            ("Maya Chen", "person", 0, 9),
            ("Maya Chen", "person", 18, 27),
        ],
    ),
    (
        "internal_punctuation",
        "Anne-Marie O'Neill visited WTIU-TV in Bloomington.",
        ["person", "organization", "location"],
        [
            ("Anne-Marie O'Neill", "person", 0, 18),
            ("WTIU-TV", "organization", 27, 34),
            ("Bloomington", "location", 38, 49),
        ],
    ),
]


class FakeGlinerModel:
    def __init__(self, predictions: list[dict]) -> None:
        self.predictions = predictions
        self.calls: list[dict] = []

    def predict_entities(self, text: str, labels: list[str], **kwargs: object) -> list[dict]:
        self.calls.append({"text": text, "labels": labels, "kwargs": kwargs})
        return self.predictions


class ProbeGlinerModel:
    def __init__(self, cases: list[ProbeCase]) -> None:
        self.predictions_by_text = {
            text: [
                {"text": entity_text, "label": entity_type, "start": start, "end": end, "score": 0.9}
                for entity_text, entity_type, start, end in expected_entities
            ]
            for _, text, _, expected_entities in cases
        }
        self.calls: list[dict] = []

    def predict_entities(self, text: str, labels: list[str], **kwargs: object) -> list[dict]:
        self.calls.append({"text": text, "labels": labels, "kwargs": kwargs})
        return self.predictions_by_text[text]


def assert_probe_case(
    test_case: unittest.TestCase,
    extractor: GlinerNamedEntityExtractor,
    name: str,
    text: str,
    labels: list[str],
    expected_entities: list[tuple[str, str, int, int]],
) -> ToolOutput:
    result = extractor.extract(text, labels, threshold=PROBE_THRESHOLD)

    test_case.assertIsInstance(result, ToolOutput, name)
    test_case.assertEqual(result.tool_name, "gliner", name)
    test_case.assertEqual(result.parameters["labels"], labels, name)
    test_case.assertEqual(result.parameters["threshold"], PROBE_THRESHOLD, name)
    test_case.assertIsInstance(result.output, NamedEntities, name)
    assert isinstance(result.output, NamedEntities)
    test_case.assertEqual(result.output.ampav_format, "named_entities/1", name)
    test_case.assertEqual(result.output.text, text, name)
    test_case.assertEqual(result.messages, [], name)

    actual_entities = [
        (entity.text, entity.entity_type, entity.begin_offset, entity.end_offset)
        for entity in result.output.spans
    ]
    test_case.assertEqual(actual_entities, expected_entities, name)
    for entity in result.output.spans:
        test_case.assertIsNotNone(entity.begin_offset, name)
        test_case.assertIsNotNone(entity.end_offset, name)
        assert entity.begin_offset is not None
        assert entity.end_offset is not None
        test_case.assertEqual(text[entity.begin_offset : entity.end_offset], entity.text, name)
        test_case.assertIsNone(entity.start_time, name)
        test_case.assertIsNone(entity.end_time, name)
    return result


class GlinerNamedEntityExtractorTest(unittest.TestCase):
    def test_extract_returns_tool_output_with_named_entities(self) -> None:
        model = FakeGlinerModel(
            [
                {"start": 0, "end": 9, "text": "Maya Chen", "label": "person", "score": 0.98},
                {"start": 18, "end": 36, "text": "Indiana University", "label": "organization", "score": 0.95},
            ]
        )
        extractor = GlinerNamedEntityExtractor(model=model)

        result = extractor.extract(SAMPLE_TEXT, ["person", "organization"], language="en")

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.tool_name, "gliner")
        self.assertEqual(result.tool_version, version("ampav-gliner"))
        self.assertIsInstance(result.output, NamedEntities)
        assert result.output is not None
        self.assertEqual(result.output.text, SAMPLE_TEXT)
        self.assertEqual(result.output.languages, ["en"])
        self.assertEqual(len(result.output.spans), 2)
        self.assertEqual(result.output.spans[0].text, "Maya Chen")
        self.assertEqual(result.output.spans[0].entity_type, "person")
        self.assertEqual(result.output.spans[0].confidence, 0.98)
        self.assertEqual(result.output.spans[0].begin_offset, 0)
        self.assertEqual(result.output.spans[0].end_offset, 9)
        self.assertEqual(result.output.spans[0].language, "en")

    def test_extract_records_model_and_inference_parameters(self) -> None:
        model = FakeGlinerModel([])
        extractor = GlinerNamedEntityExtractor(
            model_id="local/model",
            cache_dir="/tmp/gliner-cache",
            local_files_only=True,
            revision="abc123",
            map_location="cpu",
            model=model,
        )

        result = extractor.extract(
            "Maya Chen visited Bloomington.",
            [" person ", "location"],
            threshold=0.4,
            flat_ner=True,
            multi_label=False,
            batch_size=8,
            language="en",
        )

        self.assertEqual(
            result.parameters,
            {
                "model_id": "local/model",
                "cache_dir": "/tmp/gliner-cache",
                "local_files_only": True,
                "revision": "abc123",
                "map_location": "cpu",
                "labels": ["person", "location"],
                "threshold": 0.4,
                "flat_ner": True,
                "multi_label": False,
                "batch_size": 8,
                "language": "en",
            },
        )
        self.assertEqual(
            model.calls[0]["kwargs"],
            {
                "flat_ner": True,
                "multi_label": False,
                "threshold": 0.4,
                "batch_size": 8,
            },
        )

    def test_extract_uses_gliner_defaults_when_threshold_and_batch_size_are_none(self) -> None:
        model = FakeGlinerModel([])
        extractor = GlinerNamedEntityExtractor(model=model)

        extractor.extract("Maya Chen visited Bloomington.", ["person"])

        self.assertEqual(model.calls[0]["kwargs"], {"flat_ner": True, "multi_label": False})

    def test_extract_includes_raw_predictions_in_tool_private_by_default(self) -> None:
        predictions = [{"start": 0, "end": 9, "text": "Maya Chen", "label": "person", "score": 0.98}]
        model = FakeGlinerModel(predictions)
        extractor = GlinerNamedEntityExtractor(model=model)

        result = extractor.extract(
            "Maya Chen visited Bloomington.",
            ["person"],
        )

        self.assertEqual(result.tool_private, {"gliner_predictions": predictions})

    def test_extract_can_omit_raw_predictions_from_tool_private(self) -> None:
        predictions = [{"start": 0, "end": 9, "text": "Maya Chen", "label": "person", "score": 0.98}]
        model = FakeGlinerModel(predictions)
        extractor = GlinerNamedEntityExtractor(model=model, include_tool_private=False)

        result = extractor.extract(
            "Maya Chen visited Bloomington.",
            ["person"],
        )

        self.assertIsNone(result.tool_private)

    def test_extract_from_transcript_aligns_timestamps_from_canonical_words(self) -> None:
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
                {"start": 0, "end": 2, "text": "IU", "label": "organization", "score": 0.91},
                {"start": 26, "end": 35, "text": "Maya Chen", "label": "person", "score": 0.97},
            ]
        )
        extractor = GlinerNamedEntityExtractor(model=model)

        result = extractor.extract_from_transcript(transcript, ["organization", "person"])

        self.assertEqual(model.calls[0]["text"], "IU's Media School invited Maya Chen.")
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

    def test_extract_from_transcript_reports_alignment_failures_as_messages(self) -> None:
        transcript = Transcript(
            words=[
                WordSegment(word="Amazon", start_time=1.0, end_time=1.5),
            ],
        )
        predictions = [{"text": "Amazon", "label": "organization", "score": 0.92}]
        model = FakeGlinerModel(predictions)
        extractor = GlinerNamedEntityExtractor(model=model)

        result = extractor.extract_from_transcript(transcript, ["organization"])

        self.assertEqual(result.messages, ["Text span 0 timestamp alignment skipped: missing offsets."])
        self.assertEqual(result.tool_private, {"gliner_predictions": predictions})
        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        self.assertIsNone(result.output.spans[0].start_time)
        self.assertIsNone(result.output.spans[0].end_time)
        self.assertIsNone(result.output.spans[0].tool_private)

    def test_extract_from_transcript_rejects_empty_words(self) -> None:
        extractor = GlinerNamedEntityExtractor(model=FakeGlinerModel([]))

        with self.assertRaises(ValueError):
            extractor.extract_from_transcript(Transcript(), ["person"])

        with self.assertRaises(ValueError):
            extractor.extract_from_transcript(Transcript(words=[WordSegment(word="")]), ["person"])

    def test_extract_from_transcript_allows_punctuation_only_words(self) -> None:
        model = FakeGlinerModel([])
        extractor = GlinerNamedEntityExtractor(model=model)

        result = extractor.extract_from_transcript(
            Transcript(words=[WordSegment(word=".")]),
            ["person"],
        )

        self.assertEqual(model.calls[0]["text"], ".")
        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        self.assertEqual(result.output.spans, [])

    def test_conversion_preserves_multiple_mentions(self) -> None:
        predictions = [
            {"start": 0, "end": 9, "text": "Maya Chen", "label": "person", "score": 0.98},
            {"start": 38, "end": 47, "text": "Maya Chen", "label": "person", "score": 0.97},
        ]
        model = FakeGlinerModel(predictions)
        extractor = GlinerNamedEntityExtractor(model=model)

        result = extractor.extract(SAMPLE_TEXT, ["person"])

        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        entities = result.output
        self.assertEqual([entity.text for entity in entities.spans], ["Maya Chen", "Maya Chen"])
        self.assertEqual([entity.begin_offset for entity in entities.spans], [0, 38])

    def test_validate_labels_rejects_empty_labels(self) -> None:
        extractor = GlinerNamedEntityExtractor(model=FakeGlinerModel([]))
        with self.assertRaises(ValueError):
            extractor.extract("Maya Chen visited Bloomington.", ["person", " "])

    def test_validate_labels_rejects_exact_duplicates(self) -> None:
        extractor = GlinerNamedEntityExtractor(model=FakeGlinerModel([]))
        with self.assertRaises(ValueError):
            extractor.extract("Maya Chen visited Bloomington.", ["person", "person"])

    def test_validate_labels_rejects_normalized_duplicates(self) -> None:
        extractor = GlinerNamedEntityExtractor(model=FakeGlinerModel([]))
        with self.assertRaises(ValueError):
            extractor.extract("Maya Chen visited Bloomington.", ["Person", " person "])
        with self.assertRaises(ValueError):
            extractor.extract("Maya Chen visited Bloomington.", ["geo  location", "geo location"])

    def test_validate_labels_rejects_string_instead_of_sequence(self) -> None:
        extractor = GlinerNamedEntityExtractor(model=FakeGlinerModel([]))
        with self.assertRaises(TypeError):
            extractor.extract("Maya Chen visited Bloomington.", "person")

    def test_extract_rejects_empty_text(self) -> None:
        extractor = GlinerNamedEntityExtractor(model=FakeGlinerModel([]))

        with self.assertRaises(ValueError):
            extractor.extract(" ", ["person"])


class GlinerProbeFixtureTest(unittest.TestCase):
    def test_probe_fixtures_with_fake_model(self) -> None:
        model = ProbeGlinerModel(PROBE_CASES)
        extractor = GlinerNamedEntityExtractor(model=model)

        for name, text, labels, expected_entities in PROBE_CASES:
            with self.subTest(name=name):
                assert_probe_case(self, extractor, name, text, labels, expected_entities)

        self.assertEqual(
            [call["kwargs"] for call in model.calls],
            [{"flat_ner": True, "multi_label": False, "threshold": PROBE_THRESHOLD}] * len(PROBE_CASES),
        )


@unittest.skipUnless(
    os.environ.get(REAL_MODEL_TEST_ENV) == "1",
    f"set {REAL_MODEL_TEST_ENV}=1 to run cached real GLiNER model tests",
)
class GlinerRealModelProbeTest(unittest.TestCase):
    def test_probe_fixtures_with_cached_default_model(self) -> None:
        extractor = GlinerNamedEntityExtractor(local_files_only=True)

        for name, text, labels, expected_entities in PROBE_CASES:
            with self.subTest(name=name):
                assert_probe_case(self, extractor, name, text, labels, expected_entities)


if __name__ == "__main__":
    unittest.main()
