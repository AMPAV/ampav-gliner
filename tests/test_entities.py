import unittest

from ampav.core.schema import NamedEntities, ToolOutput

from ampav.gliner import (
    GlinerNamedEntityExtractor,
    extract_named_entities,
    gliner_predictions_to_named_entities,
    validate_labels,
)


SAMPLE_TEXT = (
    "Maya Chen visited Indiana University. "
    "Maya Chen later met Rafael Ortiz."
)


class FakeGlinerModel:
    def __init__(self, predictions: list[dict]) -> None:
        self.predictions = predictions
        self.calls: list[dict] = []

    def predict_entities(self, text: str, labels: list[str], **kwargs: object) -> list[dict]:
        self.calls.append({"text": text, "labels": labels, "kwargs": kwargs})
        return self.predictions


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
        self.assertIsInstance(result.output, NamedEntities)
        assert result.output is not None
        self.assertEqual(result.output.text, SAMPLE_TEXT)
        self.assertEqual(result.output.languages, ["en"])
        self.assertEqual(len(result.output.entities), 2)
        self.assertEqual(result.output.entities[0].entity_text, "Maya Chen")
        self.assertEqual(result.output.entities[0].entity_type, "person")
        self.assertEqual(result.output.entities[0].confidence, 0.98)
        self.assertEqual(result.output.entities[0].begin_offset, 0)
        self.assertEqual(result.output.entities[0].end_offset, 9)
        self.assertEqual(result.output.entities[0].language, "en")

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

    def test_extract_can_include_raw_predictions_in_tool_private(self) -> None:
        predictions = [{"start": 0, "end": 9, "text": "Maya Chen", "label": "person", "score": 0.98}]
        model = FakeGlinerModel(predictions)

        result = extract_named_entities(
            "Maya Chen visited Bloomington.",
            ["person"],
            model=model,
            include_raw_output=True,
        )

        self.assertEqual(result.tool_private, {"gliner_predictions": predictions})

    def test_conversion_preserves_multiple_mentions(self) -> None:
        predictions = [
            {"start": 0, "end": 9, "text": "Maya Chen", "label": "person", "score": 0.98},
            {"start": 38, "end": 47, "text": "Maya Chen", "label": "person", "score": 0.97},
        ]

        entities = gliner_predictions_to_named_entities(SAMPLE_TEXT, predictions)

        self.assertEqual([entity.entity_text for entity in entities.entities], ["Maya Chen", "Maya Chen"])
        self.assertEqual([entity.begin_offset for entity in entities.entities], [0, 38])

    def test_validate_labels_rejects_empty_labels(self) -> None:
        with self.assertRaises(ValueError):
            validate_labels(["person", " "])

    def test_validate_labels_rejects_exact_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            validate_labels(["person", "person"])

    def test_validate_labels_rejects_normalized_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            validate_labels(["Person", " person "])
        with self.assertRaises(ValueError):
            validate_labels(["geo  location", "geo location"])

    def test_validate_labels_rejects_string_instead_of_sequence(self) -> None:
        with self.assertRaises(TypeError):
            validate_labels("person")

    def test_extract_rejects_empty_text(self) -> None:
        extractor = GlinerNamedEntityExtractor(model=FakeGlinerModel([]))

        with self.assertRaises(ValueError):
            extractor.extract(" ", ["person"])


if __name__ == "__main__":
    unittest.main()
