import unittest

from ampav.gliner import GlinerTextClassifier


class FakeGlinerModel:
    def __init__(self, predictions: list[list[dict]]) -> None:
        self.predictions = predictions
        self.device: str | None = None
        self.calls: list[dict] = []

    def to(self, device: str) -> "FakeGlinerModel":
        self.device = device
        return self

    def inference(
        self,
        texts: list[str],
        labels: list[str],
        *,
        threshold: float,
        batch_size: int,
    ) -> list[list[dict]]:
        self.calls.append(
            {
                "texts": texts,
                "labels": labels,
                "threshold": threshold,
                "batch_size": batch_size,
            }
        )
        return self.predictions


class GlinerTextClassifierTest(unittest.TestCase):
    def test_single_label_uses_native_prompt_and_softmax_result(self) -> None:
        model = FakeGlinerModel(
            [[
                {"text": "technology", "score": 0.8},
                {"text": "sports", "score": 0.2},
            ]]
        )
        classifier = GlinerTextClassifier(
            model=model,
            device="cpu",
            prompt="Choose one of these topics: {}",
        )

        result = classifier.process(
            "AMPAV applies AI to audiovisual metadata.",
            ("technology", "sports"),
            threshold=0.1,
            batch_size=4,
        )

        self.assertEqual(result[0][0]["label"], "technology")
        self.assertAlmostEqual(result[0][0]["score"], 0.6456563)
        self.assertEqual(model.device, "cpu")
        self.assertEqual(
            model.calls,
            [{
                "texts": [
                    "Choose one of these topics: technology, sports \n "
                    "AMPAV applies AI to audiovisual metadata."
                ],
                "labels": ["match"],
                "threshold": 0.1,
                "batch_size": 4,
            }],
        )

    def test_multi_label_preserves_native_scores_in_descending_order(self) -> None:
        model = FakeGlinerModel(
            [[
                {"text": "archives", "score": 0.4},
                {"text": "technology", "score": 0.9},
            ]]
        )
        classifier = GlinerTextClassifier(model=model)

        result = classifier.process(
            "AI tools help audiovisual archives.",
            ["technology", "archives"],
            multi_label=True,
        )

        self.assertEqual(
            result,
            [[
                {"label": "technology", "score": 0.9},
                {"label": "archives", "score": 0.4},
            ]],
        )

    def test_empty_native_prediction_preserves_synthetic_other(self) -> None:
        classifier = GlinerTextClassifier(model=FakeGlinerModel([[]]))

        result = classifier.process("Ambiguous input.", ["one", "two"])

        self.assertEqual(result, [[{"label": "other", "score": 1.0}]])

    def test_process_reuses_constructed_native_classifier(self) -> None:
        model = FakeGlinerModel([[{"text": "one", "score": 0.8}]])
        classifier = GlinerTextClassifier(model=model)

        classifier.process("First input.", ["one"])
        native_classifier = classifier.classifier
        classifier.process("Second input.", ["one"])

        self.assertIs(classifier.classifier, native_classifier)
        self.assertEqual(len(model.calls), 2)

    def test_process_validates_inputs_before_loading_model(self) -> None:
        classifier = GlinerTextClassifier()

        invalid_calls = [
            (" ", ["one"], ValueError),
            (None, ["one"], TypeError),
            ("text", [], ValueError),
            ("text", "one", TypeError),
            ("text", ["one", 2], TypeError),
            ("text", ["one", " "], ValueError),
        ]
        for text, classes, error in invalid_calls:
            with self.subTest(text=text, classes=classes):
                with self.assertRaises(error):
                    classifier.process(text, classes)  # type: ignore[arg-type]

        self.assertIsNone(classifier._classifier)

    def test_constructor_rejects_prompt_without_class_placeholder(self) -> None:
        with self.assertRaises(ValueError):
            GlinerTextClassifier(prompt="Classify this text")


if __name__ == "__main__":
    unittest.main()
