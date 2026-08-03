import unittest

from ampav.gliner import GlinerSummarizer


class FakeGlinerModel:
    def __init__(self) -> None:
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
        return [
            [
                {"start": 24, "text": "The event begins Saturday."},
                {"start": 0, "text": "Indiana University hosts the race."},
            ]
        ]


class GlinerSummarizerTest(unittest.TestCase):
    def test_process_uses_native_high_level_api_and_preserves_result(self) -> None:
        model = FakeGlinerModel()
        summarizer = GlinerSummarizer(
            model=model,
            device="cpu",
            prompt="Write an extractive summary:",
        )

        result = summarizer.process(
            "Indiana University hosts the race. The event begins Saturday.",
            threshold=0.1,
            batch_size=4,
        )

        self.assertEqual(
            result,
            ["Indiana University hosts the race. The event begins Saturday."],
        )
        self.assertEqual(model.device, "cpu")
        self.assertEqual(
            model.calls,
            [
                {
                    "texts": [
                        "Write an extractive summary: \n "
                        "Indiana University hosts the race. The event begins Saturday."
                    ],
                    "labels": ["summary"],
                    "threshold": 0.1,
                    "batch_size": 4,
                }
            ],
        )

    def test_process_reuses_constructed_native_summarizer(self) -> None:
        model = FakeGlinerModel()
        summarizer = GlinerSummarizer(model=model)

        summarizer.process("First input.")
        native_summarizer = summarizer.summarizer
        summarizer.process("Second input.")

        self.assertIs(summarizer.summarizer, native_summarizer)
        self.assertEqual(len(model.calls), 2)

    def test_process_rejects_empty_text_before_loading_model(self) -> None:
        summarizer = GlinerSummarizer()

        with self.assertRaises(ValueError):
            summarizer.process("  ")

        self.assertIsNone(summarizer._summarizer)

    def test_process_rejects_non_string_text(self) -> None:
        summarizer = GlinerSummarizer()

        with self.assertRaises(TypeError):
            summarizer.process(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
