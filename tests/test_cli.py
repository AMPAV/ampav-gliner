import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from ampav.core.schema import NamedEntities, ToolOutput
from ampav_gliner_cli import entities as cli


class FakeExtractor:
    instances: list["FakeExtractor"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.process_calls: list[dict] = []
        FakeExtractor.instances.append(self)

    def process(self, text: str, labels: list[str], **kwargs: object) -> ToolOutput:
        self.process_calls.append({"text": text, "labels": labels, **kwargs})
        return ToolOutput(
            tool_name="gliner",
            output=NamedEntities(text=text),
        )


class GlinerCliTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeExtractor.instances = []

    def test_parser_accepts_model_and_inference_settings(self) -> None:
        args = cli.build_cli_parser().parse_args(
            [
                "input.txt",
                "person",
                "organization",
                "--model-id",
                "local/model",
                "--threshold",
                "0.4",
                "--nested-ner",
                "--multi-label",
                "--chunk-overlap-tokens",
                "16",
                "--include-tool-private",
            ]
        )

        self.assertEqual(args.text_file, "input.txt")
        self.assertEqual(args.labels, ["person", "organization"])
        self.assertEqual(args.model_id, "local/model")
        self.assertEqual(args.threshold, 0.4)
        self.assertTrue(args.nested_ner)
        self.assertTrue(args.multi_label)
        self.assertEqual(args.chunk_overlap_tokens, 16)
        self.assertTrue(args.include_tool_private)

    def test_main_reads_file_and_prints_tool_output_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "input.txt"
            text_path.write_text("Maya Chen visited Bloomington.", encoding="utf-8")

            original = cli.GlinerNamedEntityExtractor
            cli.GlinerNamedEntityExtractor = FakeExtractor
            try:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli.main(
                        [
                            str(text_path),
                            "person",
                            "location",
                            "--threshold",
                            "0.4",
                            "--language",
                            "en",
                        ]
                    )
            finally:
                cli.GlinerNamedEntityExtractor = original

        self.assertEqual(exit_code, 0)
        self.assertIn("tool_name: gliner", stdout.getvalue())
        instance = FakeExtractor.instances[0]
        self.assertFalse(instance.kwargs["include_tool_private"])
        self.assertEqual(
            instance.process_calls[0],
            {
                "text": "Maya Chen visited Bloomington.",
                "labels": ["person", "location"],
                "threshold": 0.4,
                "flat_ner": True,
                "multi_label": False,
                "batch_size": None,
                "language": "en",
                "chunk_overlap_tokens": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
