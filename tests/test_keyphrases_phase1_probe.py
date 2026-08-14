"""Focused tests for the preliminary GLiNER keyphrase probe."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import importlib.util
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import yaml


PROBE_PATH = Path(__file__).parents[1] / "examples" / "gliner_keyphrases_phase1.py"
SPEC = importlib.util.spec_from_file_location("gliner_keyphrases_phase1", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load probe module from {PROBE_PATH}")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)

END_SENTINEL = PROBE.END_SENTINEL
_analyze_predictions = PROBE._analyze_predictions
_build_boundary_source = PROBE._build_boundary_source
_contains_subsequence = PROBE._contains_subsequence
_prompted_text = PROBE._prompted_text
_run_relevance_model = PROBE._run_relevance_model
_write_run_record = PROBE._write_run_record


class FakeWhitespaceModel:
    """Expose the GLiNER ``prepare_inputs`` shape with whitespace words."""

    def prepare_inputs(self, texts: list[str]) -> tuple[list[list[str]], list[list[int]], list[list[int]]]:
        tokens = [text.split() for text in texts]
        starts = [list(range(len(item))) for item in tokens]
        ends = [list(range(1, len(item) + 1)) for item in tokens]
        return tokens, starts, ends


class FakePredictionModel(FakeWhitespaceModel):
    """Retain direct native calls for relevance-matrix assertions."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(max_len=128)
        self.calls: list[dict[str, object]] = []

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        **kwargs: object,
    ) -> list[dict[str, object]]:
        self.calls.append({"text": text, "labels": labels, "kwargs": kwargs})
        start = text.index("Source")
        return [
            {
                "text": "Source",
                "label": labels[0],
                "start": start,
                "end": start + len("Source"),
            }
        ]


class GlinerKeyphraseProbeTest(unittest.TestCase):
    def test_prompted_text_matches_open_extractor_format(self) -> None:
        prepared, source_start = _prompted_text("Extract phrases:", "Source text")

        self.assertEqual(prepared, "Extract phrases: \n Source text")
        self.assertEqual(prepared[source_start:], "Source text")

    def test_prediction_analysis_identifies_prompt_and_source_regions(self) -> None:
        prepared, source_start = _prompted_text("Extract phrases:", "Source text")
        predictions = [
            {"text": "phrases", "start": 8, "end": 15},
            {
                "text": "Source",
                "start": source_start,
                "end": source_start + len("Source"),
            },
        ]

        analysis = _analyze_predictions(predictions, prepared, source_start)

        self.assertEqual(analysis[0]["region"], "prompt")
        self.assertEqual(analysis[1]["region"], "source")
        self.assertEqual(analysis[1]["source_offsets"], {"start": 0, "end": 6})
        self.assertTrue(all(item["slice_matches_prepared_text"] for item in analysis))

    def test_boundary_builder_hits_exact_raw_and_prompted_counts(self) -> None:
        model = FakeWhitespaceModel()

        source, prepared, source_start = _build_boundary_source(model, 64, None)
        prompted_source, prompted, prompted_start = _build_boundary_source(
            model,
            64,
            "Extract important phrases:",
        )

        self.assertEqual(len(prepared.split()), 64)
        self.assertEqual(source_start, 0)
        self.assertIn(END_SENTINEL, source)
        self.assertEqual(len(prompted.split()), 64)
        self.assertEqual(prompted[prompted_start:], prompted_source)
        self.assertIn(END_SENTINEL, prompted_source)

    def test_contains_subsequence_requires_contiguous_order(self) -> None:
        self.assertTrue(_contains_subsequence([1, 2, 3, 4], [2, 3]))
        self.assertFalse(_contains_subsequence([1, 2, 3, 4], [2, 4]))

    def test_relevance_model_crosses_labels_with_prompt_conditions(self) -> None:
        model = FakePredictionModel()

        with redirect_stderr(StringIO()):
            result = _run_relevance_model(
                model,
                "Source text",
                model_id="test/model",
                model_revision="abc123",
                prompt="Extract globally relevant phrases.",
                labels=["key phrase", "subject term"],
                threshold=0.25,
                batch_size=4,
            )

        self.assertEqual(len(result["cases"]), 4)
        self.assertEqual(
            [case["condition"] for case in result["cases"]],
            [
                "unprompted",
                "unprompted",
                "global_relevance_prompt",
                "global_relevance_prompt",
            ],
        )
        self.assertEqual(model.calls[0]["text"], "Source text")
        self.assertTrue(
            str(model.calls[-1]["text"]).startswith(
                "Extract globally relevant phrases."
            )
        )
        self.assertTrue(
            all(
                analysis["slice_matches_prepared_text"]
                for case in result["cases"]
                for analysis in case["prediction_analysis"]
            )
        )

    def test_run_record_serializes_runtime_versions(self) -> None:
        args = argparse.Namespace(
            probe="boundary",
            model_id="test/model",
            model_revision="abc123",
            device="cpu",
            batch_size=1,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            _write_run_record(
                args,
                output_dir,
                "2026-08-04T12:00:00-04:00",
                1.25,
                {"native": "result"},
            )

            manifest = yaml.safe_load(
                (output_dir / "manifest.yaml").read_text(encoding="utf-8")
            )
            self.assertIsInstance(manifest["torch_version"], str)
            self.assertEqual(manifest["model_revision"], "abc123")


if __name__ == "__main__":
    unittest.main()
