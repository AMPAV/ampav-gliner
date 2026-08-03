"""Run the preliminary GLiNER Classification probe and retain native output.

Runtime inputs and generated run records remain caller-owned and should
normally live under ``../.work/ampav-gliner/``.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
import platform
import shlex
import sys
from time import perf_counter

import torch

from ampav.gliner import GlinerTextClassifier


DEFAULT_MODEL_ID = "knowledgator/gliner-multitask-v1.0"
DEFAULT_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    """Parse the preliminary probe's command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GLiNER's native classifier on one text input.",
    )
    parser.add_argument("input", type=Path, help="UTF-8 text file below the model limit")
    parser.add_argument("output_dir", type=Path, help="new directory for retained output")
    parser.add_argument(
        "--class-name",
        action="append",
        required=True,
        dest="classes",
        help="candidate class label; repeat for each class",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", help="replace GLiNER's native prompt template")
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        dest="thresholds",
        help="native selection threshold; repeat to compare values (default: 0.5)",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=("single", "multi"),
        dest="modes",
        help="classification mode; repeat to run both (default: both)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="number of repeated calls per setting for determinism checks",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--retain-direct-predictions",
        action="store_true",
        help="also retain raw predict_entities output for score comparison",
    )
    return parser.parse_args()


def main() -> None:
    """Run the native classifier and write a human-reviewable run record."""
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("repeat must be at least 1")

    thresholds = args.thresholds or [DEFAULT_THRESHOLD]
    modes = args.modes or ["single", "multi"]
    text = args.input.read_text(encoding="utf-8").strip()
    classifier = GlinerTextClassifier(
        model_id=args.model_id,
        device=args.device,
        prompt=args.prompt,
    )

    print(f"Loading {args.model_id} on {args.device}", file=sys.stderr, flush=True)
    load_started = perf_counter()
    native_classifier = classifier.classifier
    load_seconds = perf_counter() - load_started
    print(f"Model loaded in {load_seconds:.2f}s", file=sys.stderr, flush=True)

    prepared_text = native_classifier.prepare_texts([text], args.classes)[0]
    prepared = native_classifier.model.prepare_inputs([prepared_text])
    model_word_count = len(prepared[0][0])
    results = []
    for threshold in thresholds:
        direct_predictions = None
        if args.retain_direct_predictions:
            direct_started = perf_counter()
            direct_predictions = native_classifier.model.predict_entities(
                prepared_text,
                ["match"],
                threshold=threshold,
            )
            direct_duration = round(perf_counter() - direct_started, 3)

        for mode in modes:
            for repetition in range(1, args.repeat + 1):
                print(
                    f"Running threshold={threshold} mode={mode} repeat={repetition}",
                    file=sys.stderr,
                    flush=True,
                )
                started = perf_counter()
                native_result = classifier.process(
                    text,
                    args.classes,
                    multi_label=mode == "multi",
                    threshold=threshold,
                    batch_size=args.batch_size,
                )
                result = {
                    "threshold": threshold,
                    "mode": mode,
                    "repetition": repetition,
                    "duration_seconds": round(perf_counter() - started, 3),
                    "native_high_level_result": native_result,
                }
                if direct_predictions is not None:
                    result["direct_predictions"] = direct_predictions
                    result["direct_duration_seconds"] = direct_duration
                results.append(result)

    output = {
        "task": "AMPAV-128",
        "native_package": "gliner",
        "native_version": version("gliner"),
        "model_id": args.model_id,
        "model_max_len": native_classifier.model.config.max_len,
        "device": native_classifier.device,
        "prompt": native_classifier.prompt,
        "classes": args.classes,
        "batch_size": args.batch_size,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "model_load_seconds": round(load_seconds, 3),
        "input_name": args.input.name,
        "source_word_count": len(text.split()),
        "model_word_count_with_prompt_and_classes": model_word_count,
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "input_ref.txt").write_text(
        f"Input: {args.input}\n",
        encoding="utf-8",
    )
    (args.output_dir / "native_output.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
