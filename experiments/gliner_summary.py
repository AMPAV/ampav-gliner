"""Run the preliminary GLiNER Summary probe and retain its native output.

This executable example is intentionally CLI-like while the Phase 1 tool has no
dedicated CLI. Runtime inputs and generated run records remain caller-owned and
should normally live under ``../.work/ampav-gliner/``.
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

from ampav.gliner import GlinerSummarizer


DEFAULT_MODEL_ID = "knowledgator/gliner-multitask-large-v0.5"
DEFAULT_THRESHOLD = 0.25
MODEL_CARD_PROMPT = (
    "Summarize the given text, highlighting the most important information:\n"
)


def parse_args() -> argparse.Namespace:
    """Parse the preliminary probe's command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GLiNER's native summarization APIs on one text input.",
    )
    parser.add_argument("input", type=Path, help="UTF-8 text file below the model limit")
    parser.add_argument("output_dir", type=Path, help="new directory for retained output")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", help="replace GLiNER's native summary prompt")
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        dest="thresholds",
        help="native selection threshold; repeat to compare values (default: 0.25)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--compare-model-card-api",
        action="store_true",
        help="also retain raw predictions from the model card's direct API",
    )
    return parser.parse_args()


def main() -> None:
    """Run the native summarizer and write a human-reviewable run record."""
    args = parse_args()
    thresholds = args.thresholds or [DEFAULT_THRESHOLD]
    text = args.input.read_text(encoding="utf-8").strip()
    summarizer = GlinerSummarizer(
        model_id=args.model_id,
        device=args.device,
        prompt=args.prompt,
    )

    print(f"Loading {args.model_id} on {args.device}", file=sys.stderr, flush=True)
    load_started = perf_counter()
    native_summarizer = summarizer.summarizer
    load_seconds = perf_counter() - load_started
    print(f"Model loaded in {load_seconds:.2f}s", file=sys.stderr, flush=True)

    prompt_text = native_summarizer.prepare_texts([text])[0]
    prepared = native_summarizer.model.prepare_inputs([prompt_text])
    model_word_count = len(prepared[0][0])
    results = []
    for threshold in thresholds:
        print(f"Running threshold={threshold}", file=sys.stderr, flush=True)
        started = perf_counter()
        native_result = summarizer.process(
            text,
            threshold=threshold,
            batch_size=args.batch_size,
        )
        result = {
            "threshold": threshold,
            "high_level_duration_seconds": round(perf_counter() - started, 3),
            "native_high_level_result": native_result,
        }
        if args.compare_model_card_api:
            direct_started = perf_counter()
            result["model_card_direct_predictions"] = (
                native_summarizer.model.predict_entities(
                    MODEL_CARD_PROMPT + text,
                    ["summary"],
                    threshold=threshold,
                )
            )
            result["direct_duration_seconds"] = round(
                perf_counter() - direct_started,
                3,
            )
        results.append(result)

    output = {
        "task": "AMPAV-127",
        "native_package": "gliner",
        "native_version": version("gliner"),
        "model_id": args.model_id,
        "model_max_len": native_summarizer.model.config.max_len,
        "device": native_summarizer.device,
        "prompt": native_summarizer.prompt,
        "model_card_direct_prompt": (
            MODEL_CARD_PROMPT if args.compare_model_card_api else None
        ),
        "batch_size": args.batch_size,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "model_load_seconds": round(load_seconds, 3),
        "input_name": args.input.name,
        "source_word_count": len(text.split()),
        "model_word_count_with_prompt": model_word_count,
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
