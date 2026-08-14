"""Probe classic GLiNER's preliminary native keyphrase behavior.

The executable keeps caller-owned inputs and retained run records outside the
library package. It supports small quality and global-relevance matrices plus a
model-specific input-boundary probe without defining an AMPAV schema or
production API.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shlex
import sys
from time import perf_counter
from typing import Any, Callable
import warnings

import torch
import yaml


DEFAULT_MODEL_ID = "knowledgator/gliner-multitask-v1.0"
DEFAULT_MODEL_REVISION = "0f31be112e077396cc8fef5598f5ae6fdb8fec17"
DEFAULT_COMPARISON_MODEL_ID = "knowledgator/gliner-multitask-large-v0.5"
DEFAULT_COMPARISON_MODEL_REVISION = "7a95e168036db9ec6f914c0cc6b218edbd87f310"
DEFAULT_PROMPT = "Extract important, content-bearing key phrases:"
DEFAULT_RELEVANCE_PROMPT = (
    "Extract phrases that best represent the main subjects of the entire text; "
    "exclude generic noun phrases, pronouns, and incidental details."
)
DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following text highlighting the most important information:"
)
DEFAULT_BOUNDARY_PROMPT = (
    "Extract important, content-bearing key phrases that identify the central "
    "subjects, methods, events, and concepts in the following text:"
)
DEFAULT_THRESHOLDS = [0.1, 0.3, 0.5]
DEFAULT_LABEL_VARIANTS = ["key phrase", "keyphrase", "topic", "concept"]
DEFAULT_RELEVANCE_LABELS = [
    "key phrase",
    "key concept",
    "important topic",
    "subject term",
]
EARLY_SENTINEL = "silver metadata compass"
END_SENTINEL = "crimson archive beacon"


def parse_args() -> argparse.Namespace:
    """Parse quality, relevance, or boundary probe arguments."""
    parser = argparse.ArgumentParser(
        description="Run preliminary native GLiNER keyphrase probes.",
    )
    subparsers = parser.add_subparsers(dest="probe", required=True)

    quality = subparsers.add_parser("quality", help="run the native quality matrix")
    quality.add_argument("input", type=Path, help="UTF-8 source text")
    quality.add_argument("output_dir", type=Path, help="new retained-run directory")
    _add_model_arguments(quality)
    quality.add_argument("--prompt", default=DEFAULT_PROMPT)
    quality.add_argument(
        "--threshold",
        action="append",
        type=float,
        dest="thresholds",
        help="prompt-path threshold; repeat to compare values",
    )
    quality.add_argument(
        "--open-threshold",
        action="append",
        type=float,
        dest="open_thresholds",
        help="GLiNEROpenExtractor parity threshold; default: 0.3",
    )
    quality.add_argument(
        "--label-variant",
        action="append",
        dest="label_variants",
        help="unprompted label-driven contrast; repeat to compare labels",
    )
    quality.add_argument("--label-threshold", type=float, default=0.3)
    quality.add_argument("--repeat", type=int, default=1)

    relevance = subparsers.add_parser(
        "relevance",
        help="compare global-relevance prompts and labels across two models",
    )
    relevance.add_argument("input", type=Path, help="UTF-8 source text")
    relevance.add_argument("output_dir", type=Path, help="new retained-run directory")
    _add_model_arguments(relevance)
    relevance.add_argument(
        "--comparison-model-id",
        default=DEFAULT_COMPARISON_MODEL_ID,
    )
    relevance.add_argument(
        "--comparison-model-revision",
        default=DEFAULT_COMPARISON_MODEL_REVISION,
    )
    relevance.add_argument("--prompt", default=DEFAULT_RELEVANCE_PROMPT)
    relevance.add_argument("--threshold", type=float, default=0.25)

    boundary = subparsers.add_parser(
        "boundary",
        help="probe native word and transformer input boundaries",
    )
    boundary.add_argument("output_dir", type=Path, help="new retained-run directory")
    _add_model_arguments(boundary)
    boundary.add_argument("--prompt", default=DEFAULT_BOUNDARY_PROMPT)
    boundary.add_argument("--threshold", type=float, default=0.1)

    return parser.parse_args()


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common native-model settings to a subcommand parser."""
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face access instead of requiring cached model files",
    )


def _load_model(args: argparse.Namespace) -> Any:
    """Load the selected classic GLiNER model."""
    return _load_model_settings(
        args.model_id,
        args.model_revision,
        args.device,
        args.allow_download,
    )


def _load_model_settings(
    model_id: str,
    model_revision: str,
    device: str,
    allow_download: bool,
) -> Any:
    """Load one explicitly selected classic GLiNER model."""
    from gliner import GLiNER

    model = GLiNER.from_pretrained(
        model_id,
        revision=model_revision,
        local_files_only=not allow_download,
    )
    return model.to(device)


def _capture_call(call: Callable[[], Any]) -> tuple[Any, float, list[str]]:
    """Run one native operation and retain duration and emitted warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = perf_counter()
        result = call()
        duration = perf_counter() - started
    return result, duration, [str(item.message) for item in caught]


def _prompted_text(prompt: str, source: str) -> tuple[str, int]:
    """Prepare exactly the text produced by GLiNEROpenExtractor 0.2.28."""
    prefix = f"{prompt} \n "
    return prefix + source, len(prefix)


def _word_tokens(model: Any, text: str) -> list[str]:
    """Return GLiNER's first-stage words for one text."""
    return list(model.prepare_inputs([text])[0][0])


def _contains_subsequence(sequence: list[int], candidate: list[int]) -> bool:
    """Return whether one token-id sequence contains another contiguously."""
    if not candidate:
        return True
    width = len(candidate)
    return any(sequence[index : index + width] == candidate for index in range(len(sequence) - width + 1))


def _analyze_predictions(
    predictions: list[dict[str, Any]],
    prepared_text: str,
    source_start: int,
) -> list[dict[str, Any]]:
    """Describe native prediction grounding without changing the result."""
    analysis = []
    for prediction in predictions:
        start = prediction.get("start")
        end = prediction.get("end")
        text = prediction.get("text")
        valid_offsets = isinstance(start, int) and isinstance(end, int)
        slice_matches = (
            valid_offsets
            and 0 <= start <= end <= len(prepared_text)
            and prepared_text[start:end] == text
        )
        if not valid_offsets:
            region = "unknown"
            source_offsets = None
        elif end <= source_start:
            region = "prompt"
            source_offsets = None
        elif start >= source_start:
            region = "source"
            source_offsets = {"start": start - source_start, "end": end - source_start}
        else:
            region = "prompt_source_boundary"
            source_offsets = None
        analysis.append(
            {
                "native_prediction_index": len(analysis),
                "slice_matches_prepared_text": slice_matches,
                "region": region,
                "source_offsets": source_offsets,
            }
        )
    return analysis


def _measure_model_input(model: Any, text: str, labels: list[str]) -> dict[str, Any]:
    """Measure word and transformer inputs before and after native truncation."""
    source_words = _word_tokens(model, text)
    processor = model.data_processor
    combined_words, _ = processor.prepare_inputs([source_words], labels)
    tokenizer = processor.transformer_tokenizer

    before_encoding, before_seconds, before_warnings = _capture_call(
        lambda: tokenizer(
            combined_words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=False,
            padding="longest",
        )
    )
    prepared = model.prepare_batch([text], labels)
    native_batch, native_seconds, native_warnings = _capture_call(
        lambda: model.collate_batch(prepared["input_x"], prepared["entity_types"])
    )

    before_ids = before_encoding["input_ids"][0].tolist()
    after_ids = native_batch["input_ids"][0].tolist()
    before_length = int(before_encoding["attention_mask"][0].sum().item())
    after_length = int(native_batch["attention_mask"][0].sum().item())
    retained_words = list(native_batch["tokens"][0])

    def sentinel_metrics(sentinel: str) -> dict[str, bool]:
        sentinel_words = _word_tokens(model, sentinel)
        sentinel_ids = tokenizer(sentinel, add_special_tokens=False)["input_ids"]
        return {
            "in_source_words": _contains_words(source_words, sentinel_words),
            "in_native_retained_words": _contains_words(retained_words, sentinel_words),
            "in_combined_transformer_before": _contains_subsequence(before_ids, sentinel_ids),
            "in_combined_transformer_after": _contains_subsequence(after_ids, sentinel_ids),
        }

    return {
        "inference_text_gliner_words": len(source_words),
        "combined_gliner_words_before_native_limit": len(combined_words[0]),
        "native_retained_source_words": len(retained_words),
        "combined_transformer_tokens_before_native_truncation": before_length,
        "combined_transformer_tokens_after_native_truncation": after_length,
        "transformer_model_max_length": tokenizer.model_max_length,
        "tokenization_duration_seconds": round(before_seconds, 3),
        "native_collation_duration_seconds": round(native_seconds, 3),
        "warnings": before_warnings + native_warnings,
        "early_sentinel": sentinel_metrics(EARLY_SENTINEL),
        "end_sentinel": sentinel_metrics(END_SENTINEL),
    }


def _contains_words(words: list[str], candidate: list[str]) -> bool:
    """Return whether one GLiNER word sequence contains another."""
    width = len(candidate)
    return width == 0 or any(
        words[index : index + width] == candidate
        for index in range(len(words) - width + 1)
    )


def _build_boundary_source(model: Any, target_count: int, prompt: str | None) -> tuple[str, str, int]:
    """Build deterministic source whose prepared text has an exact word count."""
    prefix = f"important early key phrase {EARLY_SENTINEL}"
    suffix = f"critical final key phrase {END_SENTINEL}"

    def prepare(source: str) -> tuple[str, int]:
        return _prompted_text(prompt, source) if prompt is not None else (source, 0)

    source = f"{prefix} {suffix}"
    prepared_text, source_start = prepare(source)
    base_count = len(_word_tokens(model, prepared_text))
    if base_count > target_count:
        raise ValueError(f"boundary template needs {base_count} words, above target {target_count}")

    fillers = [f"background{index:05d}" for index in range(1, target_count - base_count + 1)]
    source = " ".join([prefix, *fillers, suffix])
    prepared_text, source_start = prepare(source)
    actual_count = len(_word_tokens(model, prepared_text))
    if actual_count != target_count:
        raise RuntimeError(
            f"could not construct exact GLiNER boundary input: target={target_count}, actual={actual_count}"
        )
    return source, prepared_text, source_start


def _run_quality(args: argparse.Namespace, model: Any) -> dict[str, Any]:
    """Run the approved prompt and label contrast matrix."""
    from gliner.multitask import GLiNEROpenExtractor

    if args.repeat < 1:
        raise ValueError("repeat must be at least 1")
    source = args.input.read_text(encoding="utf-8").strip()
    if not source:
        raise ValueError("input text must not be empty")

    thresholds = args.thresholds or DEFAULT_THRESHOLDS
    open_thresholds = args.open_thresholds or [0.3]
    label_variants = args.label_variants or DEFAULT_LABEL_VARIANTS
    prepared_text, source_start = _prompted_text(args.prompt, source)
    open_extractor = GLiNEROpenExtractor(model=model, device=args.device, prompt=args.prompt)

    prompt_direct = []
    direct_by_setting: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for threshold in thresholds:
        for repetition in range(1, args.repeat + 1):
            predictions, duration, emitted = _capture_call(
                lambda threshold=threshold: model.predict_entities(
                    prepared_text,
                    ["match"],
                    threshold=threshold,
                    batch_size=args.batch_size,
                )
            )
            direct_by_setting[(threshold, repetition)] = predictions
            prompt_direct.append(
                {
                    "threshold": threshold,
                    "repetition": repetition,
                    "duration_seconds": round(duration, 3),
                    "warnings": emitted,
                    "native_predictions": predictions,
                    "prediction_analysis": _analyze_predictions(
                        predictions, prepared_text, source_start
                    ),
                }
            )

    open_results = []
    for threshold in open_thresholds:
        for repetition in range(1, args.repeat + 1):
            batch_predictions, duration, emitted = _capture_call(
                lambda threshold=threshold: open_extractor(
                    source,
                    labels=["match"],
                    threshold=threshold,
                    batch_size=args.batch_size,
                )
            )
            predictions = batch_predictions[0]
            direct_predictions = direct_by_setting.get((threshold, repetition))
            if direct_predictions is None:
                direct_predictions, _, _ = _capture_call(
                    lambda threshold=threshold: model.predict_entities(
                        prepared_text,
                        ["match"],
                        threshold=threshold,
                        batch_size=args.batch_size,
                    )
                )
            open_results.append(
                {
                    "threshold": threshold,
                    "repetition": repetition,
                    "duration_seconds": round(duration, 3),
                    "warnings": emitted,
                    "matches_equivalent_direct_call": predictions == direct_predictions,
                    "native_predictions": predictions,
                    "prediction_analysis": _analyze_predictions(
                        predictions, prepared_text, source_start
                    ),
                }
            )

    label_results = []
    for label in label_variants:
        for repetition in range(1, args.repeat + 1):
            predictions, duration, emitted = _capture_call(
                lambda label=label: model.predict_entities(
                    source,
                    [label],
                    threshold=args.label_threshold,
                    batch_size=args.batch_size,
                )
            )
            label_results.append(
                {
                    "label": label,
                    "threshold": args.label_threshold,
                    "repetition": repetition,
                    "duration_seconds": round(duration, 3),
                    "warnings": emitted,
                    "native_predictions": predictions,
                    "prediction_analysis": _analyze_predictions(predictions, source, 0),
                }
            )

    return {
        "input_name": args.input.name,
        "source_character_count": len(source),
        "source_gliner_word_count": len(_word_tokens(model, source)),
        "prompt": args.prompt,
        "prepared_text": prepared_text,
        "source_start": source_start,
        "prepared_gliner_word_count": len(_word_tokens(model, prepared_text)),
        "prompt_direct_match": prompt_direct,
        "open_extractor_match": open_results,
        "unprompted_label_contrasts": label_results,
    }


def _run_prediction_case(
    model: Any,
    source: str,
    *,
    prompt: str | None,
    label: str,
    threshold: float,
    batch_size: int,
) -> dict[str, Any]:
    """Run one direct native call and retain prompt-relative diagnostics."""
    if prompt is None:
        prepared_text = source
        source_start = 0
    else:
        prepared_text, source_start = _prompted_text(prompt, source)
    predictions, duration, emitted = _capture_call(
        lambda: model.predict_entities(
            prepared_text,
            [label],
            threshold=threshold,
            batch_size=batch_size,
        )
    )
    return {
        "prompt": prompt,
        "label": label,
        "threshold": threshold,
        "source_start": source_start,
        "prepared_gliner_word_count": len(_word_tokens(model, prepared_text)),
        "duration_seconds": round(duration, 3),
        "warnings": emitted,
        "native_predictions": predictions,
        "prediction_analysis": _analyze_predictions(
            predictions,
            prepared_text,
            source_start,
        ),
    }


def _run_relevance_model(
    model: Any,
    source: str,
    *,
    model_id: str,
    model_revision: str,
    prompt: str,
    labels: list[str],
    threshold: float,
    batch_size: int,
) -> dict[str, Any]:
    """Run the approved label-by-prompt matrix for one model."""
    cases = []
    for condition, active_prompt in (
        ("unprompted", None),
        ("global_relevance_prompt", prompt),
    ):
        for label in labels:
            print(
                f"Running {model_id}: {condition}, label={label!r}",
                file=sys.stderr,
                flush=True,
            )
            case = _run_prediction_case(
                model,
                source,
                prompt=active_prompt,
                label=label,
                threshold=threshold,
                batch_size=batch_size,
            )
            case["condition"] = condition
            cases.append(case)
    return {
        "model_id": model_id,
        "model_revision": model_revision,
        "model_max_len": int(model.config.max_len),
        "source_gliner_word_count": len(_word_tokens(model, source)),
        "cases": cases,
    }


def _run_relevance(
    args: argparse.Namespace,
    primary_model: Any,
    comparison_model: Any,
) -> dict[str, Any]:
    """Run the approved final global-relevance experiment matrix."""
    source = args.input.read_text(encoding="utf-8").strip()
    if not source:
        raise ValueError("input text must not be empty")

    primary = _run_relevance_model(
        primary_model,
        source,
        model_id=args.model_id,
        model_revision=args.model_revision,
        prompt=args.prompt,
        labels=DEFAULT_RELEVANCE_LABELS,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    comparison = _run_relevance_model(
        comparison_model,
        source,
        model_id=args.comparison_model_id,
        model_revision=args.comparison_model_revision,
        prompt=args.prompt,
        labels=DEFAULT_RELEVANCE_LABELS,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    print(
        f"Running {args.model_id}: summary control",
        file=sys.stderr,
        flush=True,
    )
    summary_control = _run_prediction_case(
        primary_model,
        source,
        prompt=DEFAULT_SUMMARY_PROMPT,
        label="summary",
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    summary_control["condition"] = "summary_prompt_control"
    return {
        "input_name": args.input.name,
        "source_character_count": len(source),
        "global_relevance_prompt": args.prompt,
        "labels": DEFAULT_RELEVANCE_LABELS,
        "threshold": args.threshold,
        "models": [primary, comparison],
        "primary_model_summary_control": summary_control,
    }


def _run_boundary(args: argparse.Namespace, model: Any, output_dir: Path) -> dict[str, Any]:
    """Run focused first-stage and transformer boundary cases."""
    max_len = int(model.config.max_len)
    cases = []
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir()

    settings = [
        ("unprompted_key_phrase", None, ["key phrase"]),
        ("prompted_key_phrase", args.prompt, ["key phrase"]),
        ("prompted_match", args.prompt, ["match"]),
    ]
    for setting_name, prompt, labels in settings:
        for delta in (-1, 0, 1):
            target = max_len + delta
            source, prepared_text, source_start = _build_boundary_source(
                model, target, prompt
            )
            input_name = f"{setting_name}-prepared_{target:04d}_gliner_words.txt"
            (inputs_dir / input_name).write_text(source + "\n", encoding="utf-8")
            measurements = _measure_model_input(model, prepared_text, labels)
            predictions, duration, emitted = _capture_call(
                lambda prepared_text=prepared_text, labels=labels: model.predict_entities(
                    prepared_text,
                    labels,
                    threshold=args.threshold,
                    batch_size=args.batch_size,
                )
            )
            predicted_text = " ".join(
                str(prediction.get("text", "")) for prediction in predictions
            ).casefold()
            cases.append(
                {
                    "setting": setting_name,
                    "target_prepared_gliner_words": target,
                    "delta_from_max_len": delta,
                    "input_file": f"inputs/{input_name}",
                    "prompt": prompt,
                    "labels": labels,
                    "source_start": source_start,
                    "source_character_count": len(source),
                    "source_gliner_word_count": len(_word_tokens(model, source)),
                    "prepared_character_count": len(prepared_text),
                    "prepared_gliner_word_count": len(
                        _word_tokens(model, prepared_text)
                    ),
                    "measurements": measurements,
                    "inference_duration_seconds": round(duration, 3),
                    "inference_warnings": emitted,
                    "native_predictions": predictions,
                    "prediction_analysis": _analyze_predictions(
                        predictions, prepared_text, source_start
                    ),
                    "early_sentinel_extracted": EARLY_SENTINEL in predicted_text,
                    "end_sentinel_extracted": END_SENTINEL in predicted_text,
                }
            )
    return {
        "model_max_len": max_len,
        "longest_realistic_prompt": args.prompt,
        "threshold": args.threshold,
        "early_sentinel": EARLY_SENTINEL,
        "end_sentinel": END_SENTINEL,
        "cases": cases,
    }


def _write_run_record(
    args: argparse.Namespace,
    output_dir: Path,
    started_at: str,
    load_seconds: float | dict[str, float],
    result: dict[str, Any],
) -> None:
    """Write command, manifest, native JSON, and concise run notes."""
    manifest = {
        "task": "AMPAV-105",
        "probe": args.probe,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(),
        "native_package": "gliner",
        "native_version": version("gliner"),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "device": args.device,
        "batch_size": args.batch_size,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "model_load_seconds": (
            round(load_seconds, 3)
            if isinstance(load_seconds, float)
            else {key: round(value, 3) for key, value in load_seconds.items()}
        ),
    }
    if args.probe == "relevance":
        manifest["comparison_model_id"] = args.comparison_model_id
        manifest["comparison_model_revision"] = args.comparison_model_revision
    (output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    input_description = (
        str(args.input.resolve())
        if args.probe in {"quality", "relevance"}
        else "generated boundary inputs under inputs/"
    )
    (output_dir / "input_ref.txt").write_text(
        f"Input: {input_description}\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "native_output.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "observations.md").write_text(
        "# Observations\n\nNative output retained for human review.\n",
        encoding="utf-8",
    )


def main() -> None:
    """Load cached model(s) and retain the selected native probe."""
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started_at = datetime.now().astimezone().isoformat()

    print(f"Loading {args.model_id} on {args.device}", file=sys.stderr, flush=True)
    load_started = perf_counter()
    model = _load_model(args)
    load_seconds = perf_counter() - load_started
    print(f"Model loaded in {load_seconds:.2f}s", file=sys.stderr, flush=True)

    if args.probe == "relevance":
        print(
            f"Loading {args.comparison_model_id} on {args.device}",
            file=sys.stderr,
            flush=True,
        )
        comparison_load_started = perf_counter()
        comparison_model = _load_model_settings(
            args.comparison_model_id,
            args.comparison_model_revision,
            args.device,
            args.allow_download,
        )
        comparison_load_seconds = perf_counter() - comparison_load_started
        print(
            f"Comparison model loaded in {comparison_load_seconds:.2f}s",
            file=sys.stderr,
            flush=True,
        )
        result = _run_relevance(args, model, comparison_model)
        load_seconds = {
            args.model_id: load_seconds,
            args.comparison_model_id: comparison_load_seconds,
        }
    elif args.probe == "quality":
        result = _run_quality(args, model)
    else:
        result = _run_boundary(args, model, args.output_dir)
    _write_run_record(args, args.output_dir, started_at, load_seconds, result)
    print(f"Retained run: {args.output_dir}")


if __name__ == "__main__":
    main()
