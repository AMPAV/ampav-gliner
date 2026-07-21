"""One-shot command-line client for GLiNER named entities."""

import argparse
from collections.abc import Sequence
import logging

from ampav.core.logging import LOG_FORMAT
from ampav.gliner import DEFAULT_MODEL_ID, GlinerNamedEntityExtractor
from ampav_gliner_pipeline import extract_named_entities_from_file


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the GLiNER named-entity CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract named entities from a UTF-8 text file with GLiNER and "
            "print AMPAV ToolOutput YAML."
        )
    )
    parser.add_argument("text_file", help="Path to a UTF-8 text file")
    parser.add_argument(
        "labels",
        nargs="+",
        help="One or more entity labels, such as person organization location",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="GLiNER model ID or local path",
    )
    parser.add_argument("--cache-dir", help="Optional model cache directory")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only cached model artifacts",
    )
    parser.add_argument("--revision", help="Optional model revision, branch, or commit")
    parser.add_argument(
        "--map-location",
        default="cpu",
        help="Device for model weights (default: cpu)",
    )
    parser.add_argument("--threshold", type=float, help="Optional entity confidence threshold")
    parser.add_argument("--nested-ner", action="store_true", help="Allow nested entity spans")
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="Allow multiple labels for one span",
    )
    parser.add_argument("--batch-size", type=int, help="Optional GLiNER inference batch size")
    parser.add_argument("--language", help="Optional source language assigned to returned entities")
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=0,
        help="Model-token context to add on each side of long-text chunks",
    )
    parser.add_argument(
        "--include-tool-private",
        action="store_true",
        help="Include raw per-chunk GLiNER predictions in ToolOutput.tool_private",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one GLiNER extraction and print the resulting ToolOutput YAML."""
    args = build_cli_parser().parse_args(argv)
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    extractor = GlinerNamedEntityExtractor(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        revision=args.revision,
        map_location=args.map_location,
        include_tool_private=args.include_tool_private,
    )
    try:
        result = extract_named_entities_from_file(
            args.text_file,
            args.labels,
            extractor=extractor,
            threshold=args.threshold,
            flat_ner=not args.nested_ner,
            multi_label=args.multi_label,
            batch_size=args.batch_size,
            language=args.language,
            chunk_overlap_tokens=args.chunk_overlap_tokens,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    print(result.model_dump_yaml(sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
