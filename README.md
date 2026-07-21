# ampav-gliner

Synchronous GLiNER named-entity extraction for AMPAV. The tool accepts plain
text and returns a `ToolOutput` whose output is `NamedEntities`. Text longer
than the loaded model's token limit is chunked and reassembled with character
offsets against the complete original text.

The default CPU model is `urchade/gliner_small-v2.1`. Its files are downloaded
through the upstream GLiNER/Hugging Face stack on first use. Set
`local_files_only=True` when the required artifacts are already cached and
network access must be disabled.

## Plain text

```python
from ampav.gliner import GlinerNamedEntityExtractor

extractor = GlinerNamedEntityExtractor()
result = extractor.process(
    "Maya Chen visited Indiana University.",
    ["person", "organization"],
    threshold=0.4,
)
print(result.model_dump_yaml(sort_keys=False))
```

`process(...)` uses the loaded model's `config.max_len` automatically. Set
`chunk_overlap_tokens` when entity detection needs context around long-text
chunk boundaries. Raw per-chunk model predictions are omitted by default; use
`include_tool_private=True` on the extractor only for troubleshooting.

## Transcript and file adapters

Input-specific helpers live outside the low-level tool package:

```python
from ampav.gliner import GlinerNamedEntityExtractor
from ampav_gliner_pipeline import extract_named_entities_from_transcript

result = extract_named_entities_from_transcript(
    transcript,
    ["person", "organization"],
    extractor=GlinerNamedEntityExtractor(),
)
```

The transcript adapter builds canonical text and chunk units directly from
`Transcript.words`, reassembles all entity offsets, and then aligns entity
timestamps once. `extract_named_entities_from_file(...)` reads a caller-owned
UTF-8 text file without modifying or deleting it.

See `examples/gliner_text_example.py` and
`examples/gliner_transcript_example.py` for complete runnable examples.

## CLI

The one-shot CLI accepts a UTF-8 text file followed by one or more labels and
prints `ToolOutput` YAML:

```bash
ampav_gliner_entities transcript.txt person organization location --threshold 0.4
```

Use `ampav_gliner_entities --help` for model cache, offline, chunk overlap, and
advanced inference options.

## Tests

Run unit tests without loading a real model:

```bash
python -m unittest discover -s tests
```

Run the cached real-model probes explicitly. Offline environment variables also
prevent the upstream tokenizer loader from attempting metadata requests:

```bash
AMPAV_GLINER_REAL_MODEL_TEST=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -k GlinerRealModelProbeTest
```
