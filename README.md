# ampav-gliner

GLiNER tooling for the AMPAV environment.

This package provides a thin synchronous wrapper that extracts named entities
from plain text and returns an AMPAV `ToolOutput` containing `NamedEntities`.

`GlinerNamedEntityExtractor.extract_from_transcript(...)` accepts an AMPAV
`Transcript`, sends canonical `words_to_text(transcript.words)` text to GLiNER, and
adds best-effort timestamp alignment messages to `ToolOutput.messages`.

Long-transcript chunking is not implemented yet.

## Tests

Run the default unit tests without loading a real GLiNER model:

```bash
python -m unittest discover -s tests
```

Run the cached real-model probe tests after the default model has been downloaded:

```bash
AMPAV_GLINER_REAL_MODEL_TEST=1 python -m unittest discover -s tests -k GlinerRealModelProbeTest
```
