"""Extract entities from timestamped transcript words."""

from ampav.core.schema import Transcript, WordSegment
from ampav.gliner import GlinerNamedEntityExtractor
from ampav_gliner_pipeline import extract_named_entities_from_transcript


def main() -> None:
    """Run GLiNER and align returned entities to transcript timestamps."""
    transcript = Transcript(
        languages=["en"],
        words=[
            WordSegment(word="Maya", start_time=0.0, end_time=0.3),
            WordSegment(word="Chen", start_time=0.3, end_time=0.6),
            WordSegment(word="visited", start_time=0.7, end_time=1.0),
            WordSegment(word="Indiana", start_time=1.1, end_time=1.5),
            WordSegment(word="University", suffix=".", start_time=1.5, end_time=2.0),
        ],
    )
    result = extract_named_entities_from_transcript(
        transcript,
        ["person", "organization"],
        extractor=GlinerNamedEntityExtractor(),
        threshold=0.4,
    )
    print(result.model_dump_yaml(sort_keys=False))


if __name__ == "__main__":
    main()
