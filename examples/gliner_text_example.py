"""Extract people and organizations from a short text string."""

from ampav.gliner import GlinerNamedEntityExtractor


TEXT = "Dr. Maya Chen visited Indiana University in Bloomington."


def main() -> None:
    """Run the default CPU model and print AMPAV ToolOutput YAML."""
    extractor = GlinerNamedEntityExtractor()
    result = extractor.process(
        TEXT,
        ["person", "organization", "location"],
        threshold=0.4,
    )
    print(result.model_dump_yaml(sort_keys=False))


if __name__ == "__main__":
    main()
