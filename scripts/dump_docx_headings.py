"""
Diagnostic script: extracts and logs headings from DOCX files using python-docx.

Usage (from ingestion_pipeline directory):
    python scripts/dump_docx_headings.py <path_to_docx>

Example:
    python scripts/dump_docx_headings.py ".\data\AO Application Functional Documentation – Page Overview & Measure Logic.docx"
"""

import sys
from pathlib import Path

from docx import Document


def extract_headings(file_path: Path) -> None:
    """Extract and log headings from a DOCX file."""
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"Parsing '{file_path.name}' using python-docx...\n")

    try:
        doc = Document(file_path)
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        sys.exit(1)

    headings = []

    # Extract headings from document
    for para in doc.paragraphs:
        style = para.style.name
        # Check if the paragraph style is a heading style
        if style.startswith("Heading"):
            text = para.text.strip()
            if text:
                # Extract heading level from style name (e.g., "Heading 1" -> 1)
                try:
                    level = int(style.split()[-1])
                except (ValueError, IndexError):
                    level = 1
                headings.append((level, style, text))

    # Print results
    print(f"{'Level':<8} {'Style':<20} Heading text")
    print("-" * 100)
    for level, style, text in headings:
        print(f"h{level:<7} {style:<20} {text}")

    print(f"\nTotal headings found: {len(headings)}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    file_path = Path(args[0])
    extract_headings(file_path)


if __name__ == "__main__":
    main()
