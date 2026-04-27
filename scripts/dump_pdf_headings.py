"""
Diagnostic script: extracts and logs headings from PDF files using pymupdf.

Usage (from ingestion_pipeline directory):
    python scripts/dump_pdf_headings.py <path_to_pdf>

Example:
    python scripts/dump_pdf_headings.py ".\data\sample.pdf"
"""

import sys
from pathlib import Path

import fitz


def extract_headings(file_path: Path) -> None:
    """Extract and log headings from a PDF file."""
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"Parsing '{file_path.name}' using pymupdf...\n")

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        sys.exit(1)

    headings = []

    # Try to extract headings from the table of contents
    toc = doc.get_toc()
    if toc:
        print("Headings from Table of Contents:")
        print(f"{'Level':<8} {'Page':<8} Heading text")
        print("-" * 100)
        for item in toc:
            level = item[0]
            text = item[1]
            page = item[2] if len(item) > 2 else "N/A"
            print(f"{level:<8} {str(page):<8} {text}")
            headings.append((level, text, page))

        print(f"\nTotal headings from TOC: {len(headings)}\n")
    else:
        print("No table of contents found. Extracting headings by font size...\n")

        # Extract headings by looking for larger font sizes
        for page_num, page in enumerate(doc, start=1):
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font_size = span.get("size", 0)
                            text = span.get("text", "").strip()
                            
                            # Consider text with font size > 12 as potential heading
                            if font_size > 12 and text:
                                headings.append((font_size, text, page_num))

        if headings:
            print(f"{'Font Size':<12} {'Page':<8} Heading text")
            print("-" * 100)
            for font_size, text, page_num in headings:
                print(f"{font_size:<12.1f} {page_num:<8} {text}")

            print(f"\nTotal headings extracted by font size: {len(headings)}")
        else:
            print("No headings found based on font size criteria.")

    doc.close()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    file_path = Path(args[0])
    extract_headings(file_path)


if __name__ == "__main__":
    main()
