"""Utilities for processing document text and combining it with figure descriptions."""

import logging

from .listfilestrategy import File
from .page import Page
from .textsplitter import Chunk, TextSplitter

logger = logging.getLogger("scripts")


class Section:
    """A section of a document that is indexed in Azure Search."""

    def __init__(self, chunk: Chunk, content: File, category: str | None = None):
        self.chunk = chunk
        self.content = content
        self.category = category


def combine_text_with_figures(page: "Page") -> None:
    """Replace figure placeholders in page text with full description markup."""
    for image in page.images:
        if image.description and image.placeholder in page.text:
            # Simple figure markup with description
            figure_markup = f"<figure><figcaption>{image.title or image.figure_id}<br>{image.description}</figcaption></figure>"
            page.text = page.text.replace(image.placeholder, figure_markup)
            logger.info("Replaced placeholder for figure %s with description markup", image.figure_id)
        else:
            logger.warning(
                "Figure '%s' on page %d has NO description — placeholder left as-is in indexed text. "
                "Image was extracted (%d bytes) but will NOT contribute to search or LLM answers. "
                "To fix: add a GPT-4o Vision call to populate image.description before indexing.",
                image.figure_id,
                image.page_num,
                len(image.bytes),
            )


def process_text(
    pages: list["Page"],
    file: "File",
    splitter: "TextSplitter",
    category: str | None = None,
) -> list["Section"]:
    """Process document text and figures into searchable sections."""
    # Step 1: Combine text with figures on each page
    for page in pages:
        combine_text_with_figures(page)

    # Step 2: Split combined text into chunks
    logger.info("Splitting '%s' into sections", file.filename())
    sections = [Section(chunk, content=file, category=category) for chunk in splitter.split_pages(pages)]

    # Step 3: Add images back to each section based on page number
    for section in sections:
        section.chunk.images = [
            image for page in pages if page.page_num == section.chunk.page_num for image in page.images
        ]

    return sections
