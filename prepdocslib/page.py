import base64
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ImageOnPage:
    bytes: bytes
    bbox: tuple[float, float, float, float]  # Pixels
    filename: str
    figure_id: str
    page_num: int  # 0-indexed
    placeholder: str  # HTML placeholder in page text, e.g. '<figure id="fig_..."></figure>'
    mime_type: str = "image/png"
    url: Optional[str] = None
    title: str = ""
    embedding: Optional[list[float]] = None
    description: Optional[str] = None


@dataclass
class Page:
    """
    A single page from a document

    Attributes:
        page_num (int): Page number (0-indexed)
        offset (int): Character offset in the full document
        text (str): The text of the page
        images (list[ImageOnPage]): Images on this page
        tables (list[str]): HTML table strings on this page
    """

    page_num: int
    offset: int
    text: str
    images: list[ImageOnPage] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """Semantic chunk emitted by the splitter.

    Attributes:
        page_num (int): Logical source page number (0-indexed)
        text (str): Textual content of the chunk
        images (list[ImageOnPage]): Images associated with this chunk
        breadcrumb (str): Hierarchical heading path, e.g.
            "AO Overview > Base Measures > OnHand Calculation".
            Built from the document's Heading 1 → Heading N stack at the
            point where this chunk was extracted.  Empty when no heading
            context has been encountered yet.
    """

    page_num: int
    text: str
    images: list[ImageOnPage] = field(default_factory=list)
    breadcrumb: str = ""
