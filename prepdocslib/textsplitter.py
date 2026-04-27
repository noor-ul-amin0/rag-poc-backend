import logging
import re
from abc import ABC
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

from .page import Chunk, Page

logger = logging.getLogger("scripts")


class TextSplitter(ABC):
    """Splits a list of pages into smaller chunks."""

    def split_pages(self, pages: list[Page]) -> Generator[Chunk, None, None]:
        if False:
            yield


ENCODING_MODEL = "text-embedding-ada-002"

STANDARD_WORD_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]

CJK_WORD_BREAKS = [
    "、", "，", "；", "：", "（", "）", "【", "】", "「", "」", "『", "』",
    "〔", "〕", "〈", "〉", "《", "》", "〖", "〗", "〘", "〙", "〚", "〛",
    "〝", "〞", "〟", "〰", "–", "—", "'", "'", "‚", "‛", """, """, "„", "‟", "‹", "›",
]

STANDARD_SENTENCE_ENDINGS = [".", "!", "?"]

CJK_SENTENCE_ENDINGS = ["。", "！", "？", "‼", "⁇", "⁈", "⁉"]

bpe = tiktoken.encoding_for_model(ENCODING_MODEL)

DEFAULT_OVERLAP_PERCENT = 10
DEFAULT_SECTION_LENGTH = 2000


def _safe_concat(a: str, b: str) -> str:
    """Concatenate two non-empty segments safely."""
    assert a and b, "_safe_concat expects non-empty strings"
    a_last = a[-1]
    b_first = b[0]
    if a_last.isspace() or b_first.isspace():
        return a + b
    if a_last == ">":
        return a + b
    if a_last.isalnum() and b_first.isalnum():
        return a + " " + b
    return a + b


def _normalize_chunk(text: str, max_chars: int) -> str:
    """Normalize a chunk that may slightly exceed max_chars."""
    lower = text.lower()
    if "<figure" in lower:
        return text
    if len(text) <= max_chars:
        return text
    trimmed = text
    while trimmed.startswith(" ") and len(trimmed) > max_chars:
        trimmed = trimmed[1:]
    if len(trimmed) > max_chars and len(trimmed) <= max_chars + 3:
        if trimmed.endswith(" ") or trimmed.endswith("\n"):
            trimmed = trimmed.rstrip()
    return trimmed


@dataclass
class _ChunkBuilder:
    """Accumulates sentence-like spans for a single page until size limits are reached."""

    page_num: int
    max_chars: int
    max_tokens: int
    breadcrumb: str = ""
    parts: list[str] = field(default_factory=list)
    token_len: int = 0

    def can_fit(self, text: str, token_count: int) -> bool:
        if not self.parts:
            return token_count <= self.max_tokens and len(text) <= self.max_chars
        return (len("".join(self.parts)) + len(text) <= self.max_chars) and (
            self.token_len + token_count <= self.max_tokens
        )

    def add(self, text: str, token_count: int) -> bool:
        if not self.can_fit(text, token_count):
            return False
        self.parts.append(text)
        self.token_len += token_count
        return True

    def force_append(self, text: str):
        self.parts.append(text)

    def flush_into(self, out: list[Chunk]):
        if self.parts:
            chunk = "".join(self.parts)
            if chunk.strip():
                out.append(Chunk(page_num=self.page_num, text=chunk, breadcrumb=self.breadcrumb))
        self.parts.clear()
        self.token_len = 0

    def has_content(self) -> bool:
        return bool(self.parts)

    def append_figure_and_flush(self, figure_text: str, out: list[Chunk]):
        """Append a figure (allowed to overflow) to current accumulation and flush in one step."""
        self.force_append(figure_text)
        self.flush_into(out)


class SentenceTextSplitter(TextSplitter):
    """Splits pages into semantic chunks using sentence-aware boundaries."""

    def __init__(self, max_tokens_per_section: int = 800):
        self.sentence_endings = STANDARD_SENTENCE_ENDINGS + CJK_SENTENCE_ENDINGS
        self.word_breaks = STANDARD_WORD_BREAKS + CJK_WORD_BREAKS
        self.max_section_length = DEFAULT_SECTION_LENGTH
        self.sentence_search_limit = 100
        self.max_tokens_per_section = max_tokens_per_section
        self.section_overlap = int(self.max_section_length * DEFAULT_OVERLAP_PERCENT / 100)
        self.semantic_overlap_percent = 10

    def _find_split_pos(self, text: str) -> tuple[int, bool]:
        """Find a good split position near midpoint."""
        length = len(text)
        if length < 4:
            return -1, True
        mid = length // 2
        window_limit = length // 3

        pos = 0
        while mid - pos > window_limit:
            left = mid - pos
            right = mid + pos
            if left >= 0 and text[left] in self.sentence_endings:
                return left, False
            if right < length and text[right] in self.sentence_endings:
                return right, False
            pos += 1

        pos = 0
        while mid - pos > window_limit:
            left = mid - pos
            right = mid + pos
            if left >= 0 and text[left] in self.word_breaks:
                return left, False
            if right < length and text[right] in self.word_breaks:
                return right, False
            pos += 1

        return -1, True

    def split_page_by_max_tokens(self, page_num: int, text: str, breadcrumb: str = "") -> Generator[Chunk, None, None]:
        """Recursively split plain text by token count."""
        tokens = bpe.encode(text)
        if len(tokens) <= self.max_tokens_per_section:
            yield Chunk(page_num=page_num, text=text, breadcrumb=breadcrumb)
            return

        split_pos, use_overlap = self._find_split_pos(text)
        if not use_overlap and split_pos > 0:
            first_half = text[: split_pos + 1]
            second_half = text[split_pos + 1 :]
        else:
            middle = len(text) // 2
            overlap = int(len(text) * (DEFAULT_OVERLAP_PERCENT / 100))
            first_half = text[: middle + overlap]
            second_half = text[middle - overlap :]

        yield from self.split_page_by_max_tokens(page_num, first_half, breadcrumb)
        yield from self.split_page_by_max_tokens(page_num, second_half, breadcrumb)

    def _is_heading_like(self, line: str) -> bool:
        """Heuristic heading detector."""
        line_str = line.strip()
        if not line_str:
            return False
        if line_str.startswith("#"):
            return True
        if len(line_str) <= 80 and (line_str.isupper() or (line_str.istitle() and len(line_str.split()) <= 12)):
            return True
        if re.match(r"^(?:\d+|[IVXLCM]+)[.)]\s", line_str):
            return True
        if line_str.startswith(("- ", "* ", "• ")):
            return True
        return False

    def _should_cross_page_overlap(self, prev: Chunk, nxt: Chunk) -> bool:
        if not prev or not nxt:
            return False
        if "<figure" in prev.text.lower() or "<figure" in nxt.text[:40].lower():
            return False
        # Never merge content that belongs to different heading sections.
        if prev.breadcrumb != nxt.breadcrumb:
            return False
        prev_last = prev.text.rstrip()[-1:] if prev.text.rstrip() else ""
        if prev_last in self.sentence_endings:
            return False
        nxt_stripped = nxt.text.lstrip()
        if not nxt_stripped:
            return False
        first_char = nxt_stripped[0]
        if not first_char.islower():
            return False
        first_line = nxt_stripped.splitlines()[0]
        if self._is_heading_like(first_line):
            return False
        return True

    def _append_overlap(self, prev_chunk: Chunk, next_chunk: Chunk) -> Chunk:
        """Append semantic overlap prefix from next_chunk to prev_chunk."""
        if not prev_chunk or not next_chunk:
            return prev_chunk
        if "<figure" in prev_chunk.text.lower() or "<figure" in next_chunk.text[:60].lower():
            return prev_chunk

        target = int(self.max_section_length * self.semantic_overlap_percent / 100)
        if target <= 0:
            return prev_chunk

        base_prefix = next_chunk.text[:target]
        if not base_prefix.strip():
            return prev_chunk

        extension_limit = min(len(next_chunk.text), target * 2)
        prefix = base_prefix
        boundary_found = False
        for i in range(target, extension_limit):
            ch = next_chunk.text[i]
            prefix += ch
            if ch in self.sentence_endings:
                boundary_found = True
                break
            if ch in self.word_breaks and i - target > 20:
                boundary_found = True
                break
        if not boundary_found:
            while prefix and prefix[-1].isalnum() and len(prefix) > target:
                prefix = prefix[:-1]

        if prev_chunk.text.endswith(prefix):
            return prev_chunk

        candidate = prev_chunk.text + prefix
        max_chars = int(self.max_section_length * 1.2)
        if len(candidate) > max_chars or len(bpe.encode(candidate)) > self.max_tokens_per_section:
            shrink = prefix
            while shrink and (
                len(prev_chunk.text + shrink) > max_chars
                or len(bpe.encode(prev_chunk.text + shrink)) > self.max_tokens_per_section
            ):
                cut_index = 1
                for i, ch in enumerate(shrink):
                    if ch in self.word_breaks or ch in self.sentence_endings:
                        cut_index = i + 1
                        break
                shrink = shrink[:-cut_index] if cut_index < len(shrink) else ""
            if not shrink:
                return prev_chunk
            candidate = prev_chunk.text + shrink
            if len(candidate) > max_chars or len(bpe.encode(candidate)) > self.max_tokens_per_section:
                return prev_chunk
        return Chunk(page_num=prev_chunk.page_num, text=candidate, breadcrumb=prev_chunk.breadcrumb)

    def split_pages(self, pages: list[Page]) -> Generator[Chunk, None, None]:
        """Split each page into semantic chunks using token-aware accumulation.

        Heading lines (``# …``, ``## …``, …) are detected inside every text
        block.  Whenever a heading is encountered the current builder is
        flushed, the heading stack is updated, a fresh builder is started with
        the new breadcrumb, and the heading line is prepended to the new chunk
        so the LLM always sees which section a chunk belongs to.

        The ``breadcrumb`` field on every emitted ``Chunk`` records the full
        heading path at the time of extraction, e.g.
        ``"AO Overview > Base Measures > OnHand Calculation"``.
        """
        figure_regex = re.compile(r"<figure.*?</figure>", re.IGNORECASE | re.DOTALL)
        # Capture group keeps heading lines as separate elements after re.split
        heading_split_re = re.compile(r"(^#{1,6}[ \t]+[^\n]*)$", re.MULTILINE)
        heading_match_re = re.compile(r"^(#{1,6})[ \t]+(.+)$")

        # Heading stack persists across pages – a heading on page 2 keeps its
        # H1 ancestor from page 1.
        heading_stack: list[str] = []
        previous_chunk: Optional[Chunk] = None

        def get_breadcrumb() -> str:
            return " > ".join(h for h in heading_stack if h)

        def update_heading_stack(level: int, title: str) -> None:
            idx = level - 1
            while len(heading_stack) <= idx:
                heading_stack.append("")
            heading_stack[idx] = title.strip().rstrip(":").strip()
            del heading_stack[idx + 1 :]

        for page in pages:
            raw = page.text or ""
            if not raw.strip():
                continue

            # ── Step 1: isolate <figure> blocks ──────────────────────────────
            blocks: list[tuple[str, str]] = []
            last = 0
            for m in figure_regex.finditer(raw):
                if m.start() > last:
                    blocks.append(("text", raw[last : m.start()]))
                blocks.append(("figure", m.group()))
                last = m.end()
            if last < len(raw):
                blocks.append(("text", raw[last:]))

            page_chunks: list[Chunk] = []
            builder = _ChunkBuilder(
                page_num=page.page_num,
                max_chars=self.max_section_length,
                max_tokens=self.max_tokens_per_section,
                breadcrumb=get_breadcrumb(),
            )

            for btype, btext in blocks:
                if btype == "figure":
                    if builder.has_content():
                        builder.append_figure_and_flush(btext, page_chunks)
                    else:
                        if btext.strip():
                            page_chunks.append(
                                Chunk(page_num=page.page_num, text=btext, breadcrumb=builder.breadcrumb)
                            )
                    continue

                # ── Step 2: within each text block, split on heading lines ───
                # re.split with a capture group keeps the heading lines as
                # standalone elements in the resulting list.
                sub_parts = heading_split_re.split(btext)
                for sub_part in sub_parts:
                    if not sub_part:
                        continue

                    hm = heading_match_re.match(sub_part.strip())
                    if hm:
                        # Flush whatever was accumulated under the old breadcrumb.
                        builder.flush_into(page_chunks)

                        # Advance the heading stack.
                        level = len(hm.group(1))
                        title = hm.group(2).strip().rstrip(":").strip()
                        update_heading_stack(level, title)
                        logger.info(
                            "[heading] h%d | %r | breadcrumb=%r",
                            level,
                            title,
                            get_breadcrumb(),
                        )

                        # Fresh builder with the updated breadcrumb.
                        builder = _ChunkBuilder(
                            page_num=page.page_num,
                            max_chars=self.max_section_length,
                            max_tokens=self.max_tokens_per_section,
                            breadcrumb=get_breadcrumb(),
                        )
                        # Include the heading line as the opening of the new
                        # chunk so the LLM sees the section title in context.
                        heading_line = sub_part.strip() + "\n"
                        builder.force_append(heading_line)
                        builder.token_len = len(bpe.encode(heading_line))
                        continue

                    # ── Step 3: normal sentence-span accumulation ────────────
                    spans: list[str] = []
                    current_chars: list[str] = []
                    for ch in sub_part:
                        current_chars.append(ch)
                        if ch in self.sentence_endings:
                            spans.append("".join(current_chars))
                            current_chars = []
                    if current_chars:
                        spans.append("".join(current_chars))

                    for span in spans:
                        span_tokens = len(bpe.encode(span))
                        if span_tokens > self.max_tokens_per_section:
                            builder.flush_into(page_chunks)
                            for chunk in self.split_page_by_max_tokens(
                                page.page_num, span, builder.breadcrumb
                            ):
                                page_chunks.append(chunk)
                            continue
                        if not builder.add(span, span_tokens):
                            builder.flush_into(page_chunks)
                            if not builder.add(span, span_tokens):
                                page_chunks.append(
                                    Chunk(
                                        page_num=page.page_num,
                                        text=span,
                                        breadcrumb=builder.breadcrumb,
                                    )
                                )

            builder.flush_into(page_chunks)

            # ── Step 4: optional cross-page sentence continuation merge ──────
            # Only merge when the two chunks share the same breadcrumb so that
            # content from different sections is never joined.
            if previous_chunk and page_chunks:
                prev_last_char = previous_chunk.text.rstrip()[-1:] if previous_chunk.text.rstrip() else ""
                first_new = page_chunks[0]
                first_new_stripped = first_new.text.lstrip()
                first_char = first_new_stripped[:1]
                if (
                    prev_last_char
                    and prev_last_char not in self.sentence_endings
                    and not first_new_stripped.startswith("#")
                    and first_char
                    and first_char.islower()
                    and "<figure" not in first_new_stripped[:20].lower()
                    and previous_chunk.breadcrumb == first_new.breadcrumb
                ):
                    combined_text = _safe_concat(previous_chunk.text, first_new.text)
                    if len(bpe.encode(combined_text)) <= self.max_tokens_per_section and len(
                        combined_text
                    ) <= int(self.max_section_length * 1.2):
                        previous_chunk = Chunk(
                            page_num=previous_chunk.page_num,
                            text=combined_text,
                            breadcrumb=previous_chunk.breadcrumb,
                        )
                        page_chunks = page_chunks[1:]

            max_chars = int(self.max_section_length * 1.2)
            if previous_chunk:
                previous_chunk = Chunk(
                    page_num=previous_chunk.page_num,
                    text=_normalize_chunk(previous_chunk.text, max_chars),
                    breadcrumb=previous_chunk.breadcrumb,
                )
            if page_chunks:
                page_chunks = [
                    Chunk(
                        page_num=chunk.page_num,
                        text=_normalize_chunk(chunk.text, max_chars),
                        breadcrumb=chunk.breadcrumb,
                    )
                    for chunk in page_chunks
                ]

            if self.semantic_overlap_percent > 0:
                if previous_chunk and page_chunks and self._should_cross_page_overlap(
                    previous_chunk, page_chunks[0]
                ):
                    previous_chunk = self._append_overlap(previous_chunk, page_chunks[0])

                if len(page_chunks) > 1:
                    for i in range(1, len(page_chunks)):
                        prev_c = page_chunks[i - 1]
                        curr_c = page_chunks[i]
                        if "<figure" in prev_c.text.lower() or "<figure" in curr_c.text.lower():
                            continue
                        page_chunks[i - 1] = self._append_overlap(prev_c, curr_c)

            if previous_chunk:
                yield previous_chunk

            if page_chunks:
                if len(page_chunks) == 1:
                    previous_chunk = page_chunks[0]
                else:
                    yield from page_chunks[:-1]
                    previous_chunk = page_chunks[-1]
            else:
                previous_chunk = None

        if previous_chunk:
            yield previous_chunk

