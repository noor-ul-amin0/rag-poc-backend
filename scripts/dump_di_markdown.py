"""
Diagnostic script: dumps the raw markdown returned by Azure Document Intelligence
for a given file, then extracts and prints only the heading lines.

Usage (from ingestion_pipeline directory):
    python scripts/dump_di_markdown.py <path_to_docx_or_pdf> [--full]

Options:
    --full    Also write the complete raw markdown to <filename>.di_output.md

Example:
    python scripts/dump_di_markdown.py ".\data\AO Application Functional Documentation – Page Overview & Measure Logic.docx"
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential


async def dump_markdown(file_path: Path, config: dict, write_full: bool) -> None:
    endpoint = f"https://{config['document_intelligence_service']}.cognitiveservices.azure.com/"
    key = config["document_intelligence_key"]

    print(f"Sending '{file_path.name}' to Azure Document Intelligence...")

    async with DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key)) as client:
        content_bytes = file_path.read_bytes()
        poller = await client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=AnalyzeDocumentRequest(bytes_source=content_bytes),
            output_content_format="markdown",
        )
        result = await poller.result()

    raw_markdown: str = result.content or ""

    if write_full:
        out_path = file_path.with_suffix(".di_output.md")
        out_path.write_text(raw_markdown, encoding="utf-8")
        print(f"\nFull DI markdown written to: {out_path}\n")

    # Extract heading lines only
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    headings = heading_re.findall(raw_markdown)

    print(f"{'Level':<8} {'Hashes':<10} Heading text")
    print("-" * 80)
    for hashes, text in headings:
        level = len(hashes)
        print(f"h{level:<7} {hashes:<10} {text.strip()}")

    print(f"\nTotal headings detected by DI: {len(headings)}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    write_full = "--full" in args
    file_args = [a for a in args if not a.startswith("--")]
    file_path = Path(file_args[0])

    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    config_path = Path(__file__).parent.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    asyncio.run(dump_markdown(file_path, config, write_full))


if __name__ == "__main__":
    main()
