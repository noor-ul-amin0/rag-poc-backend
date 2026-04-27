## Ingestion Pipeline - Complete File Structure

```
ingestion_pipeline/
├── prepdocslib/                    # Core parsing and chunking library
│   ├── __init__.py
│   ├── page.py                     # Page, Chunk, ImageOnPage data classes
│   ├── parser.py                   # Abstract Parser base class
│   ├── fileprocessor.py            # FileProcessor dataclass
│   ├── pdfparser.py                # LocalPdfParser & DocumentAnalysisParser for PDF/DOCX
│   ├── textsplitter.py             # SentenceTextSplitter (CORE)
│   ├── listfilestrategy.py         # LocalListFileStrategy for file discovery
│   ├── textprocessor.py            # process_text() - combines parsing + chunking
│   └── servicesetup.py             # build_file_processors() - factory function
│
├── scripts/
│   └── run_ingestion.py            # Main entry point - run this!
│
├── config.json                     # Configuration (YOUR AZURE CREDENTIALS)
├── requirements.txt                # Python dependencies
├── README.md                       # Full documentation
├── QUICKSTART.md                   # This file
├── example_usage.py                # Example: programmatic usage
├── setup.sh                        # Setup script (macOS/Linux)
├── setup.bat                       # Setup script (Windows)
├── .gitignore
└── data/                           # (Create this) Place DOCX and PDF files here

```

## Quick Start (5 minutes)

### 1. Prerequisites

```bash
# Make sure you have:
# - Python 3.10+
# - Azure Document Intelligence service (optional, for DOCX/PDF)
# - Azure Search service (optional, for indexing)
```

### 2. Initialize

```bash
# macOS/Linux
bash setup.sh

# Windows
setup.bat

# Or manually:
pip install -r requirements.txt
mkdir -p data
```

### 3. Configure

Edit `config.json`:

````json
{
  "document_intelligence_service": "your-doc-intel-name",
  "search_service": "your-search-name",

```bash
cp /path/to/your/document.docx ./data/
cp /path/to/your/file.pdf ./data/
````

### 5. Run Ingestion

```bash
# Parse and chunk from ./data directory
python scripts/run_ingestion.py ./data config.json

# Or use example script
python example_usage.py ./data config.json
```

## Output

The pipeline outputs `Section` objects with:

- `chunk.text` - The parsed and chunked content
- `chunk.page_num` - Original page number
- `chunk.images` - Associated figures (if extracted)
- `content.filename` - Source document filename
- `category` - Document category

## Supported Formats

| Format  | Parser                                | Requires                       |
| ------- | ------------------------------------- | ------------------------------ |
| `.docx` | DocumentAnalysisParser                | Document Intelligence          |
| `.pdf`  | DocumentAnalysisParser/LocalPdfParser | Optional Document Intelligence |

## Configuration Options

```json
{
  // Document Intelligence (for DOCX and optional PDF)
  "document_intelligence_service": "your-service-name",
  "document_intelligence_key": null, // null = use DefaultAzureCredential
  "use_local_pdf_parser": false, // true = use PyPDF instead of Document Intelligence
  "process_figures": false, // true = extract images

  // Azure Search (optional, for indexing)
  "search_service": "your-search-name",
  "search_index": "documents",
  "search_key": null, // null = use DefaultAzureCredential

  // Document categorization
  "category": "default"
}
```

## Core Modules Explained

### `textsplitter.py` - The Smart Chunking Engine

**Key Features:**

- **SentenceTextSplitter** (default): Uses semantic boundaries
  - Splits at sentence endings: `.`, `!`, `?`
  - Respects token limits (500 tokens max)
  - 10% overlap between chunks for context
  - Handles CJK languages

- **Parameters:**
  - `max_tokens_per_section`: 500 (hard limit)
  - `max_section_length`: 1000 chars (soft limit)
  - `semantic_overlap_percent`: 10%

### `pdfparser.py` - Document Intelligence Integration

- **DocumentAnalysisParser**: Uses Azure AI Document Intelligence
  - Supports 22+ document formats
  - Extracts tables as HTML
  - Optionally extracts figures with captions
  - Outputs markdown-formatted text

- **LocalPdfParser**: Lightweight alternative
  - Uses PyPDF, no API calls
  - Fast but less accurate
  - Good for simple PDFs

### `fileprocessor.py` - Parser + Splitter Pairing

Each file type gets a `FileProcessor` that pairs:

- A **Parser** (converts bytes → Pages)
- A **TextSplitter** (converts Pages → Chunks)

Example: DOCX → DocumentAnalysisParser + SentenceTextSplitter

## Usage Patterns

### Pattern 1: Simple Batch Processing

```bash
python scripts/run_ingestion.py ./documents config.json
```

### Pattern 2: Programmatic Usage

```python
import asyncio
from example_usage import ingest_folder

sections = asyncio.run(ingest_folder("./data", "config.json"))
for section in sections:
    print(section.chunk.text)
```

### Pattern 3: Custom Processing

```python
from prepdocslib.servicesetup import build_file_processors
from prepdocslib.textprocessor import process_text

# Build processors
file_processors = build_file_processors(...)

# Parse custom document
parser = file_processors[".docx"].parser
pages = [p async for p in parser.parse(file_content)]

# Chunk with custom settings
splitter = file_processors[".docx"].splitter
chunks = list(splitter.split_pages(pages))
```

## Troubleshooting

| Issue                             | Solution                                        |
| --------------------------------- | ----------------------------------------------- |
| `No module named 'prepdocslib'`   | Run from `ingestion_pipeline/` directory        |
| `Document Intelligence not found` | Check service name in `config.json`             |
| `Unsupported file type`           | Enable Document Intelligence in config          |
| `Slow processing`                 | Use `use_local_pdf_parser: true` for PDFs       |
| `Out of memory`                   | Process files one at a time; reduce token limit |

## Performance Tips

1. **Local parsing**: Use `use_local_pdf_parser: true` for faster processing (~10x)
2. **Batch processing**: Process many small files faster than one huge file
3. **Chunk size**: Larger chunks (1000+ tokens) = faster, less context
4. **Skip figures**: Set `process_figures: false` to skip image extraction

## File Descriptions

| File                  | Purpose                                    |
| --------------------- | ------------------------------------------ |
| `page.py`             | Data structures (Page, Chunk, ImageOnPage) |
| `parser.py`           | Abstract Parser base class                 |
| `textsplitter.py`     | **Sentence-aware chunking logic**          |
| `pdfparser.py`        | **Document Intelligence integration**      |
| `textprocessor.py`    | Combines figure descriptions with text     |
| `listfilestrategy.py` | Recursively lists files                    |
| `servicesetup.py`     | Factory for creating processors            |
| `run_ingestion.py`    | **Main entry point**                       |

## Next Steps

1. ✅ Complete - Ingestion pipeline created
2. → Configure Azure services in `config.json`
3. → Run: `python scripts/run_ingestion.py ./data config.json`
4. → (Optional) Set up Azure Search for indexing
5. → Integrate ingested chunks into your RAG system

## For More Information

- See [README.md](README.md) for full documentation
- See [example_usage.py](example_usage.py) for code examples
- See [config.json](config.json) for all configuration options
