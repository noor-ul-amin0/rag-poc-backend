# Document Ingestion Pipeline

A document parsing, chunking, and indexing pipeline for DOCX and PDF documents using Azure AI services.

## Features

- **DOCX & PDF Support**: Optimized parsing for Word and PDF documents
- **Azure Document Intelligence**: Uses Azure AI Document Intelligence for high-quality parsing
- **Mandatory Vector Embeddings**: Every chunk is embedded and stored in Azure AI Search
- **Intelligent Chunking**: SentenceTextSplitter uses semantic boundaries and token awareness
- **Figure Extraction**: Optionally extract and process figures from documents
- **Azure Search Integration**: Index chunks in Azure Cognitive Search with vectors
- **Cross-page Context**: Maintains semantic continuity across page boundaries

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `config.json` file in the root directory with your Azure credentials:

```json
{
  "document_intelligence_service": "your-doc-intel-service-name",
  "document_intelligence_key": null,
  "use_local_pdf_parser": false,
  "process_figures": false,

  "search_service": "your-search-service-name",
  "search_index": "documents",
  "search_key": null,

  "azure_openai_service": "your-azure-openai-service-name",
  "azure_openai_endpoint": null,
  "azure_openai_key": "your-azure-openai-api-key",
  "azure_openai_api_version": "2024-06-01",
  "azure_openai_embedding_deployment": "text-embedding-3-small",
  "azure_openai_embedding_model": "text-embedding-3-small",
  "azure_openai_embedding_dimensions": 1536,
  "search_embedding_field": "embedding",

  "category": "default"
}
```

3. (Optional) Set up Azure authentication:
   - If `document_intelligence_key` is null, DefaultAzureCredential will be used
   - Make sure you're logged in: `az login`

## Usage

### Parse and Chunk DOCX and PDF Documents

```bash
python scripts/run_ingestion.py ./data config.json
```

This will:

1. Scan all DOCX and PDF files in `./data` directory (recursively)
2. Parse each document using Document Intelligence (or local PDF parser)
3. Split content into semantic chunks
4. Generate embeddings for every chunk (mandatory)
5. Store chunks and vectors in Azure AI Search

### Configuration Options

**Parsing:**

- `document_intelligence_service`: Name of your Document Intelligence service (required for DOCX, optional for PDF)
- `document_intelligence_key`: API key (optional, uses DefaultAzureCredential if null)
- `use_local_pdf_parser`: Use PyPDF instead of Document Intelligence for PDFs (faster but less accurate)
- `process_figures`: Extract figures from documents as images

**Indexing:**

- `search_service`: Name of your Azure Search service
- `search_index`: Index name in Azure Search (default: "documents")
- `search_key`: Search service API key (optional, uses DefaultAzureCredential if null)
- `search_embedding_field`: Vector field name in the index (default: "embedding")

**Embeddings (Required):**

- `azure_openai_service`: Azure OpenAI service name (used if `azure_openai_endpoint` is null)
- `azure_openai_endpoint`: Optional explicit endpoint override
- `azure_openai_key`: Azure OpenAI API key
- `azure_openai_api_version`: API version for embeddings API
- `azure_openai_embedding_deployment`: Azure OpenAI embedding deployment name
- `azure_openai_embedding_model`: Embedding model name (for logs/config clarity)
- `azure_openai_embedding_dimensions`: Embedding vector size (must match model/index)

**Processing:**

- `category`: Document category for indexing (default: "default")

## Supported File Formats

| Format  | Parser                                   | Requirements                               |
| ------- | ---------------------------------------- | ------------------------------------------ |
| `.docx` | DocumentAnalysisParser                   | Document Intelligence                      |
| `.pdf`  | DocumentAnalysisParser or LocalPdfParser | Document Intelligence (or pypdf for local) |

**Note:** To parse DOCX files, you must configure `document_intelligence_service`. PDF files can use either Document Intelligence (recommended for complex PDFs with tables/figures) or local PyPDF parser (faster for simple text PDFs).

## Chunking Strategy

The `SentenceTextSplitter` uses an intelligent approach:

1. **Sentence-aware boundaries**: Prefers splitting at sentence endings (`.`, `!`, `?`)
2. **Token-aware limits**: Hard cap of 500 tokens per chunk (~400-500 characters for English)
3. **Semantic overlap**: 10% overlap between chunks for context preservation
4. **Cross-page merging**: Combines chunks across page boundaries when semantically continuous
5. **Figure handling**: Treats figures as atomic blocks (never split)
6. **Multi-language support**: Recognizes CJK (Chinese, Japanese, Korean) sentence boundaries

## Output Structure

The pipeline produces a list of `Section` objects, each containing:

```python
{
    "chunk": {
        "page_num": 0,          # 0-indexed page number
        "text": "...",          # Chunk text
        "images": []            # ImageOnPage objects if present
    },
    "content": {
        "filename": "doc.docx", # Original filename
        "url": "..."            # File URL/path
    },
    "category": "default"       # Document category
}
```

## Example

```bash
# Prepare test data
mkdir -p ./data
cp /path/to/your/document.docx ./data/

# Configure Azure services
# Edit config.json with your service names

# Run ingestion
python scripts/run_ingestion.py ./data config.json

# Output:
# ============================================================
# Ingestion Summary
# ============================================================
# Total sections created: 42
# Input directory: ./data
# Configuration: config.json
# ============================================================
```

## Azure Service Setup

### Document Intelligence

1. Create a Document Intelligence resource in Azure
2. Copy the service name and key
3. Update `config.json` with your service name

### Azure Cognitive Search

1. Create a Search service in Azure
2. The script auto-creates/updates the index if needed
3. Ensure Azure OpenAI embedding config is set in `config.json`

## Performance Notes

- **Local PDF parsing** is faster but less accurate than Document Intelligence
- **Figure extraction** adds processing time but captures visual content
- **Token-based chunking** ensures consistency with embedding model limits
- Large documents (100+ MB) may take several minutes

## Troubleshooting

**"No module named 'prepdocslib'"**

- Make sure you're running from the `ingestion_pipeline` directory
- Or add the directory to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

**"Document Intelligence service not found"**

- Check service name in config.json
- Verify authentication: `az login`

**"Unsupported file type"**

- Enable Document Intelligence in config.json to support more formats
- Or use appropriate local parsers

## Development

To modify the chunking strategy:

1. Edit `prepdocslib/textsplitter.py`
2. Adjust `DEFAULT_SECTION_LENGTH` (character limit)
3. Adjust `semantic_overlap_percent` (overlap amount)
4. Run tests with sample documents

## License

This code is based on Azure RAG patterns and is provided as-is.
