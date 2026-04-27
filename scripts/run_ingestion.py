"""Main ingestion pipeline entry point."""

import asyncio
import argparse
import base64
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add parent directory to Python path so prepdocslib can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from openai import AsyncAzureOpenAI

from prepdocslib.listfilestrategy import File, LocalListFileStrategy
from prepdocslib.servicesetup import build_file_processors
from prepdocslib.textprocessor import process_text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("scripts")

MAX_UPLOAD_BATCH_BYTES = int(os.getenv("SEARCH_UPLOAD_BATCH_MAX_BYTES", "12000000"))


def build_section_id(source_filename: str, page_num: int) -> str:
    """Build a deterministic Azure Search document id for a section."""
    filename_hash = hashlib.sha1(source_filename.encode("utf-8")).hexdigest()[:16]
    return f"{filename_hash}-{page_num}"


def sourcepage_from_file_page(filename: str, page: int = 0) -> str:
    """Match root app sourcepage behavior used for citations in Azure AI Search."""
    if os.path.splitext(filename)[1].lower() == ".pdf":
        return f"{os.path.basename(filename)}#page={page + 1}"
    return os.path.basename(filename)


def get_required_config(config: dict, key: str) -> str:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required config value: '{key}'")
    return str(value)


def resolve_azure_openai_endpoint(config: dict) -> str:
    endpoint = config.get("azure_openai_endpoint")
    if endpoint:
        return str(endpoint).rstrip("/")
    service_name = get_required_config(config, "azure_openai_service")
    return f"https://{service_name}.openai.azure.com"


def create_embedding_client(config: dict) -> tuple[AsyncAzureOpenAI, str, str, int, str]:
    """Create Azure OpenAI embeddings client and return runtime embedding settings."""
    endpoint = resolve_azure_openai_endpoint(config)
    api_key = get_required_config(config, "azure_openai_key")
    deployment = get_required_config(config, "azure_openai_embedding_deployment")
    model_name = config.get("azure_openai_embedding_model", "text-embedding-3-small")
    dimensions = int(config.get("azure_openai_embedding_dimensions", 1536))
    field_name = config.get("search_embedding_field", "embedding")
    api_version = config.get("azure_openai_api_version", "2024-06-01")

    client = AsyncAzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
    return client, deployment, str(model_name), dimensions, str(field_name)


async def create_text_embeddings(
    embedding_client: AsyncAzureOpenAI,
    deployment: str,
    model_name: str,
    dimensions: int,
    texts: list[str],
) -> list[list[float]]:
    """Create embeddings for all chunk texts in batches."""
    if len(texts) == 0:
        return []

    batch_size = 16
    vectors: list[list[float]] = []
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        response = await embedding_client.embeddings.create(
            model=deployment,
            input=batch,
            dimensions=dimensions,
        )
        vectors.extend([item.embedding for item in response.data])
        logger.info(
            "Computed embeddings for batch %d-%d using model '%s'",
            batch_start,
            min(batch_start + len(batch), len(texts)),
            model_name,
        )
    return vectors


def make_embedding_field(field_name: str, dimensions: int) -> SearchField:
    return SearchField(
        name=field_name,
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        filterable=False,
        sortable=False,
        facetable=False,
        vector_search_dimensions=dimensions,
        vector_search_profile_name=f"{field_name}-profile",
    )


def make_vector_search(field_name: str) -> VectorSearch:
    return VectorSearch(
        profiles=[
            VectorSearchProfile(
                name=f"{field_name}-profile",
                algorithm_configuration_name=f"{field_name}-hnsw",
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name=f"{field_name}-hnsw",
                parameters=HnswParameters(metric="cosine"),
            )
        ],
    )


def split_documents_by_payload_size(documents: list[dict[str, Any]], max_payload_bytes: int) -> list[list[dict[str, Any]]]:
    """Split documents into batches that stay under an approximate payload size limit."""
    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_size = 0

    for doc in documents:
        doc_size = len(json.dumps(doc).encode("utf-8"))
        if doc_size > max_payload_bytes:
            raise ValueError(
                f"Document '{doc.get('id', '<unknown>')}' is too large for Azure Search upload "
                f"({doc_size} bytes > max batch limit {max_payload_bytes} bytes)."
            )

        # Add a small overhead per document for JSON array punctuation.
        projected_size = current_size + doc_size + 2
        if current_batch and projected_size > max_payload_bytes:
            batches.append(current_batch)
            current_batch = [doc]
            current_size = doc_size + 2
        else:
            current_batch.append(doc)
            current_size = projected_size

    if current_batch:
        batches.append(current_batch)

    return batches


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return json.load(f)


async def parse_and_chunk_documents(config: dict, input_dir: str) -> list:
    """Parse and chunk documents from input directory."""
    
    logger.info("Starting document ingestion from: %s", input_dir)
    
    # Get credentials
    azure_credential = DefaultAzureCredential()
    
    # Build file processors
    file_processors = build_file_processors(
        azure_credential=azure_credential,
        document_intelligence_service=config.get("document_intelligence_service"),
        document_intelligence_key=config.get("document_intelligence_key"),
        use_local_pdf_parser=config.get("use_local_pdf_parser", False),
        process_figures=config.get("process_figures", False),
    )
    
    logger.info("Available file processors: %s", list(file_processors.keys()))
    
    # List files to ingest
    file_list_strategy = LocalListFileStrategy(
        path_pattern=os.path.join(input_dir, "**", "*"),
        enable_global_documents=True,
    )
    
    all_sections = []
    file_count = 0
    
    # Process each file
    async for file in file_list_strategy.list():
        file_count += 1
        logger.info("Found file: %s", file.filename())
        try:
            file_extension = file.file_extension().lower()
            
            if file_extension not in file_processors:
                logger.warning("Skipping '%s', unsupported file type", file.filename())
                continue
            
            processor = file_processors[file_extension]
            
            logger.info("Ingesting '%s'", file.filename())
            
            # Parse file into pages
            pages = []
            async for page in processor.parser.parse(content=file.content):
                pages.append(page)
            
            # Process text and split into chunks
            sections = process_text(
                pages,
                file,
                processor.splitter,
                category=config.get("category", "default")
            )
            
            all_sections.extend(sections)
            logger.info("Successfully ingested '%s': %d sections", file.filename(), len(sections))
            
        except Exception as e:
            logger.error("Error ingesting '%s': %s", file.filename(), str(e), exc_info=True)
        finally:
            file.close()
    
    logger.info("Total files selected for ingestion: %d", file_count)
    logger.info(
        "File scan summary: seen=%d, selected=%d",
        file_list_strategy.total_files_seen,
        file_list_strategy.total_selected,
    )
    
    return all_sections


async def index_sections(config: dict, sections: list):
    """Index sections in Azure Search."""
    
    if not config.get("search_service"):
        logger.warning("Search service not configured, skipping indexing")
        return
    
    logger.info("Indexing %d sections in Azure Search", len(sections))
    
    # Get credentials
    if config.get("search_key"):
        credential = AzureKeyCredential(config["search_key"])
    else:
        credential = DefaultAzureCredential()
    
    if len(sections) == 0:
        logger.info("No sections to upload. Skipping Azure Search upload.")
        return

    # Embeddings are mandatory in this project
    embedding_client, embedding_deployment, embedding_model, embedding_dimensions, embedding_field_name = (
        create_embedding_client(config)
    )

    # Connect to search service
    search_endpoint = f"https://{config['search_service']}.search.windows.net"
    index_name = config["search_index"]

    try:
        # Ensure index exists and contains required vector field/schema.
        async with SearchIndexClient(endpoint=search_endpoint, credential=credential) as search_index_client:
            existing_indexes = [name async for name in search_index_client.list_index_names()]
            if index_name not in existing_indexes:
                logger.info("Search index '%s' not found. Creating it.", index_name)
                index = SearchIndex(
                    name=index_name,
                    fields=[
                        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                        SearchableField(name="content", type=SearchFieldDataType.String),
                        SearchableField(
                            name="breadcrumb",
                            type=SearchFieldDataType.String,
                            filterable=True,
                        ),
                        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
                        SimpleField(name="sourcepage", type=SearchFieldDataType.String, filterable=True, facetable=True),
                        SimpleField(name="sourcefile", type=SearchFieldDataType.String, filterable=True, facetable=True),
                        SearchField(name="page_num", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                        SimpleField(name="images", type=SearchFieldDataType.String, filterable=False),
                        make_embedding_field(embedding_field_name, embedding_dimensions),
                    ],
                    semantic_search=SemanticSearch(
                        default_configuration_name="default",
                        configurations=[
                            SemanticConfiguration(
                                name="default",
                                prioritized_fields=SemanticPrioritizedFields(
                                    title_field=SemanticField(field_name="sourcepage"),
                                    content_fields=[SemanticField(field_name="content")],
                                    keywords_fields=[SemanticField(field_name="breadcrumb")],
                                ),
                            )
                        ],
                    ),
                    vector_search=make_vector_search(embedding_field_name),
                )
                await search_index_client.create_index(index)
                logger.info("Created search index '%s'", index_name)
            else:
                existing_index = await search_index_client.get_index(index_name)
                existing_field_names = {field.name for field in existing_index.fields}
                needs_update = False
                if embedding_field_name not in existing_field_names:
                    logger.info(
                        "Index '%s' missing vector field '%s'. Adding it.",
                        index_name,
                        embedding_field_name,
                    )
                    existing_index.fields.append(make_embedding_field(embedding_field_name, embedding_dimensions))
                    if existing_index.vector_search is None:
                        existing_index.vector_search = make_vector_search(embedding_field_name)
                    needs_update = True
                if "images" not in existing_field_names:
                    logger.info("Index '%s' missing 'images' field. Adding it.", index_name)
                    existing_index.fields.append(
                        SimpleField(name="images", type=SearchFieldDataType.String, filterable=False)
                    )
                    needs_update = True
                if "breadcrumb" not in existing_field_names:
                    logger.info("Index '%s' missing 'breadcrumb' field. Adding it.", index_name)
                    existing_index.fields.append(
                        SearchableField(name="breadcrumb", type=SearchFieldDataType.String, filterable=True)
                    )
                    needs_update = True
                if needs_update:
                    await search_index_client.create_or_update_index(existing_index)
                    logger.info("Updated index '%s' schema", index_name)

        # Prepare documents using root app field naming and id strategy
        documents: list[dict[str, Any]] = []
        for section_index, section in enumerate(sections):
            source_filename = section.content.filename()
            source_id = build_section_id(source_filename, section.chunk.page_num)
            images_data: list[dict[str, str]] = []
            for img in section.chunk.images:
                # Keep only images that are actually referenced by this chunk text.
                if img.bytes and img.placeholder in section.chunk.text:
                    images_data.append({
                        "figure_id": img.figure_id,
                        "data": base64.b64encode(img.bytes).decode("utf-8"),
                        "mime_type": img.mime_type,
                        "title": img.title or "",
                    })
                    logger.info(
                        "Storing image '%s' (%d bytes) for chunk '%s-chunk-%d'",
                        img.figure_id,
                        len(img.bytes),
                        source_id,
                        section_index,
                    )
            documents.append(
                {
                    # section index prevents collisions when multiple chunks come from same page
                    "id": f"{source_id}-chunk-{section_index}",
                    "content": section.chunk.text,
                    "breadcrumb": section.chunk.breadcrumb,
                    "category": section.category or "default",
                    "sourcepage": sourcepage_from_file_page(source_filename, page=section.chunk.page_num),
                    "sourcefile": source_filename,
                    "page_num": section.chunk.page_num,
                    "images": json.dumps(images_data),
                }
            )

        # Embeddings are always generated and stored.
        # The embedding input prepends the breadcrumb so the vector captures
        # the full heading hierarchy, not just the raw chunk text.  This lets
        # the retriever distinguish "OnHand on Summary page" from
        # "OnHand on Críticos page" even when the body text is nearly identical.
        logger.info(
            "Generating embeddings for %d chunks using deployment '%s'",
            len(documents),
            embedding_deployment,
        )
        embedding_texts = [
            f"{doc['breadcrumb']}\n\n{doc['content']}" if doc.get("breadcrumb") else doc["content"]
            for doc in documents
        ]
        vectors = await create_text_embeddings(
            embedding_client=embedding_client,
            deployment=embedding_deployment,
            model_name=embedding_model,
            dimensions=embedding_dimensions,
            texts=embedding_texts,
        )
        for idx, vector in enumerate(vectors):
            documents[idx][embedding_field_name] = vector

        logger.info("Uploading %d documents to index '%s'", len(documents), index_name)

        # Upload documents in payload-safe batches and validate indexing results.
        upload_batches = split_documents_by_payload_size(documents, max_payload_bytes=MAX_UPLOAD_BATCH_BYTES)
        logger.info(
            "Prepared %d upload batch(es) with max payload size %d bytes",
            len(upload_batches),
            MAX_UPLOAD_BATCH_BYTES,
        )

        async with SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential) as search_client:
            uploaded_count = 0
            for batch_index, batch in enumerate(upload_batches, start=1):
                try:
                    results = await search_client.upload_documents(documents=batch)
                    failed = [r for r in results if not getattr(r, "succeeded", False)]
                    if failed:
                        failed_keys = [getattr(r, "key", "<unknown>") for r in failed[:10]]
                        raise RuntimeError(
                            f"Failed to index {len(failed)} document(s) in upload batch {batch_index}. "
                            f"Example keys: {failed_keys}"
                        )
                    uploaded_count += len(batch)
                    logger.info(
                        "Uploaded batch %d/%d (%d docs, cumulative=%d/%d)",
                        batch_index,
                        len(upload_batches),
                        len(batch),
                        uploaded_count,
                        len(documents),
                    )
                except Exception as e:
                    logger.error("Error uploading batch %d: %s", batch_index, str(e), exc_info=True)
                    raise
    finally:
        await embedding_client.close()


async def main():
    """Main ingestion pipeline."""

    parser = argparse.ArgumentParser(description="Parse, chunk, and optionally index documents.")
    parser.add_argument("input_directory", help="Directory containing source documents")
    parser.add_argument("config_file", nargs="?", default="config.json", help="Path to config JSON")
    args = parser.parse_args()

    input_dir = args.input_directory
    config_file = args.config_file
    
    # Validate input directory
    if not os.path.isdir(input_dir):
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(config_file)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Parse and chunk documents
    try:
        sections = await parse_and_chunk_documents(config, input_dir)
        logger.info("Successfully parsed and chunked %d sections", len(sections))
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Ingestion Summary")
        print(f"{'='*60}")
        print(f"Total sections created: {len(sections)}")
        print(f"Input directory: {input_dir}")
        print(f"Configuration: {config_file}")
        
        # Index sections if search service is configured
        if config.get("search_service"):
            await index_sections(config, sections)
        else:
            print("\nNote: Search service not configured. Sections were parsed and chunked but not indexed.")
            print("To index, configure search_service in config.json and re-run.")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error("Pipeline error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
