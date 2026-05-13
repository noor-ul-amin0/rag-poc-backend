"""Service setup helpers for document ingestion."""

import logging
from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential

from .fileprocessor import FileProcessor
from .pdfparser import DocumentAnalysisParser, LocalPdfParser
from .textsplitter import SentenceTextSplitter

logger = logging.getLogger("scripts")


def build_file_processors(
    *,
    azure_credential: AsyncTokenCredential,
    document_intelligence_service: str | None,
    document_intelligence_key: str | None = None,
    use_local_pdf_parser: bool = False,
    process_figures: bool = False,
    preprocess_docx_images: bool = False,
) -> dict[str, FileProcessor]:
    """Build file processors for DOCX and PDF documents."""
    sentence_text_splitter = SentenceTextSplitter()

    # Set up Document Intelligence parser for DOCX and PDF
    doc_int_parser: Optional[DocumentAnalysisParser] = None
    if document_intelligence_service:
        credential: AsyncTokenCredential | AzureKeyCredential
        if document_intelligence_key:
            credential = AzureKeyCredential(document_intelligence_key)
        else:
            credential = azure_credential
        doc_int_parser = DocumentAnalysisParser(
            endpoint=f"https://{document_intelligence_service}.cognitiveservices.azure.com/",
            credential=credential,
            process_figures=process_figures,
            preprocess_docx_images=preprocess_docx_images,
        )

    # Set up PDF parser (prioritize Document Intelligence, fallback to local)
    pdf_parser: Optional = None
    if use_local_pdf_parser:
        pdf_parser = LocalPdfParser()
    elif doc_int_parser is not None:
        pdf_parser = doc_int_parser
    else:
        logger.warning("No PDF parser available. Set document_intelligence_service or use_local_pdf_parser.")

    # Build file processors for DOCX and PDF
    file_processors = {}
    
    # DOCX requires Document Intelligence
    if doc_int_parser is not None:
        file_processors[".docx"] = FileProcessor(doc_int_parser, sentence_text_splitter)
    else:
        logger.warning("DOCX support requires Document Intelligence service. Please configure document_intelligence_service.")
    
    # PDF can use either Document Intelligence or local parser
    if pdf_parser is not None:
        file_processors[".pdf"] = FileProcessor(pdf_parser, sentence_text_splitter)
    
    return file_processors
