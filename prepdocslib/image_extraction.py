"""
Pre-processes DOCX files to work around Azure Document Intelligence's
inability to extract images from DOCX documents.

Pipeline:
  1. Load the DOCX from raw bytes (in-memory, no local filesystem).
  2. Walk all paragraphs and locate embedded images via Office XML blip tags.
  3. Upload each image to a dedicated Azure Blob Storage container.
  4. Replace the image XML run with a markdown image tag pointing to the blob URL.
  5. Return the modified DOCX as bytes, ready for Azure DI parsing.
"""

import io
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from azure.storage.blob import BlobSasPermissions, BlobServiceClient, ContentSettings, generate_blob_sas
from docx import Document

logger = logging.getLogger("scripts")

# ── Office XML namespace constants ────────────────────────────────────────────

BLIP_TAG = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
EMBED_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"


def _parse_connection_string(conn_str: str) -> dict[str, str]:
    """Extract key-value pairs from an Azure Storage connection string.

    AccountKey is treated specially because its base64 value can contain '='.
    """
    parts: dict[str, str] = {}
    for segment in conn_str.split(";"):
        segment = segment.strip()
        if "=" in segment:
            key, _, _ = segment.partition("=")
            # Take everything after the first '=' to preserve base64 padding in AccountKey.
            parts[key.strip()] = segment.split("=", 1)[1].strip()
    return parts


def _upload_image_to_blob(
    image_bytes: bytes,
    blob_name: str,
    content_type: str,
    connection_string: str,
    container_name: str,
) -> str:
    """Upload image bytes to the images blob container and return a SAS URL.

    The container is auto-created on first use.  The returned URL includes a
    read-only SAS token valid for 5 years so the browser can render the image
    without additional auth.
    """
    blob_service = BlobServiceClient.from_connection_string(connection_string)

    container_client = blob_service.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()
        logger.info("Created images blob container '%s'", container_name)

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        image_bytes,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )

    conn_parts = _parse_connection_string(connection_string)
    account_name = conn_parts.get("AccountName", "")
    account_key = conn_parts.get("AccountKey", "")

    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(days=365 * 5),
    )

    encoded_blob_name = quote(blob_name, safe="/")
    url = f"https://{account_name}.blob.core.windows.net/{container_name}/{encoded_blob_name}?{sas_token}"
    return url


def extract_and_replace_images(
    docx_bytes: bytes,
    source_blob_name: str,
    connection_string: str,
    container_name: str,
) -> tuple[bytes, list[str]]:
    """Extract images from a DOCX, upload each to blob storage, and replace
    the image XML with a markdown image tag containing the blob URL.

    Args:
        docx_bytes: Raw bytes of the original DOCX file.
        source_blob_name: Name/path of the source file (used as a prefix for
            uploaded image blob names to avoid collisions).
        connection_string: Azure Storage connection string.
        container_name: Blob container name for uploaded images.

    Returns:
        A tuple of (modified_docx_bytes, image_urls) where *image_urls* is
        the list of blob SAS URLs for all extracted images.
    """
    doc = Document(io.BytesIO(docx_bytes))
    image_urls: list[str] = []
    img_counter = 1

    # Cross-platform: derive the source filename from either Windows or POSIX
    # paths so local directories never leak into blob names.
    filename = re.split(r"[\\/]", source_blob_name)[-1]
    stem = os.path.splitext(filename)[0]
    unique_prefix = f"{stem}_{uuid.uuid4().hex[:8]}"

    for para_index, para in enumerate(doc.paragraphs):
        for run in para.runs:
            blips = run.element.findall(f".//{BLIP_TAG}")
            if not blips:
                continue

            for blip in blips:
                rel_id = blip.get(EMBED_ATTR)
                if not rel_id:
                    continue

                rel = doc.part.rels.get(rel_id)
                if not rel:
                    continue

                image_bytes = rel.target_part.blob
                content_type = rel.target_part.content_type
                ext = content_type.split("/")[-1].replace("jpeg", "jpg")

                blob_name = f"{unique_prefix}/IM{img_counter}.{ext}"
                url = _upload_image_to_blob(
                    image_bytes,
                    blob_name,
                    content_type,
                    connection_string,
                    container_name,
                )
                image_urls.append(url)

                run.element.clear()
                run.text = f"\n\n![Image{img_counter}]({url})\n\n"

                logger.debug(
                    "Para %d: replaced image -> ![Image%d](blob_url)",
                    para_index,
                    img_counter,
                )
                img_counter += 1

    buf = io.BytesIO()
    doc.save(buf)
    modified_bytes = buf.getvalue()

    logger.info(
        "Extracted and uploaded %d image(s) from DOCX '%s'",
        len(image_urls),
        source_blob_name,
    )
    return modified_bytes, image_urls
