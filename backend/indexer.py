from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import boto3
import chromadb
import requests
from botocore.exceptions import BotoCoreError, ClientError
from requests.exceptions import RequestException


# =========================================================
# Project configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local folder where ChromaDB stores vectors and documents.
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# IMPORTANT:
# This collection name must also be used in main.py.
COLLECTION_NAME = "medical_knowledge"

# Delete and rebuild the collection every time indexer.py runs.
# This prevents duplicate or outdated chunks.
RESET_COLLECTION = True


# =========================================================
# AWS S3 configuration
# =========================================================

AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "medical-rag-manideep-dev"
S3_DATASET_PREFIX = "datasets/"


# =========================================================
# Ollama embedding configuration
# =========================================================

OLLAMA_BASE_URL = "http://localhost:11434"

# Official Ollama embedding endpoint.
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"

EMBEDDING_MODEL = "nomic-embed-text"

# Number of chunks sent to Ollama in one request.
EMBEDDING_BATCH_SIZE = 16

# Timeout for one Ollama embedding request.
OLLAMA_TIMEOUT_SECONDS = 180


# =========================================================
# Chunking configuration
# =========================================================

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 210

# Ignore extremely small chunks.
MINIMUM_CHUNK_SIZE = 100

# Number of records inserted into ChromaDB at one time.
CHROMA_BATCH_SIZE = 100


# =========================================================
# AWS and ChromaDB clients
# =========================================================

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DB_PATH),
)


# =========================================================
# S3 functions
# =========================================================

def verify_s3_access() -> None:
    """
    Verify that the configured AWS credentials can access
    the required S3 bucket.
    """

    try:
        s3_client.head_bucket(
            Bucket=S3_BUCKET_NAME
        )

        print(
            f"Connected to S3 bucket: "
            f"{S3_BUCKET_NAME}"
        )

    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(
            "Unable to access the S3 bucket. "
            "Verify your AWS credentials, bucket name, "
            "region, and IAM permissions."
        ) from error


def list_s3_documents() -> list[str]:
    """
    Return every .txt object under datasets/ in S3.

    A paginator is used so the script continues to work
    even when the bucket contains more than 1,000 objects.
    """

    document_keys: list[str] = []

    try:
        paginator = s3_client.get_paginator(
            "list_objects_v2"
        )

        pages = paginator.paginate(
            Bucket=S3_BUCKET_NAME,
            Prefix=S3_DATASET_PREFIX,
        )

        for page in pages:
            for item in page.get("Contents", []):
                key = item.get("Key", "")

                if (
                    key.lower().endswith(".txt")
                    and not key.endswith("/")
                ):
                    document_keys.append(key)

    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(
            "Failed to list dataset objects from S3."
        ) from error

    return sorted(document_keys)


def download_s3_document(s3_key: str) -> str:
    """
    Download one S3 text object directly into memory.

    No permanent local file is created.
    """

    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
        )

        raw_content = response["Body"].read()

        return raw_content.decode(
            "utf-8",
            errors="replace",
        )

    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(
            f"Failed to download S3 object: {s3_key}"
        ) from error


# =========================================================
# Document parsing functions
# =========================================================

def clean_text(text: str) -> str:
    """
    Remove unnecessary spaces, tabs, and repeated line breaks.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Keep paragraph boundaries while cleaning spaces.
    paragraphs = []

    for paragraph in re.split(r"\n\s*\n", text):
        cleaned_paragraph = re.sub(
            r"\s+",
            " ",
            paragraph,
        ).strip()

        if cleaned_paragraph:
            paragraphs.append(cleaned_paragraph)

    return "\n\n".join(paragraphs)


def extract_header_value(
    document_text: str,
    header_name: str,
) -> str:
    """
    Extract a value such as Title, Source URL, or Topic
    from the beginning of the document.

    Example:
        Title: Diabetes
    """

    pattern = rf"^{re.escape(header_name)}\s*:\s*(.+)$"

    match = re.search(
        pattern,
        document_text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return ""


def parse_document(
    document_text: str,
    s3_key: str,
) -> dict[str, str]:
    """
    Separate document metadata from the medical content.

    Expected document structure:

        Title: ...
        Source URL: ...
        Topic: ...

        --- CONTENT ---

        Medical text...
    """

    title = extract_header_value(
        document_text,
        "Title",
    )

    source_url = extract_header_value(
        document_text,
        "Source URL",
    )

    topic = extract_header_value(
        document_text,
        "Topic",
    )

    content_marker = "--- CONTENT ---"

    if content_marker in document_text:
        content = document_text.split(
            content_marker,
            1,
        )[1]
    else:
        # Fallback for documents without the expected marker.
        content = document_text

    content = clean_text(content)

    # Derive topic from the S3 key when the Topic header is missing.
    # Example:
    # datasets/diabetes/file.txt
    if not topic:
        key_parts = s3_key.split("/")

        if len(key_parts) >= 3:
            topic = key_parts[1]

    if not title:
        title = Path(s3_key).stem.replace(
            "_",
            " ",
        ).title()

    return {
        "title": title or "Untitled",
        "source_url": source_url or "Unknown",
        "topic": topic or "unknown",
        "content": content,
        "s3_key": s3_key,
    }


# =========================================================
# Chunking functions
# =========================================================

def find_chunk_end(
    text: str,
    start: int,
    desired_end: int,
) -> int:
    """
    Try to finish a chunk at a natural boundary such as
    a paragraph, sentence, or space.
    """

    if desired_end >= len(text):
        return len(text)

    search_start = max(
        start + int(CHUNK_SIZE * 0.6),
        start,
    )

    section = text[search_start:desired_end]

    # Try paragraph boundary first.
    paragraph_position = section.rfind("\n\n")

    if paragraph_position != -1:
        return search_start + paragraph_position + 2

    # Then try sentence endings.
    sentence_positions = [
        section.rfind(". "),
        section.rfind("? "),
        section.rfind("! "),
    ]

    best_sentence_position = max(sentence_positions)

    if best_sentence_position != -1:
        return search_start + best_sentence_position + 1

    # Finally, try a normal space.
    space_position = section.rfind(" ")

    if space_position != -1:
        return search_start + space_position

    return desired_end


def chunk_text(text: str) -> list[str]:
    """
    Split a document into overlapping chunks.

    Overlap helps preserve information that appears near
    the boundary between two chunks.
    """

    if CHUNK_SIZE <= 0:
        raise ValueError(
            "CHUNK_SIZE must be greater than zero."
        )

    if CHUNK_OVERLAP < 0:
        raise ValueError(
            "CHUNK_OVERLAP cannot be negative."
        )

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    text = clean_text(text)

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        desired_end = min(
            start + CHUNK_SIZE,
            text_length,
        )

        end = find_chunk_end(
            text=text,
            start=start,
            desired_end=desired_end,
        )

        # Safety protection against an invalid boundary.
        if end <= start:
            end = desired_end

        chunk = text[start:end].strip()

        if len(chunk) >= MINIMUM_CHUNK_SIZE:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - CHUNK_OVERLAP

        # Prevent an endless loop.
        if next_start <= start:
            next_start = start + (
                CHUNK_SIZE - CHUNK_OVERLAP
            )

        start = next_start

    return chunks


# =========================================================
# Ollama embedding functions
# =========================================================

def verify_ollama() -> None:
    """
    Confirm that Ollama is running and that the embedding
    model is available.
    """

    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        available_models = {
            model.get("name", "")
            for model in result.get("models", [])
        }

        model_is_available = any(
            model_name == EMBEDDING_MODEL
            or model_name.startswith(
                f"{EMBEDDING_MODEL}:"
            )
            for model_name in available_models
        )

        if not model_is_available:
            raise RuntimeError(
                f"Ollama is running, but the model "
                f"'{EMBEDDING_MODEL}' is not installed.\n"
                f"Run: ollama pull {EMBEDDING_MODEL}"
            )

        print(
            f"Connected to Ollama embedding model: "
            f"{EMBEDDING_MODEL}"
        )

    except RequestException as error:
        raise RuntimeError(
            "Ollama is not reachable at "
            f"{OLLAMA_BASE_URL}.\n"
            "Start Ollama and try again."
        ) from error


def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embedding vectors for a batch of text chunks
    using Ollama.
    """

    if not texts:
        return []

    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBEDDING_MODEL,
                "input": texts,
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        response_data = response.json()
        embeddings = response_data.get(
            "embeddings",
            [],
        )

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Ollama returned an unexpected number "
                "of embeddings."
            )

        return embeddings

    except RequestException as error:
        raise RuntimeError(
            "Failed to generate embeddings through Ollama."
        ) from error


# =========================================================
# ChromaDB functions
# =========================================================

def create_or_reset_collection():
    """
    Create the ChromaDB collection.

    When RESET_COLLECTION is True, the previous collection
    is removed and rebuilt from the S3 dataset.
    """

    if RESET_COLLECTION:
        try:
            chroma_client.delete_collection(
                name=COLLECTION_NAME
            )

            print(
                f"Deleted existing Chroma collection: "
                f"{COLLECTION_NAME}"
            )

        except Exception:
            # The collection may not exist during the first run.
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Medical knowledge chunks loaded from "
                "Amazon S3 and embedded with Ollama"
            ),
            "embedding_model": EMBEDDING_MODEL,
            "hnsw:space": "cosine",
        },
    )

    return collection


def create_chunk_id(
    s3_key: str,
    chunk_number: int,
) -> str:
    """
    Create a consistent unique ID for one chunk.
    """

    raw_id = f"{s3_key}::chunk::{chunk_number}"

    return hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()


def store_batch_in_chroma(
    collection: Any,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    """
    Store one batch of records in ChromaDB.
    """

    if not ids:
        return

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# =========================================================
# Main indexing process
# =========================================================

def build_index() -> None:
    """
    Complete indexing pipeline:

    S3
      -> documents
      -> chunks
      -> Ollama embeddings
      -> ChromaDB
    """

    print("=" * 60)
    print("Medical RAG is S3 bucket Indexing Pipeline")
    print("=" * 60)

    verify_s3_access()
    verify_ollama()

    print(
        f"ChromaDB location: {CHROMA_DB_PATH}"
    )

    collection = create_or_reset_collection()

    s3_keys = list_s3_documents()

    if not s3_keys:
        raise RuntimeError(
            f"No .txt documents were found under "
            f"s3://{S3_BUCKET_NAME}/"
            f"{S3_DATASET_PREFIX}"
        )

    print(
        f"Found {len(s3_keys)} documents in S3."
    )

    total_documents = 0
    total_chunks = 0
    skipped_documents = 0
    failed_documents = 0

    # Temporary Chroma batch containers.
    pending_ids: list[str] = []
    pending_documents: list[str] = []
    pending_embeddings: list[list[float]] = []
    pending_metadatas: list[dict[str, Any]] = []

    for document_position, s3_key in enumerate(
        s3_keys,
        start=1,
    ):
        print()
        print(
            f"[{document_position}/{len(s3_keys)}] "
            f"Reading: {s3_key}"
        )

        try:
            raw_document = download_s3_document(
                s3_key
            )

            parsed_document = parse_document(
                document_text=raw_document,
                s3_key=s3_key,
            )

            content = parsed_document["content"]

            if len(content) < MINIMUM_CHUNK_SIZE:
                print(
                    "Skipped: document content is too short."
                )

                skipped_documents += 1
                continue

            chunks = chunk_text(content)

            if not chunks:
                print(
                    "Skipped: no usable chunks were created."
                )

                skipped_documents += 1
                continue

            print(
                f"Created {len(chunks)} chunks."
            )

            # Generate embeddings in smaller batches.
            for batch_start in range(
                0,
                len(chunks),
                EMBEDDING_BATCH_SIZE,
            ):
                batch_end = min(
                    batch_start
                    + EMBEDDING_BATCH_SIZE,
                    len(chunks),
                )

                chunk_batch = chunks[
                    batch_start:batch_end
                ]

                print(
                    f"Embedding chunks "
                    f"{batch_start + 1}-{batch_end}..."
                )

                embedding_batch = generate_embeddings(
                    chunk_batch
                )

                for local_index, (
                    chunk,
                    embedding,
                ) in enumerate(
                    zip(
                        chunk_batch,
                        embedding_batch,
                    )
                ):
                    chunk_number = (
                        batch_start
                        + local_index
                        + 1
                    )

                    chunk_id = create_chunk_id(
                        s3_key=s3_key,
                        chunk_number=chunk_number,
                    )

                    metadata = {
                        "topic": parsed_document["topic"],
                        "title": parsed_document["title"],
                        "source_url": (
                            parsed_document["source_url"]
                        ),
                        "s3_bucket": S3_BUCKET_NAME,
                        "s3_key": s3_key,
                        "chunk_number": chunk_number,
                        "total_chunks": len(chunks),
                        "embedding_model": (
                            EMBEDDING_MODEL
                        ),
                    }

                    pending_ids.append(chunk_id)
                    pending_documents.append(chunk)
                    pending_embeddings.append(embedding)
                    pending_metadatas.append(metadata)

                    # Store records periodically instead of
                    # holding every vector in memory.
                    if (
                        len(pending_ids)
                        >= CHROMA_BATCH_SIZE
                    ):
                        store_batch_in_chroma(
                            collection=collection,
                            ids=pending_ids,
                            documents=pending_documents,
                            embeddings=pending_embeddings,
                            metadatas=pending_metadatas,
                        )

                        print(
                            f"Stored batch of "
                            f"{len(pending_ids)} chunks "
                            f"in ChromaDB."
                        )

                        pending_ids.clear()
                        pending_documents.clear()
                        pending_embeddings.clear()
                        pending_metadatas.clear()

            total_documents += 1
            total_chunks += len(chunks)

            print("Document indexed successfully.")

        except Exception as error:
            failed_documents += 1

            print(
                f"Failed to index: {s3_key}"
            )
            print(
                f"Reason: {error}"
            )

    # Store records remaining after the final complete batch.
    if pending_ids:
        store_batch_in_chroma(
            collection=collection,
            ids=pending_ids,
            documents=pending_documents,
            embeddings=pending_embeddings,
            metadatas=pending_metadatas,
        )

        print()
        print(
            f"Stored final batch of "
            f"{len(pending_ids)} chunks in ChromaDB."
        )

    stored_record_count = collection.count()

    print()
    print("=" * 60)
    print("Indexing completed")
    print("=" * 60)
    print(
        f"Documents found in S3: {len(s3_keys)}"
    )
    print(
        f"Documents indexed: {total_documents}"
    )
    print(
        f"Documents skipped: {skipped_documents}"
    )
    print(
        f"Documents failed: {failed_documents}"
    )
    print(
        f"Chunks created: {total_chunks}"
    )
    print(
        f"Records stored in ChromaDB: "
        f"{stored_record_count}"
    )
    print(
        f"Collection name: {COLLECTION_NAME}"
    )
    print(
        f"ChromaDB path: {CHROMA_DB_PATH}"
    )


# =========================================================
# Script entry point
# =========================================================

if __name__ == "__main__":
    try:
        build_index()

    except KeyboardInterrupt:
        print(
            "\nIndexing stopped by the user."
        )
        sys.exit(1)

    except Exception as error:
        print()
        print("Indexing could not be completed.")
        print(f"Reason: {error}")
        sys.exit(1)