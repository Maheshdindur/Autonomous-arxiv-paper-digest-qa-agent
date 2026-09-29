"""Chroma local persistent vector database wrapper.

Handles:
- Persistent storage of chunk embeddings and text in local directory
- Preservation of chunk metadata (paper_id, chunk_id, page_start, page_end, section, word_count)
- Top-K similarity search returning chunk content, structured metadata, and similarity scores
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from src.config import load_config
from src.graph.state import ChunkMetadata, DocumentChunk
from src.utils.embedding import embed_query, embed_texts

logger = logging.getLogger("arxiv_agent.vector_store")


def sanitize_collection_name(paper_id: str) -> str:
    """Sanitize paper ID to comply with Chroma collection naming rules.

    Chroma rules: 3-63 characters, letters, numbers, underscores, hyphens,
    must start and end with alphanumeric character.
    """
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", paper_id)
    # Ensure starts and ends with alphanumeric
    clean = clean.strip("_-")
    name = f"paper_{clean}"
    if len(name) > 63:
        name = name[:63].rstrip("_-")
    return name


def get_chroma_client(persist_dir: Optional[Path] = None) -> ClientAPI:
    """Initialize or return a persistent Chroma client pointing to local directory."""
    if persist_dir is None:
        config = load_config(require_llm_key=False)
        persist_dir = config.chroma_persist_dir

    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_or_create_paper_collection(
    paper_id: str,
    client: Optional[ClientAPI] = None,
) -> Collection:
    """Get or create a dedicated Chroma collection for a paper with cosine distance."""
    if client is None:
        client = get_chroma_client()

    col_name = sanitize_collection_name(paper_id)
    # HNSW space configured for cosine similarity
    return client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_paper_chunks(
    paper_id: str,
    chunks: List[DocumentChunk],
    client: Optional[ClientAPI] = None,
    embedding_model: Optional[str] = None,
) -> int:
    """Generate embeddings and store chunks + metadata in Chroma.

    Args:
        paper_id: arXiv identifier.
        chunks: List of DocumentChunk dicts.
        client: Optional Chroma client.
        embedding_model: Optional embedding model name.

    Returns:
        Number of chunks successfully indexed.
    """
    if not chunks:
        logger.warning("No chunks provided for paper '%s' to index.", paper_id)
        return 0

    collection = get_or_create_paper_collection(paper_id, client=client)

    texts = [c["text"] for c in chunks]
    logger.info("Generating embeddings for %d chunks of paper '%s'...", len(chunks), paper_id)
    embeddings = embed_texts(texts, model_name=embedding_model, normalize=True)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        chunk_id = meta.get("chunk_id") or f"{paper_id}_c{idx:04d}"
        ids.append(chunk_id)
        documents.append(chunk["text"])

        # Chroma metadata requires primitive types (str, int, float, bool).
        # None is stored as empty string and restored on retrieval.
        chroma_meta = {
            "paper_id": str(meta.get("paper_id", paper_id)),
            "chunk_id": str(chunk_id),
            "page_start": int(meta.get("page_start", 1)),
            "page_end": int(meta.get("page_end", 1)),
            "section": str(meta.get("section") or ""),
            "word_count": int(meta.get("word_count", len(chunk["text"].split()))),
        }
        metadatas.append(chroma_meta)

    # Upsert into Chroma (batch size 100)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    logger.info("Successfully indexed %d chunks for paper '%s' into collection '%s'", len(ids), paper_id, collection.name)
    return len(ids)


def query_similar_chunks(
    paper_id: str,
    query: str,
    top_k: int = 4,
    client: Optional[ClientAPI] = None,
    embedding_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query Chroma for the top-K most semantically similar chunks.

    Args:
        paper_id: arXiv identifier for the target paper.
        query: User question or search query string.
        top_k: Number of chunks to retrieve (configurable).
        client: Optional Chroma client.
        embedding_model: Optional embedding model name.

    Returns:
        List of dicts containing retrieved chunk text, metadata, distance, and similarity_score.
    """
    collection = get_or_create_paper_collection(paper_id, client=client)

    count = collection.count()
    if count == 0:
        logger.warning("Collection for paper '%s' is empty.", paper_id)
        return []

    # Adjust top_k if collection has fewer items than k
    effective_k = min(top_k, count)

    query_vec = embed_query(query, model_name=embedding_model, normalize=True)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_items: List[Dict[str, Any]] = []

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for idx in range(len(docs)):
        meta = metas[idx]
        section = meta.get("section")
        # Restore empty string to None
        if not section:
            section = None

        chunk_meta: ChunkMetadata = {
            "chunk_id": str(meta.get("chunk_id", ids[idx])),
            "paper_id": str(meta.get("paper_id", paper_id)),
            "page_start": int(meta.get("page_start", 1)),
            "page_end": int(meta.get("page_end", 1)),
            "section": section,
            "word_count": int(meta.get("word_count", 0)),
        }

        dist = float(distances[idx])
        # For cosine distance (range 0.0 to 2.0), similarity is 1.0 - dist
        similarity = max(0.0, min(1.0, 1.0 - dist))

        retrieved_items.append({
            "chunk_id": ids[idx],
            "text": docs[idx],
            "metadata": chunk_meta,
            "distance": dist,
            "similarity_score": similarity,
        })

    logger.info("Retrieved %d chunks for paper '%s' (top score: %.4f)", len(retrieved_items), paper_id, retrieved_items[0]["similarity_score"] if retrieved_items else 0.0)
    return retrieved_items
