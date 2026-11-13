import hashlib
import logging

import chromadb

from core.config import CHROMA_DIR
from rag.embeddings import embed_texts


logger = logging.getLogger(__name__)
COLLECTION_NAME = "finance_knowledge"


def build_or_get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def add_knowledge_documents(documents):
    if documents is None:
        return []

    texts = []
    sources = []
    ids = []
    for document in documents:
        if isinstance(document, dict):
            text = str(document.get("text", "")).strip()
            source = str(document.get("source", "finance_guidance"))
        else:
            text = str(document).strip()
            source = "finance_guidance"
        if not text:
            continue
        texts.append(text)
        sources.append(source)
        ids.append(hashlib.sha256(f"{source}:{text}".encode("utf-8")).hexdigest())

    if not texts:
        return []
    _, collection = build_or_get_collection()
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embed_texts(texts),
        metadatas=[{"source": source} for source in sources],
    )
    return texts


def query_collection(question, n_results=5, where=None):
    _, collection = build_or_get_collection()
    if where:
        count = collection.count(where=where)
    else:
        count = collection.count()
    if count == 0:
        return {"documents": [[]]}
    docs = collection.query(
        query_embeddings=embed_texts([question]),
        n_results=min(max(int(n_results), 1), count),
        where=where,
    )
    return docs
