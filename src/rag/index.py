import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME


def build_collection(chunks: list[dict]):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[
            {
                "page_number": chunk["page_number"],
                "page_chunk_index": chunk["page_chunk_index"],
                "token_count": chunk["token_count"],
            }
            for chunk in chunks
        ],
    )
    return collection


def run_index(chunks: list[dict]):
    return build_collection(chunks)
