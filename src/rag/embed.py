from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME


def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)

    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        embedded_chunks.append({**chunk, "embedding": embedding.tolist()})

    return embedded_chunks


def run_embed(chunks: list[dict]) -> list[dict]:
    model = load_embedding_model()
    return embed_chunks(chunks, model)
