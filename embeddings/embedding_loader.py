from sentence_transformers import SentenceTransformer
from config.config import EMBEDDING_MODEL

def load_embeddings():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    return embedder
