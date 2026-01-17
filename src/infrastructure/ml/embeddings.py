from sentence_transformers import SentenceTransformer
from src.domain.agents.ports import EmbeddingService
from src.config.settings import settings


class LocalHuggingFaceEmbeddings(EmbeddingService):
    """Local embedding service using HuggingFace models."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_PATH
        self.model = SentenceTransformer(self.model_name)

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]
