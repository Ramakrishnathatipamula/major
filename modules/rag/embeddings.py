import sys
from pathlib import Path
from typing import List, Optional
from sentence_transformers import SentenceTransformer

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

class EmbeddingManager:
    """
    Reusable Vector Embedding Manager using SentenceTransformers (BAAI/bge-small-en-v1.5).
    """
    _model_instance: Optional[SentenceTransformer] = None

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = self._get_or_load_model(self.model_name)

    @classmethod
    def _get_or_load_model(cls, model_name: str) -> SentenceTransformer:
        """Load and cache SentenceTransformer model instance in memory."""
        if cls._model_instance is None:
            cls._model_instance = SentenceTransformer(model_name)
        return cls._model_instance

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of text strings.

        Args:
            texts: List of text chunk strings

        Returns:
            List of float vector arrays.
        """
        if not texts:
            return []

        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding vector for a search query string.

        Args:
            query: User query string

        Returns:
            Vector float array.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        embedding = self.model.encode(query, show_progress_bar=False, normalize_embeddings=True)
        return embedding.tolist()
