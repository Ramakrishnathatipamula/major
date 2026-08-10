import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.rag.embeddings import EmbeddingManager
from modules.rag.vector_store import VectorStore

@dataclass
class SearchResult:
    """
    Structured Retrieval Result holding matching chunk text, distance/similarity score, and source metadata.
    """
    chunk_text: str
    distance: float
    similarity_score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGRetriever:
    """
    Reusable Retrieval Foundation Interface for Agricultural Knowledge RAG.
    Executes semantic vector searches with optional crop and disease metadata filters.
    """
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        self.embedding_manager = embedding_manager if embedding_manager is not None else EmbeddingManager()

    def retrieve(
        self,
        query: str,
        crop: Optional[str] = "Potato",
        disease: Optional[str] = None,
        top_k: int = 3
    ) -> List[SearchResult]:
        """
        Retrieve semantically relevant knowledge chunks for a query with metadata filtering.

        Args:
            query: Natural language query string
            crop: Optional crop filter (e.g. "Potato")
            disease: Optional disease filter (e.g. "Late Blight" or "Early Blight")
            top_k: Number of nearest neighbor results to retrieve

        Returns:
            List of SearchResult objects containing chunk text, distance, score, and source metadata.
        """
        if not query or not query.strip():
            return []

        # Construct metadata filter (ChromaDB where clause)
        where_clause = {}
        conditions = []
        if crop and crop != "all":
            conditions.append({"crop": {"$eq": crop}})

        if disease and disease != "all":
            # Support matching friendly name or target pathogen
            conditions.append({"disease": {"$eq": disease}})

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        # Generate query vector embedding
        query_vector = self.embedding_manager.embed_query(query)

        # Execute query in vector store (fallback without metadata filter if filtered query returns 0)
        results = self.vector_store.query(
            query_embedding=query_vector,
            where_clause=where_clause if where_clause else None,
            top_k=top_k
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        # If metadata filter returned 0 results, retry without metadata filter as fallback
        if len(docs) == 0 and where_clause:
            results = self.vector_store.query(
                query_embedding=query_vector,
                where_clause=None,
                top_k=top_k
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

        search_results = []
        for text, meta, dist in zip(docs, metas, dists):
            distance_val = round(float(dist), 4)
            # Similarity score for cosine distance: 1.0 - distance
            similarity = round(max(0.0, 1.0 - distance_val), 4)
            search_results.append(
                SearchResult(
                    chunk_text=text,
                    distance=distance_val,
                    similarity_score=similarity,
                    metadata=meta
                )
            )

        return search_results
