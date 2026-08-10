import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import chromadb
from chromadb.config import Settings

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings
from modules.rag.chunker import DocumentChunk

DEFAULT_COLLECTION_NAME = "agricultural_knowledge"

class VectorStore:
    """
    Persistent ChromaDB Vector Store for Agricultural Knowledge Chunks.
    Handles embedding storage, deduplication, metadata indexing, and vector query execution.
    """
    def __init__(
        self,
        persist_dir: Optional[Union[str, Path]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME
    ):
        if persist_dir is None:
            persist_dir = settings.DATA_DIR / "vector_store"

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Instantiate persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir.resolve()))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AI Crop Doctor Agricultural Knowledge Vector Store"}
        )

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add document chunks and vector embeddings into ChromaDB collection with deduplication.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: List of embedding vector float arrays

        Returns:
            Count of newly added chunks.
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatched lengths: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

        # Check existing IDs in ChromaDB for deduplication
        candidate_ids = [c.chunk_id for c in chunks]
        existing_res = self.collection.get(ids=candidate_ids)
        existing_ids = set(existing_res.get("ids", []))

        new_ids = []
        new_embeddings = []
        new_metadatas = []
        new_documents = []

        for chunk, emb in zip(chunks, embeddings):
            if chunk.chunk_id not in existing_ids:
                new_ids.append(chunk.chunk_id)
                new_embeddings.append(emb)
                new_metadatas.append(chunk.metadata)
                new_documents.append(chunk.chunk_text)

        if new_ids:
            self.collection.add(
                ids=new_ids,
                embeddings=new_embeddings,
                metadatas=new_metadatas,
                documents=new_documents
            )

        return len(new_ids)

    def count(self) -> int:
        """Return total document chunk count in collection."""
        return self.collection.count()

    def query(
        self,
        query_embedding: List[float],
        where_clause: Optional[Dict[str, Any]] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Query vector store using a query embedding vector and optional metadata filter.

        Args:
            query_embedding: Vector float array
            where_clause: Optional ChromaDB metadata filter dictionary
            top_k: Number of nearest neighbor results to return

        Returns:
            ChromaDB query results dictionary containing ids, distances, metadatas, documents.
        """
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }

        if where_clause:
            query_kwargs["where"] = where_clause

        return self.collection.query(**query_kwargs)

    def clear(self) -> None:
        """Delete all items in the collection for testing/resetting."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AI Crop Doctor Agricultural Knowledge Vector Store"}
        )
