"""
RAG Knowledge Base & Retrieval System for AI Crop Doctor.
Provides document loading, semantic chunking, vector embedding, persistent ChromaDB indexing,
metadata-filtered retrieval, and grounded LLM answer generation.
"""

from .document_loader import DocumentLoader, AgriculturalDocument
from .chunker import DocumentChunker, DocumentChunk
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .retriever import RAGRetriever, SearchResult
from .answerer import RAGAnswerer, RAGAnswer

__all__ = [
    "DocumentLoader",
    "AgriculturalDocument",
    "DocumentChunker",
    "DocumentChunk",
    "EmbeddingManager",
    "VectorStore",
    "RAGRetriever",
    "SearchResult",
    "RAGAnswerer",
    "RAGAnswer",
]
