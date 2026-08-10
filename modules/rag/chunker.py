import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.rag.document_loader import AgriculturalDocument

@dataclass
class DocumentChunk:
    """
    Represents a processed text chunk with attached source metadata and deterministic ID.
    """
    chunk_id: str
    chunk_text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentChunker:
    """
    Splits agricultural documents into semantically coherent text chunks (~500 chars, ~100 overlap),
    preserving paragraph/sentence boundaries and metadata.
    """
    def __init__(self, target_chunk_size: int = 500, overlap: int = 100):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: AgriculturalDocument) -> List[DocumentChunk]:
        """
        Chunk an AgriculturalDocument into metadata-annotated DocumentChunks.

        Args:
            doc: AgriculturalDocument object

        Returns:
            List of DocumentChunk objects with metadata preservation.
        """
        raw_text = doc.content.strip()
        if not raw_text:
            return []

        # Split into logical paragraphs first to preserve boundaries
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [raw_text]

        chunks_text = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= self.target_chunk_size:
                current_chunk = f"{current_chunk}\n\n{paragraph}".strip()
            else:
                if current_chunk:
                    chunks_text.append(current_chunk)

                # If paragraph itself is larger than chunk size, split by sentences
                if len(paragraph) > self.target_chunk_size:
                    sentences = [s.strip() + "." for s in paragraph.split(". ") if s.strip()]
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) + 1 <= self.target_chunk_size:
                            temp_chunk = f"{temp_chunk} {sentence}".strip()
                        else:
                            if temp_chunk:
                                chunks_text.append(temp_chunk)
                            temp_chunk = sentence
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks_text.append(current_chunk)

        # Apply overlap where appropriate
        final_chunks_text = []
        for i, text in enumerate(chunks_text):
            if i > 0 and len(chunks_text[i - 1]) >= self.overlap:
                overlap_prefix = chunks_text[i - 1][-self.overlap:]
                combined = f"{overlap_prefix}... {text}".strip()
                final_chunks_text.append(combined)
            else:
                final_chunks_text.append(text)

        # Build DocumentChunk objects with metadata and deterministic IDs
        document_chunks = []
        base_meta = doc.to_dict()
        base_meta.pop("content", None)  # Remove full content string from metadata

        for idx, chunk_str in enumerate(final_chunks_text):
            # Compute deterministic ID based on doc title + chunk index + content hash
            hash_input = f"{doc.document_title}_{idx}_{chunk_str}".encode("utf-8")
            chunk_id = hashlib.sha256(hash_input).hexdigest()[:16]

            # Clone metadata and attach chunk sequence index
            chunk_meta = dict(base_meta)
            chunk_meta["chunk_index"] = idx
            chunk_meta["chunk_count"] = len(final_chunks_text)

            # Ensure metadata values are JSON/ChromaDB compatible (strings, ints, floats, bools)
            clean_meta = {}
            for k, v in chunk_meta.items():
                if v is None:
                    clean_meta[k] = "none"
                else:
                    clean_meta[k] = v

            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk_str,
                    metadata=clean_meta
                )
            )

        return document_chunks
