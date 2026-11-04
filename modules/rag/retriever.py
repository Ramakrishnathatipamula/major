import sys
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.crops import get_crop_registry
from modules.rag.embeddings import EmbeddingManager
from modules.rag.vector_store import VectorStore


_ALL_FILTER_VALUES = {"", "all", "any"}
_MISSING_METADATA_VALUES = {"", "none", "null", "unknown", "n/a", "na"}

# Common user-facing aliases that are not crop-registry IDs or display names.
_CROP_ALIASES = {
    "corn": "maize",
    "corn (maize)": "maize",
    "corn_(maize)": "maize",
    "maize/corn": "maize",
    "paddy": "rice",
    "bell pepper": "pepper, bell",
}

# Disease words that are too generic to establish disease compatibility on their own.
_GENERIC_DISEASE_TOKENS = {
    "blight", "spot", "spots", "leaf", "leaves", "mold", "mould",
    "disease", "diseases", "pest", "pests", "management", "control",
}

_CROP_NAME_TOKENS = {
    "potato", "tomato", "maize", "corn", "rice", "wheat", "cotton",
    "mango", "banana", "apple", "blueberry", "cherry", "grape", "orange",
    "peach", "pepper", "bell", "raspberry", "soybean", "squash", "strawberry",
}


def _canonical_crop_name(crop_value: Any) -> str:
    """Return the crop registry display name for a crop id/name/alias."""
    if crop_value is None:
        return ""

    raw = str(crop_value).strip()
    if not raw:
        return ""

    lookup_key = re.sub(r"\s+", " ", raw.casefold())
    lookup_value = _CROP_ALIASES.get(lookup_key, raw)
    crop_spec = get_crop_registry().get_crop(lookup_value)
    return crop_spec.display_name if crop_spec is not None else raw


def _is_active_filter(value: Optional[str]) -> bool:
    """Return True when a crop/disease value represents a real filter."""
    if value is None:
        return False
    return str(value).strip().casefold() not in _ALL_FILTER_VALUES


def _metadata_crop_matches(metadata: Dict[str, Any], requested_crop: str) -> bool:
    """Reject missing or mismatched crop metadata for a crop-specific search."""
    if not isinstance(metadata, dict):
        return False

    metadata_crop = metadata.get("crop")
    if metadata_crop is None:
        return False

    raw_metadata_crop = str(metadata_crop).strip()
    if raw_metadata_crop.casefold() in _MISSING_METADATA_VALUES:
        return False

    return _canonical_crop_name(raw_metadata_crop).casefold() == requested_crop.casefold()


def _stem_token(token: str) -> str:
    """Apply a small, conservative singularization for disease metadata matching."""
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _disease_alternatives(value: Any) -> List[set]:
    """Convert a disease label into normalized alternative token sets."""
    if value is None:
        return []

    text = str(value).casefold().replace("___", " ")
    parts = re.split(r"\s*(?:/|,|\bor\b|\band\b)\s*", text)

    alternatives = []
    for part in parts:
        tokens = {
            _stem_token(token)
            for token in re.findall(r"[a-z0-9]+", part)
            if token not in _CROP_NAME_TOKENS
        }
        if tokens:
            alternatives.append(tokens)
    return alternatives


def _metadata_disease_matches(requested_disease: str, metadata_disease: Any) -> bool:
    """
    Check disease compatibility without requiring exact metadata equality.

    The verified store contains labels such as "Late Blight / Early Blight" and
    "Multiple (...)", so exact equality would discard valid Tomato evidence.
    """
    requested_alternatives = _disease_alternatives(requested_disease)
    metadata_alternatives = _disease_alternatives(metadata_disease)
    if not requested_alternatives:
        return True
    if not metadata_alternatives:
        return False

    metadata_tokens = set().union(*metadata_alternatives)
    for requested_tokens in requested_alternatives:
        for metadata_tokens_for_alternative in metadata_alternatives:
            if requested_tokens.issubset(metadata_tokens_for_alternative):
                return True

        # Allow diseases such as "Septoria leaf spot" to match a metadata entry
        # that only names the distinctive pathogen term "Septoria".
        distinctive_tokens = requested_tokens - _GENERIC_DISEASE_TOKENS
        if distinctive_tokens and distinctive_tokens & metadata_tokens:
            return True

    return False

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

        if top_k <= 0:
            return []

        crop_specific = _is_active_filter(crop)
        disease_specific = _is_active_filter(disease)

        canonical_crop = _canonical_crop_name(crop) if crop_specific else None

        # Crop is enforced in the vector-store query AND re-checked below. Disease
        # cannot be safely enforced with an exact Chroma clause because verified
        # metadata may contain combined labels (e.g. "Late Blight / Early Blight").
        where_clause = (
            {"crop": {"$eq": canonical_crop}}
            if crop_specific
            else None
        )

        # Generate query vector embedding
        query_vector = self.embedding_manager.embed_query(query)

        # Retrieve a wider candidate pool when disease filtering is requested so
        # compatible chunks are not lost before post-filtering.
        candidate_k = max(top_k, top_k * 5) if disease_specific else top_k

        # Execute exactly one query. There is deliberately NO unfiltered retry:
        # a crop-specific search that returns no compatible evidence must return
        # an empty result rather than cross-crop knowledge.
        results = self.vector_store.query(
            query_embedding=query_vector,
            where_clause=where_clause,
            top_k=candidate_k
        )

        docs = (results.get("documents") or [[]])[0] or []
        metas = (results.get("metadatas") or [[]])[0] or []
        dists = (results.get("distances") or [[]])[0] or []

        search_results = []
        for text, meta, dist in zip(docs, metas, dists):
            metadata = meta if isinstance(meta, dict) else {}

            # Enforce crop compatibility on every result, even if a vector store
            # implementation ignores or partially honors the metadata filter.
            if crop_specific and not _metadata_crop_matches(metadata, canonical_crop):
                continue

            if disease_specific and not _metadata_disease_matches(disease, metadata.get("disease")):
                continue

            distance_val = round(float(dist), 4)
            # Similarity score for cosine distance: 1.0 - distance
            similarity = round(max(0.0, 1.0 - distance_val), 4)
            search_results.append(
                SearchResult(
                    chunk_text=text,
                    distance=distance_val,
                    similarity_score=similarity,
                    metadata=metadata
                )
            )

            if len(search_results) >= top_k:
                break

        return search_results
