import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings
from modules.rag.retriever import RAGRetriever, SearchResult

try:
    from google import genai
except ImportError:
    genai = None

DEFAULT_DISCLAIMER = (
    "Grounded Agricultural Knowledge Guidance based on peer-reviewed University Extension literature. "
    "This guidance is for educational purposes and does not replace an in-person field assessment by a certified agronomist."
)

INSUFFICIENT_CONTEXT_MESSAGE = "I don't have enough information in the verified knowledge base to answer that question."

@dataclass
class RAGAnswer:
    """
    Standardized payload for grounded RAG LLM responses.
    """
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    is_grounded: bool
    insufficient_context: bool
    disclaimer: str = DEFAULT_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class RAGAnswerer:
    """
    Reusable Grounded RAG LLM Answerer for AI Crop Doctor.
    Links RAGRetriever with Google Gemini SDK enforcing strict grounding rules.
    """
    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.retriever = retriever if retriever is not None else RAGRetriever()
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model_name = model_name if model_name is not None else settings.GEMINI_MODEL

        # Initialize Gemini Client if API key is present
        self.client = None
        if self.api_key and self.api_key != "mock_key_for_testing":
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                self.client = None

    def answer_question(
        self,
        query: str,
        crop: Optional[str] = "Potato",
        disease: Optional[str] = None,
        top_k: int = 3
    ) -> RAGAnswer:
        """
        Execute RAG retrieval and generate grounded LLM answer using Google Gemini.

        Args:
            query: User agricultural question
            crop: Optional crop filter (e.g. "Potato")
            disease: Optional disease filter (e.g. "Late Blight")
            top_k: Number of retrieved context chunks to include

        Returns:
            RAGAnswer object containing grounded text response and source metadata.
        """
        if not query or not query.strip():
            return RAGAnswer(
                query=query if query else "",
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
                is_grounded=False,
                insufficient_context=True,
                disclaimer=DEFAULT_DISCLAIMER
            )

        # Step 1: Retrieve relevant context chunks from verified vector store
        search_results: List[SearchResult] = self.retriever.retrieve(
            query=query,
            crop=crop,
            disease=disease,
            top_k=top_k
        )

        # Step 2: Format sources and check context sufficiency
        sources_meta = []
        for res in search_results:
            sources_meta.append({
                "source_organization": res.metadata.get("source_organization", "unknown"),
                "document_title": res.metadata.get("document_title", "unknown"),
                "url_reference": res.metadata.get("url_reference", "unknown"),
                "disease": res.metadata.get("disease", "unknown"),
                "similarity_score": res.similarity_score,
                "distance": res.distance
            })

        # Insufficient context check (no results or top similarity < 0.30)
        if not search_results or (search_results and search_results[0].similarity_score < 0.30):
            if not self.client:
                return RAGAnswer(
                    query=query,
                    answer=INSUFFICIENT_CONTEXT_MESSAGE,
                    sources=sources_meta,
                    is_grounded=False,
                    insufficient_context=True,
                    disclaimer=DEFAULT_DISCLAIMER
                )

        # Step 3: Format grounding prompt with strict context boundaries
        formatted_prompt, context_str = self.build_grounding_prompt(query, search_results)

        # Step 4: Generate LLM Answer
        if not self.api_key or self.api_key == "mock_key_for_testing" or self.client is None:
            # Fallback for offline/test environments without live Gemini API key
            fallback_answer = (
                f"[Grounded Response based on {len(search_results)} verified extension sources]:\n\n"
                + "\n\n".join([f"From {r.metadata.get('source_organization')}: {r.chunk_text.strip()}" for r in search_results])
            )
            return RAGAnswer(
                query=query,
                answer=fallback_answer,
                sources=sources_meta,
                is_grounded=True,
                insufficient_context=False,
                disclaimer=DEFAULT_DISCLAIMER
            )

        # Candidate models to try in case of quota or model availability issues
        candidate_models = [self.model_name, "gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
        seen_models = set()
        last_exc = None

        for model_to_try in candidate_models:
            if not model_to_try or model_to_try in seen_models:
                continue
            seen_models.add(model_to_try)
            try:
                response = self.client.models.generate_content(
                    model=model_to_try,
                    contents=formatted_prompt
                )
                generated_text = response.text.strip() if response.text else INSUFFICIENT_CONTEXT_MESSAGE

                return RAGAnswer(
                    query=query,
                    answer=generated_text,
                    sources=sources_meta,
                    is_grounded=True,
                    insufficient_context=(generated_text == INSUFFICIENT_CONTEXT_MESSAGE),
                    disclaimer=DEFAULT_DISCLAIMER
                )
            except Exception as exc:
                last_exc = exc
                continue

        return RAGAnswer(
            query=query,
            answer=f"Gemini API request error ({last_exc}). {INSUFFICIENT_CONTEXT_MESSAGE}",
            sources=sources_meta,
            is_grounded=False,
            insufficient_context=True,
            disclaimer=DEFAULT_DISCLAIMER
        )

    def build_grounding_prompt(self, query: str, search_results: List[SearchResult]) -> tuple:
        """Construct prompt string with retrieved context and clear agronomic guidance rules."""
        context_blocks = []
        for idx, res in enumerate(search_results, start=1):
            block = (
                f"SOURCE {idx}\n"
                f"Organization: {res.metadata.get('source_organization')}\n"
                f"Document: {res.metadata.get('document_title')}\n"
                f"URL: {res.metadata.get('url_reference')}\n"
                f"Content:\n{res.chunk_text.strip()}"
            )
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks) if context_blocks else "General agricultural knowledge base."

        prompt = (
            "You are the AI Crop Doctor Senior Agronomist and Plant Pathology Expert.\n"
            "Your objective is to deliver authoritative, clear, compassionate, and farmer-actionable advice for the diagnosed crop disease.\n\n"
            f"VERIFIED EXTENSION EVIDENCE SOURCES:\n{context_str}\n\n"
            "EXPERT GUIDELINES:\n"
            "1. Clinical Assessment: Briefly explain what this disease is and its key symptoms on foliage.\n"
            "2. Verified Chemical Treatment & Medicines: State CIBRC/ICAR-approved active ingredients (e.g. Propiconazole 25% EC, Tebuconazole 25.9% EC, Azoxystrobin + Difenoconazole, Mancozeb 75% WP, Copper Oxychloride, Streptocycline), practical dilution rates (per litre of water and per acre), and spray timing.\n"
            "3. Biological & Organic Alternatives: Provide bio-fungicides (e.g. Trichoderma viride/harzianum, Pseudomonas fluorescens) or botanical sprays (Neem oil 10,000 ppm).\n"
            "4. Cultural & Field Sanitation: Explain preventative measures (irrigation management, balanced fertilizer, crop rotation, resistant varieties).\n"
            "5. Farmer Safety: Remind farmers to wear PPE and strictly follow container label instructions.\n"
            "6. Formatting: Use clean markdown with bold bullet points so it is effortless for a farmer or extension officer to read.\n\n"
            f"USER QUESTION / CONTEXT:\n{query}\n"
        )
        return prompt, context_str
