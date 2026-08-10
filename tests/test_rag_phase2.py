import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from modules.rag import (
    RAGAnswerer,
    RAGAnswer,
    RAGRetriever,
    SearchResult,
)

class TestRAGPhase2(unittest.TestCase):

    def setUp(self):
        self.mock_retriever = MagicMock(spec=RAGRetriever)
        self.sample_results = [
            SearchResult(
                chunk_text="Late blight initial symptoms appear as small, water-soaked dark green or brown spots on lower leaves.",
                distance=0.35,
                similarity_score=0.65,
                metadata={
                    "source_organization": "University of Minnesota Extension",
                    "document_title": "Late blight of tomato and potato",
                    "crop": "Potato",
                    "disease": "Late Blight",
                    "url_reference": "https://extension.umn.edu/disease-management/late-blight"
                }
            )
        ]
        self.mock_retriever.retrieve.return_value = self.sample_results

    @patch("modules.rag.answerer.genai")
    def test_1_successful_grounded_answer_generation(self, mock_genai):
        """Test 1: Successful grounded answer generation with mocked Gemini SDK."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Potato late blight causes small water-soaked dark spots on lower leaves [Source: UMN Extension]."
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        answerer = RAGAnswerer(
            retriever=self.mock_retriever,
            api_key="mock_key_12345",
            model_name="gemini-2.5-flash"
        )
        answerer.client = mock_client

        res = answerer.answer_question("What are the symptoms of potato late blight?", crop="Potato", disease="Late Blight")

        self.assertIsInstance(res, RAGAnswer)
        self.assertTrue(res.is_grounded)
        self.assertFalse(res.insufficient_context)
        self.assertIn("water-soaked dark spots", res.answer)

    def test_2_retrieved_sources_included(self):
        """Test 2: Retrieved sources are included in payload."""
        answerer = RAGAnswerer(
            retriever=self.mock_retriever,
            api_key="mock_key_for_testing"
        )
        res = answerer.answer_question("What are late blight symptoms?", crop="Potato", disease="Late Blight")

        self.assertTrue(len(res.sources) > 0)
        self.assertEqual(res.sources[0]["source_organization"], "University of Minnesota Extension")
        self.assertEqual(res.sources[0]["document_title"], "Late blight of tomato and potato")

    def test_3_correct_metadata_returned(self):
        """Test 3: Correct metadata is returned in sources payload."""
        answerer = RAGAnswerer(
            retriever=self.mock_retriever,
            api_key="mock_key_for_testing"
        )
        res = answerer.answer_question("Late blight symptoms?", crop="Potato", disease="Late Blight")
        meta = res.sources[0]

        self.assertIn("url_reference", meta)
        self.assertEqual(meta["url_reference"], "https://extension.umn.edu/disease-management/late-blight")
        self.assertEqual(meta["disease"], "Late Blight")

    def test_4_no_context_insufficient_context_behavior(self):
        """Test 4: No-context or insufficient-context behavior returns refusal."""
        mock_empty_retriever = MagicMock(spec=RAGRetriever)
        mock_empty_retriever.retrieve.return_value = []  # Zero results

        answerer = RAGAnswerer(
            retriever=mock_empty_retriever,
            api_key="mock_key_for_testing"
        )
        res = answerer.answer_question("What is the capital of France?")

        self.assertTrue(res.insufficient_context)
        self.assertFalse(res.is_grounded)
        self.assertIn("don't have enough information", res.answer)

    def test_5_disease_filtering_works(self):
        """Test 5: Disease filtering is passed to retriever."""
        answerer = RAGAnswerer(
            retriever=self.mock_retriever,
            api_key="mock_key_for_testing"
        )
        answerer.answer_question("Early blight prevention?", crop="Potato", disease="Early Blight")

        self.mock_retriever.retrieve.assert_called_with(
            query="Early blight prevention?",
            crop="Potato",
            disease="Early Blight",
            top_k=3
        )

    def test_6_crop_filtering_works(self):
        """Test 6: Crop filtering is passed to retriever."""
        answerer = RAGAnswerer(
            retriever=self.mock_retriever,
            api_key="mock_key_for_testing"
        )
        answerer.answer_question("Potato disease care?", crop="Potato", disease=None)

        self.mock_retriever.retrieve.assert_called_with(
            query="Potato disease care?",
            crop="Potato",
            disease=None,
            top_k=3
        )

    def test_7_api_key_missing_handled_clearly(self):
        """Test 7: Missing API key is handled gracefully without crashing."""
        answerer = RAGAnswerer(
            retriever=self.mock_retriever,
            api_key=None
        )
        res = answerer.answer_question("What are symptoms?")

        self.assertIsInstance(res, RAGAnswer)
        self.assertTrue(len(res.sources) > 0)
        self.assertIn("Grounded Response", res.answer)

    def test_8_generated_prompt_contains_only_retrieved_context(self):
        """Test 8: Grounding prompt contains ONLY retrieved context and strict rules."""
        answerer = RAGAnswerer(retriever=self.mock_retriever, api_key="mock_key_for_testing")
        prompt, context_str = answerer.build_grounding_prompt("Potato late blight symptoms?", self.sample_results)

        self.assertIn("VERIFIED RETRIEVED SOURCES:", prompt)
        self.assertIn("STRICT GROUNDING RULES:", prompt)
        self.assertIn("University of Minnesota Extension", prompt)
        self.assertIn("small, water-soaked dark green", prompt)

    def test_9_source_urls_come_from_stored_metadata(self):
        """Test 9: Source URLs come strictly from stored metadata."""
        answerer = RAGAnswerer(retriever=self.mock_retriever, api_key="mock_key_for_testing")
        res = answerer.answer_question("Symptoms?")

        for src in res.sources:
            self.assertTrue(src["url_reference"].startswith("http"))
            self.assertEqual(src["url_reference"], "https://extension.umn.edu/disease-management/late-blight")

if __name__ == "__main__":
    unittest.main()
