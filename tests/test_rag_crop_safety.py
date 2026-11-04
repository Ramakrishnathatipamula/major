import unittest

from modules.rag import RAGAnswerer, RAGRetriever


class _FakeEmbeddingManager:
    def embed_query(self, query):
        return [0.1, 0.2, 0.3]


class _FakeVectorStore:
    """Vector store stub that returns canned results and records each query."""

    def __init__(self, documents=None, metadatas=None, distances=None):
        self.documents = documents or []
        self.metadatas = metadatas or []
        self.distances = distances or [0.2] * len(self.documents)
        self.calls = []

    def query(self, query_embedding, where_clause=None, top_k=3):
        self.calls.append({
            "query_embedding": query_embedding,
            "where_clause": where_clause,
            "top_k": top_k,
        })
        return {
            "documents": [list(self.documents)],
            "metadatas": [list(self.metadatas)],
            "distances": [list(self.distances)],
        }


def _make_retriever(metadatas, documents=None, distances=None):
    documents = documents or [f"Knowledge chunk {idx}" for idx in range(len(metadatas))]
    store = _FakeVectorStore(documents=documents, metadatas=metadatas, distances=distances)
    return RAGRetriever(vector_store=store, embedding_manager=_FakeEmbeddingManager()), store


class TestCropSafeRetrieval(unittest.TestCase):
    def test_potato_retrieves_only_potato_chunks(self):
        retriever, store = _make_retriever([
            {"crop": "Potato", "disease": "Late Blight"},
            {"crop": "Tomato", "disease": "Late Blight"},
            {"crop": "Potato", "disease": "Early Blight"},
        ])

        results = retriever.retrieve("potato late blight", crop="Potato", top_k=5)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.metadata["crop"] == "Potato" for r in results))
        self.assertEqual(store.calls[0]["where_clause"], {"crop": {"$eq": "Potato"}})
        self.assertEqual(len(store.calls), 1)

    def test_tomato_retrieves_only_tomato_chunks(self):
        retriever, _ = _make_retriever([
            {"crop": "Potato", "disease": "Late Blight"},
            {"crop": "Tomato", "disease": "Late Blight / Early Blight"},
            {"crop": "Tomato", "disease": "Multiple (...)"},
        ])

        results = retriever.retrieve("tomato late blight", crop="Tomato", top_k=5)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.metadata["crop"] == "Tomato" for r in results))

    def test_maize_never_retrieves_solanaceae_chunks(self):
        retriever, _ = _make_retriever([
            {"crop": "Potato", "disease": "Late Blight"},
            {"crop": "Tomato", "disease": "Late Blight / Early Blight"},
        ])

        self.assertEqual(retriever.retrieve("maize leaf disease", crop="Maize"), [])

    def test_wheat_never_retrieves_solanaceae_chunks(self):
        retriever, _ = _make_retriever([
            {"crop": "Potato", "disease": "Late Blight"},
            {"crop": "Tomato", "disease": "Late Blight / Early Blight"},
        ])

        self.assertEqual(retriever.retrieve("wheat leaf disease", crop="Wheat"), [])

    def test_empty_filtered_results_do_not_trigger_unfiltered_retrieval(self):
        retriever, store = _make_retriever([])

        results = retriever.retrieve("maize leaf disease", crop="Maize", top_k=3)

        self.assertEqual(results, [])
        self.assertEqual(len(store.calls), 1)
        self.assertEqual(store.calls[0]["where_clause"], {"crop": {"$eq": "Maize"}})

    def test_missing_or_unknown_crop_metadata_is_rejected(self):
        retriever, _ = _make_retriever([
            {"disease": "Late Blight"},
            {"crop": "", "disease": "Late Blight"},
            {"crop": "none", "disease": "Late Blight"},
        ])

        self.assertEqual(retriever.retrieve("potato late blight", crop="Potato"), [])

    def test_crop_name_case_and_alias_handling(self):
        retriever, _ = _make_retriever([
            {"crop": "potato", "disease": "Late Blight"},
            {"crop": "Maize", "disease": "Common Rust"},
        ])

        potato_results = retriever.retrieve("potato late blight", crop="potato")
        corn_results = retriever.retrieve("corn rust", crop="Corn")

        self.assertEqual(len(potato_results), 1)
        self.assertEqual(potato_results[0].metadata["crop"], "potato")
        self.assertEqual(len(corn_results), 1)
        self.assertEqual(corn_results[0].metadata["crop"], "Maize")

    def test_combined_tomato_disease_metadata_is_preserved(self):
        retriever, _ = _make_retriever([
            {"crop": "Tomato", "disease": "Late Blight / Early Blight"},
            {"crop": "Tomato", "disease": "Multiple (Bacterial Spot, Leaf Mold, Septoria, Spider Mites, Viruses)"},
            {"crop": "Potato", "disease": "Late Blight"},
        ])

        results = retriever.retrieve(
            "tomato late blight management",
            crop="Tomato",
            disease="Late Blight",
            top_k=5,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["crop"], "Tomato")
        self.assertEqual(results[0].metadata["disease"], "Late Blight / Early Blight")

    def test_early_blight_query_does_not_match_late_blight_chunks(self):
        retriever, _ = _make_retriever([
            {"crop": "Potato", "disease": "Late Blight"},
        ])

        self.assertEqual(
            retriever.retrieve("early blight", crop="Potato", disease="Early Blight"),
            [],
        )

    def test_septoria_query_matches_distinctive_tomato_metadata(self):
        retriever, _ = _make_retriever([
            {"crop": "Tomato", "disease": "Multiple (Bacterial Spot, Leaf Mold, Septoria, Spider Mites, Viruses)"},
        ])

        results = retriever.retrieve(
            "tomato septoria leaf spot",
            crop="Tomato",
            disease="Tomato___Septoria_leaf_spot",
        )

        self.assertEqual(len(results), 1)


class TestRAGCropSafeAnswerGeneration(unittest.TestCase):
    def setUp(self):
        self.retriever, _ = _make_retriever([
            {
                "crop": "Potato",
                "disease": "Late Blight",
                "source_organization": "Potato Extension",
                "document_title": "Potato Late Blight Guide",
                "url_reference": "https://example.org/potato",
            },
            {
                "crop": "Tomato",
                "disease": "Late Blight / Early Blight",
                "source_organization": "Tomato Extension",
                "document_title": "Tomato Blight Guide",
                "url_reference": "https://example.org/tomato",
            },
        ])
        self.answerer = RAGAnswerer(
            retriever=self.retriever,
            api_key="mock_key_for_testing",
        )

    def test_potato_answer_generation_still_works(self):
        answer = self.answerer.answer_question(
            "potato late blight guidance", crop="Potato", disease="Late Blight"
        )

        self.assertFalse(answer.insufficient_context)
        self.assertTrue(answer.is_grounded)
        self.assertIn("Grounded Response", answer.answer)
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0]["source_organization"], "Potato Extension")

    def test_tomato_answer_generation_still_works(self):
        answer = self.answerer.answer_question(
            "tomato late blight guidance", crop="Tomato", disease="Late Blight"
        )

        self.assertFalse(answer.insufficient_context)
        self.assertTrue(answer.is_grounded)
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0]["source_organization"], "Tomato Extension")

    def test_unsupported_crop_returns_insufficient_verified_knowledge(self):
        answer = self.answerer.answer_question(
            "maize late blight guidance", crop="Maize", disease="Late Blight"
        )

        self.assertTrue(answer.insufficient_context)
        self.assertFalse(answer.is_grounded)
        self.assertEqual(answer.sources, [])
        self.assertIn("don't have enough information", answer.answer)


if __name__ == "__main__":
    unittest.main()
