import unittest
import tempfile
import shutil
from pathlib import Path
from modules.rag import (
    DocumentLoader,
    AgriculturalDocument,
    DocumentChunker,
    DocumentChunk,
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    SearchResult,
)

class TestRAGPhase1(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create a small local test document
        self.sample_doc_path = self.temp_dir / "potato_late_blight_guide.txt"
        self.sample_content = (
            "Potato Late Blight is caused by the oomycete pathogen Phytophthora infestans.\n\n"
            "Visual Symptoms: Early symptoms appear as small, water-soaked brown lesions on leaves. "
            "Under humid conditions, white mold spores form on the underside of leaves.\n\n"
            "Environmental Favorable Conditions: Late Blight develops rapidly when relative humidity "
            "exceeds 90% and temperatures stay between 15°C and 22°C for continuous periods of 9+ hours.\n\n"
            "Preventive Management: Plant certified disease-free seed tubers. Eliminate cull piles. "
            "Use drip irrigation to reduce canopy wetness duration."
        )
        self.sample_doc_path.write_text(self.sample_content, encoding="utf-8")

        self.sample_metadata = {
            "source_organization": "University Extension Research",
            "document_title": "Potato Late Blight Management Guide",
            "crop": "Potato",
            "disease": "Late Blight",
            "publication_date": "2023-04-10",
            "url_reference": "https://extension.example.edu/potato_late_blight.pdf",
            "document_type": "extension_bulletin",
            "target_pathogen": "Phytophthora infestans",
            "geographic_relevance": "Temperate Potato Regions"
        }

        # Vector store in temp directory for test isolation
        self.vector_store = VectorStore(
            persist_dir=self.temp_dir / "test_vector_store",
            collection_name="test_rag_collection"
        )
        self.embedding_manager = EmbeddingManager()
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1_document_loading(self):
        """Test 1: Document loading extracts text and assigns metadata."""
        loader = DocumentLoader()
        doc = loader.load_file(self.sample_doc_path, self.sample_metadata)

        self.assertIsInstance(doc, AgriculturalDocument)
        self.assertEqual(doc.crop, "Potato")
        self.assertEqual(doc.disease, "Late Blight")
        self.assertEqual(doc.source_organization, "University Extension Research")
        self.assertIn("Phytophthora infestans", doc.content)

    def test_2_document_chunking(self):
        """Test 2: Document chunking produces chunks preserving boundaries."""
        loader = DocumentLoader()
        doc = loader.load_file(self.sample_doc_path, self.sample_metadata)
        chunker = DocumentChunker(target_chunk_size=300, overlap=50)

        chunks = chunker.chunk_document(doc)
        self.assertTrue(len(chunks) > 0)
        self.assertIsInstance(chunks[0], DocumentChunk)
        self.assertTrue(len(chunks[0].chunk_text) > 0)
        self.assertTrue(len(chunks[0].chunk_id) > 0)

    def test_3_metadata_preservation(self):
        """Test 3: Metadata attributes are attached to every single chunk."""
        loader = DocumentLoader()
        doc = loader.load_file(self.sample_doc_path, self.sample_metadata)
        chunker = DocumentChunker()

        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["crop"], "Potato")
            self.assertEqual(chunk.metadata["disease"], "Late Blight")
            self.assertEqual(chunk.metadata["source_organization"], "University Extension Research")
            self.assertIn("chunk_index", chunk.metadata)

    def test_4_embedding_generation(self):
        """Test 4: EmbeddingManager generates vectors for texts and queries."""
        texts = ["Late blight causes severe foliage rot.", "High humidity favors spore germination."]
        embeddings = self.embedding_manager.embed_texts(texts)

        self.assertEqual(len(embeddings), 2)
        self.assertTrue(len(embeddings[0]) > 0)

        query_vec = self.embedding_manager.embed_query("What causes late blight?")
        self.assertTrue(len(query_vec) > 0)

    def test_5_chromadb_insertion(self):
        """Test 5: VectorStore inserts chunks into persistent ChromaDB store."""
        loader = DocumentLoader()
        doc = loader.load_file(self.sample_doc_path, self.sample_metadata)
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)

        embeddings = self.embedding_manager.embed_texts([c.chunk_text for c in chunks])
        added_count = self.vector_store.add_chunks(chunks, embeddings)

        self.assertEqual(added_count, len(chunks))
        self.assertEqual(self.vector_store.count(), len(chunks))

    def test_6_duplicate_handling(self):
        """Test 6: Inserting identical duplicate chunks is prevented."""
        loader = DocumentLoader()
        doc = loader.load_file(self.sample_doc_path, self.sample_metadata)
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)
        embeddings = self.embedding_manager.embed_texts([c.chunk_text for c in chunks])

        added_first = self.vector_store.add_chunks(chunks, embeddings)
        self.assertTrue(added_first > 0)

        # Attempt to add same chunks again
        added_second = self.vector_store.add_chunks(chunks, embeddings)
        self.assertEqual(added_second, 0)  # Duplicates skipped
        self.assertEqual(self.vector_store.count(), added_first)

    def test_7_retrieval(self):
        """Test 7: RAGRetriever retrieves semantically relevant chunks with metadata."""
        loader = DocumentLoader()
        doc = loader.load_file(self.sample_doc_path, self.sample_metadata)
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)
        embeddings = self.embedding_manager.embed_texts([c.chunk_text for c in chunks])
        self.vector_store.add_chunks(chunks, embeddings)

        results = self.retriever.retrieve(
            query="What humidity causes late blight spore germination?",
            crop="Potato",
            disease="Late Blight",
            top_k=2
        )

        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], SearchResult)
        self.assertIn("humidity", results[0].chunk_text.lower())
        self.assertEqual(results[0].metadata["crop"], "Potato")
        self.assertEqual(results[0].metadata["disease"], "Late Blight")

    def test_8_invalid_and_empty_document_handling(self):
        """Test 8: Gracefully handles non-existent, unreadable, or empty files."""
        loader = DocumentLoader()

        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            loader.load_file(self.temp_dir / "non_existent.txt", self.sample_metadata)

        # Empty file
        empty_file = self.temp_dir / "empty.txt"
        empty_file.write_text("   \n", encoding="utf-8")
        with self.assertRaises(ValueError):
            loader.load_file(empty_file, self.sample_metadata)

if __name__ == "__main__":
    unittest.main()
