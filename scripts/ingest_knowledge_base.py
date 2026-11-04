import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.rag import (
    DocumentLoader,
    DocumentChunker,
    EmbeddingManager,
    VectorStore,
)

def main():
    print("=== AI Crop Doctor — Verified Knowledge Base Ingestion ===")

    knowledge_dir = ROOT_DIR / "data" / "knowledge"
    if not knowledge_dir.exists():
        print(f"Error: Knowledge directory not found at {knowledge_dir}")
        sys.exit(1)

    loader = DocumentLoader()
    chunker = DocumentChunker(target_chunk_size=500, overlap=100)
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(persist_dir=ROOT_DIR / "data" / "vector_store")

    # Clear existing ChromaDB collection to purge any unverified chunks
    print("Clearing persistent vector store to eliminate unverified chunks...")
    vector_store.clear()

    # List of verified production extension documents
    verified_documents = [
        {
            "file": knowledge_dir / "verified_umn_potato_late_blight.txt",
            "metadata": {
                "source_organization": "University of Minnesota Extension",
                "document_title": "Late blight of tomato and potato",
                "crop": "Potato",
                "disease": "Late Blight",
                "publication_date": "none",
                "url_reference": "https://extension.umn.edu/disease-management/late-blight",
                "document_type": "extension_guide",
                "target_pathogen": "Phytophthora infestans",
                "geographic_relevance": "Upper Midwest / North America",
                "is_verified": "true"
            }
        },
        {
            "file": knowledge_dir / "verified_psu_potato_late_blight.txt",
            "metadata": {
                "source_organization": "Penn State Extension",
                "document_title": "Tomato-Potato Late Blight in the Home Garden",
                "crop": "Potato",
                "disease": "Late Blight",
                "publication_date": "none",
                "url_reference": "https://extension.psu.edu/tomato-potato-late-blight-in-the-home-garden",
                "document_type": "extension_guide",
                "target_pathogen": "Phytophthora infestans",
                "geographic_relevance": "Northeastern US / Temperate Regions",
                "is_verified": "true"
            }
        },
        {
            "file": knowledge_dir / "verified_umn_potato_early_blight.txt",
            "metadata": {
                "source_organization": "University of Minnesota Extension",
                "document_title": "Early blight in tomato and potato",
                "crop": "Potato",
                "disease": "Early Blight",
                "publication_date": "none",
                "url_reference": "https://extension.umn.edu/disease-management/early-blight-tomato-and-potato",
                "document_type": "extension_guide",
                "target_pathogen": "Alternaria solani",
                "geographic_relevance": "Upper Midwest / North America",
                "is_verified": "true"
            }
        },
        {
            "file": knowledge_dir / "verified_ndsu_potato_early_blight.txt",
            "metadata": {
                "source_organization": "North Dakota State University Extension",
                "document_title": "Early Blight in Potato",
                "crop": "Potato",
                "disease": "Early Blight",
                "publication_date": "none",
                "url_reference": "https://www.ag.ndsu.edu/potatoextension/early-blight-in-potato",
                "document_type": "extension_guide",
                "target_pathogen": "Alternaria solani",
                "geographic_relevance": "Northern Great Plains / North America",
                "is_verified": "true"
            }
        },
        {
            "file": knowledge_dir / "verified_cibrc_icar_potato_late_blight_treatments.txt",
            "metadata": {
                "source_organization": "CIBRC / ICAR-CPRI / TNAU",
                "document_title": "Official CIBRC Registered Fungicides & ICAR-CPRI / TNAU Management Guidelines for Potato Late Blight",
                "crop": "Potato",
                "disease": "Late Blight",
                "publication_date": "2024",
                "url_reference": "https://ppqs.gov.in",
                "document_type": "treatment_guide",
                "target_pathogen": "Phytophthora infestans",
                "geographic_relevance": "India / National & State Level (ICAR-CPRI / TNAU)",
                "is_verified": "true"
            }
        },
        {
            "file": knowledge_dir / "verified_cibrc_icar_potato_early_blight_treatments.txt",
            "metadata": {
                "source_organization": "CIBRC / TNAU / NDSU Extension",
                "document_title": "Official CIBRC Registered Fungicides & TNAU / NDSU Management Guidelines for Potato Early Blight",
                "crop": "Potato",
                "disease": "Early Blight",
                "publication_date": "2024",
                "url_reference": "https://ppqs.gov.in",
                "document_type": "treatment_guide",
                "target_pathogen": "Alternaria solani",
                "geographic_relevance": "India & North America (CIBRC / TNAU / NDSU)",
                "is_verified": "true"
            }
        },
        {
            "file": knowledge_dir / "verified_cibrc_iivr_tomato_blight_treatments.txt",
            "metadata": {
                "source_organization": "CIBRC / ICAR-IIVR / TNAU",
                "document_title": "Official CIBRC Registered Fungicides & ICAR-IIVR / TNAU Guidelines for Tomato Blights",
                "crop": "Tomato",
                "disease": "Late Blight / Early Blight",
                "publication_date": "2024",
                "url_reference": "https://ppqs.gov.in",
                "document_type": "treatment_guide",
                "target_pathogen": "Phytophthora infestans & Alternaria solani",
                "geographic_relevance": "India / National & State Level (CIBRC / ICAR-IIVR / TNAU)",
                "is_verified": "true"
            }
        },
        {
            "file": knowledge_dir / "verified_cibrc_iivr_tomato_diseases_management.txt",
            "metadata": {
                "source_organization": "CIBRC / ICAR-IIVR / TNAU",
                "document_title": "CIBRC / ICAR-IIVR Tomato Disease & Pest Integrated Management Guidelines",
                "crop": "Tomato",
                "disease": "Multiple (Bacterial Spot, Leaf Mold, Septoria, Spider Mites, Viruses)",
                "publication_date": "2024",
                "url_reference": "https://iivr.icar.gov.in",
                "document_type": "extension_guide",
                "target_pathogen": "Various Solanaceae pathogens & pests",
                "geographic_relevance": "India / National & State Level (CIBRC / ICAR-IIVR / TNAU)",
                "is_verified": "true"
            }
        },
        {
            "file": knowledge_dir / "verified_wheat_diseases_management.txt",
            "metadata": {
                "source_organization": "CIBRC / ICAR-IIWBR / PAU",
                "document_title": "Official CIBRC Registered Fungicides and ICAR-IIWBR Guidelines for Wheat Rusts and Blotch",
                "crop": "Wheat",
                "disease": "Stripe Rust / Leaf Rust / Septoria Leaf Blotch",
                "publication_date": "2024",
                "url_reference": "https://iiwbr.icar.gov.in",
                "document_type": "treatment_guide",
                "target_pathogen": "Puccinia striiformis / Zymoseptoria tritici",
                "geographic_relevance": "India and Global Wheat Growing Belts (CIBRC / ICAR-IIWBR / PAU)",
                "is_verified": "true"
            }
        },
        {
            "file": knowledge_dir / "verified_maize_diseases_management.txt",
            "metadata": {
                "source_organization": "CIBRC / ICAR-IIMR / TNAU",
                "document_title": "Official CIBRC Registered Fungicides and ICAR-IIMR Guidelines for Maize Foliar Diseases",
                "crop": "Maize",
                "disease": "Common Rust / Turcicum Leaf Blight / Gray Leaf Spot",
                "publication_date": "2024",
                "url_reference": "https://iimr.icar.gov.in",
                "document_type": "treatment_guide",
                "target_pathogen": "Puccinia sorghi / Exserohilum turcicum / Cercospora zeae-maydis",
                "geographic_relevance": "India and Tropical/Subtropical Maize Belts (CIBRC / ICAR-IIMR / TNAU)",
                "is_verified": "true"
            }
        },
        {
            "file": knowledge_dir / "verified_rice_diseases_management.txt",
            "metadata": {
                "source_organization": "CIBRC / ICAR-NRRI / IRRI",
                "document_title": "Official CIBRC Registered Fungicides and ICAR-NRRI Guidelines for Rice Diseases",
                "crop": "Rice",
                "disease": "Bacterial Leaf Blight / Brown Spot / Leaf Smut / Blast",
                "publication_date": "2024",
                "url_reference": "https://nrri.icar.gov.in",
                "document_type": "treatment_guide",
                "target_pathogen": "Xanthomonas oryzae / Bipolaris oryzae / Entyloma oryzae",
                "geographic_relevance": "India and Asia Rice Growing Ecosystems (CIBRC / ICAR-NRRI)",
                "is_verified": "true"
            }
        },
        {
            "file": knowledge_dir / "verified_fruits_vegetables_diseases_management.txt",
            "metadata": {
                "source_organization": "CIBRC / ICAR-IIHR / State Extension",
                "document_title": "Integrated Disease Management and CIBRC Prescriptions for Fruits and Vegetables",
                "crop": "Multiple (Apple, Grape, Pepper, Citrus, Strawberry, Peach, Squash, Soybean)",
                "disease": "Scab, Mildews, Anthracnose, Leaf Spots",
                "publication_date": "2024",
                "url_reference": "https://ppqs.gov.in",
                "document_type": "treatment_guide",
                "target_pathogen": "Venturia inaequalis, Phytophthora, Colletotrichum, Erysiphaceae",
                "geographic_relevance": "National and International Horticultural Belts",
                "is_verified": "true"
            }
        }
    ]

    total_added = 0
    for doc_info in verified_documents:
        file_path = doc_info["file"]
        if not file_path.exists():
            print(f"Warning: Verified file {file_path.name} missing. Skipping.")
            continue

        meta = doc_info["metadata"]
        print(f"\nProcessing Verified Source: {file_path.name}")
        print(f" -> Title: {meta['document_title']}")
        print(f" -> URL: {meta['url_reference']}")

        doc = loader.load_file(file_path, meta)
        chunks = chunker.chunk_document(doc)
        print(f" -> Generated {len(chunks)} text chunks.")

        texts = [c.chunk_text for c in chunks]
        embeddings = embedding_manager.embed_texts(texts)
        added_count = vector_store.add_chunks(chunks, embeddings)
        print(f" -> Inserted {added_count} chunks into ChromaDB.")
        total_added += added_count

    print(f"\nVerified Ingestion Complete!")
    print(f"Total Verified Chunks Stored in ChromaDB: {vector_store.count()}")
    print(f"ChromaDB Storage Directory: {vector_store.persist_dir.resolve()}")

if __name__ == "__main__":
    main()
