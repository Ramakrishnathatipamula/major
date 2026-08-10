import sys
import json
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.rag import RAGAnswerer

def main():
    print("=========================================================")
    print("AI CROP DOCTOR — RAG PHASE 2 GROUNDED LLM DEMONSTRATION")
    print("=========================================================\n")

    answerer = RAGAnswerer()

    # DEMO 1: Supported Potato Late Blight Question
    query_1 = "What are the symptoms of potato late blight?"
    print(f"--- DEMO 1: SUPPORTED AGRICULTURAL QUERY ---")
    print(f"QUESTION: \"{query_1}\"")
    print(f"Filters : crop = Potato, disease = Late Blight\n")

    ans_1 = answerer.answer_question(query_1, crop="Potato", disease="Late Blight")

    print(f"ANSWER:\n{ans_1.answer}\n")
    print("VERIFIED SOURCES RETURNED:")
    for src in ans_1.sources:
        print(f" • Organization: {src.get('source_organization')}")
        print(f"   Document    : {src.get('document_title')}")
        print(f"   URL         : {src.get('url_reference')}")
        print(f"   Similarity  : {src.get('similarity_score')}\n")

    # DEMO 2: Unsupported / Unrelated Question (Graceful Refusal)
    query_2 = "What is the capital of France and how do I train a dog?"
    print("---------------------------------------------------------")
    print(f"--- DEMO 2: UNSUPPORTED / UNRELATED QUERY ---")
    print(f"QUESTION: \"{query_2}\"\n")

    ans_2 = answerer.answer_question(query_2, crop="Potato", disease=None)

    print(f"ANSWER:\n{ans_2.answer}\n")
    print(f"Insufficient Context Flag : {ans_2.insufficient_context}")
    print(f"Grounded Status           : {ans_2.is_grounded}\n")
    print("=========================================================")

if __name__ == "__main__":
    main()
