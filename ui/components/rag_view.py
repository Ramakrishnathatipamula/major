from typing import Dict, Any
import streamlit as st

def render_rag_view(payload: Dict[str, Any]):
    """
    Render Grounded Agricultural Knowledge (RAG + Gemini) Section.
    """
    st.markdown("## 📚 GROUNDED AGRICULTURAL KNOWLEDGE")
    st.markdown("##### 🛡️ Grounded in Verified Agricultural Extension Sources")

    know = payload.get("KNOWLEDGE", {})
    query = know.get("generated_query", "")
    answer = know.get("grounded_answer", "")
    is_grounded = know.get("grounding_status", False)
    insufficient_context = know.get("insufficient_context", False)
    gemini_avail = know.get("gemini_available", True)
    sources = know.get("retrieved_sources", [])

    if query:
        st.caption(f"🎯 **Targeted RAG Query**: *\"{query}\"*")

    # Grounded Answer Container
    if insufficient_context:
        st.info("ℹ️ **Knowledge Base Notice**: I don't have enough information in the verified knowledge base to answer that query.")
    elif not gemini_avail:
        st.warning("⚠️ **LLM Service Unavailable**: Natural-language explanation generator is currently offline. Verified reference sources are listed below.")
    else:
        st.markdown("### 📖 Grounded Knowledge Narrative")
        st.info(answer)

    # Verified Sources Table / References
    if sources:
        with st.expander("🔗 Verified Reference Extension Sources", expanded=True):
            for idx, src in enumerate(sources, start=1):
                org = src.get("source_organization", "Unknown Organization")
                title = src.get("document_title", "Extension Guide")
                url = src.get("url_reference", "#")
                disease = src.get("disease", "Potato")
                sim = src.get("similarity_score", 0.0)

                st.markdown(f"**{idx}. [{title}]({url})**")
                st.markdown(f"• **Organization**: {org} | **Disease**: {disease} | **Match Score**: {sim:.4f}")
                st.markdown(f"• **Verified URL**: [{url}]({url})")
                st.markdown("---")
