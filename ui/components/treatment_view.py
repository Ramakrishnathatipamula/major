from typing import Dict, Any, List
import streamlit as st


def render_treatment_view(payload: Dict[str, Any]):
    """
    Render dedicated Recommended Treatment / Medicine Section.
    Directly answers: Plant Image -> Crop Identification -> Disease Detection -> Recommended Medicine/Treatment.
    Strictly grounded in verified CIBRC, ICAR-CPRI, and University Extension documentation.
    """
    treatment = payload.get("TREATMENT") or payload.get("treatment_recommendation")
    if not treatment:
        # Fallback if pipeline ran without treatment payload
        st.markdown("## 💊 RECOMMENDED TREATMENT / MEDICINE")
        st.info("ℹ️ Treatment recommendation data is not available for this run.")
        return

    crop = treatment.get("crop", "Potato")
    disease = treatment.get("predicted_disease", "Unknown")
    status = treatment.get("status", "")
    is_healthy = treatment.get("is_healthy", False)
    is_supported = treatment.get("is_supported", False)
    summary_msg = treatment.get("summary_message", "")
    chemical_options: List[Dict[str, Any]] = treatment.get("chemical_treatments", [])
    non_chemical_measures: List[str] = treatment.get("non_chemical_control_measures", [])
    safety_precautions: List[str] = treatment.get("safety_and_environmental_precautions", [])
    evidence_sources: List[Dict[str, Any]] = treatment.get("evidence_sources", [])
    evidence_gaps: List[str] = treatment.get("evidence_gaps", [])
    disclaimer = treatment.get("disclaimer", "")

    st.markdown("## 💊 RECOMMENDED TREATMENT / MEDICINE")
    st.markdown("##### 🛡️ Official CIBRC & ICAR-CPRI Verified Agricultural Guidance")

    # Metadata Header Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Crop", crop)
    with col2:
        st.metric("Diagnosed Disease", disease)
    with col3:
        if is_healthy:
            st.metric("Treatment Status", "Preventive Care Only 🌿")
        elif status == "PREDICTION_REJECTED":
            st.metric("Treatment Status", "Chemicals Suppressed ❌")
        elif status == "UNCERTAIN_DIAGNOSIS_WARNING":
            st.metric("Treatment Status", "Uncertain (Chemicals Withheld) ⚠️")
        elif is_supported and status == "VERIFIED_RECOMMENDATION_AVAILABLE":
            st.metric("Treatment Status", "Verified Medicine Active 🟢")
        else:
            st.metric("Treatment Status", "Pending / Unavailable ⚠️")

    # 1. Healthy Foliage Scenario (Strict No-Chemical Rule)
    if is_healthy:
        st.success(f"🌿 **{crop} Foliage is Healthy**: No disease-specific chemical fungicides or curative medicine applications are required or recommended.")
        st.info(summary_msg)

        if non_chemical_measures:
            st.markdown("### 🚜 Recommended Cultural & Preventive Crop Care")
            for measure in non_chemical_measures:
                st.markdown(f"• {measure}")

        if safety_precautions:
            with st.expander("🛡️ Field & Agro-Ecological Safety Notes", expanded=False):
                for p in safety_precautions:
                    st.markdown(f"• {p}")

        if evidence_sources:
            with st.expander("🔗 Verified Reference Documentation", expanded=False):
                for src in evidence_sources:
                    org = src.get("source_organization", "Extension Authority")
                    title = src.get("document_title", "Extension Guide")
                    url = src.get("url_reference", "#")
                    st.markdown(f"• **[{title}]({url})** — *{org}*")

        st.caption(f"⚠️ *{disclaimer}*")
        return

    # 2. Prediction Rejected Scenario (<40% Confidence or Invalid Input)
    if status == "PREDICTION_REJECTED":
        st.error(f"❌ **Visual Prediction Rejected**: {summary_msg}")
        st.warning(
            "🛡️ **Chemical Safety Protocol**: Disease-specific chemical fungicides are strictly withheld "
            "on inconclusive diagnoses (<40.0% confidence). Applying fungicides without a confirmed diagnosis "
            "incurs unnecessary farmer expenses, risks crop injury, and accelerates pathogen resistance."
        )

        if non_chemical_measures:
            st.markdown("### 🚜 Recommended Physical Scouting & Diagnostic Steps")
            for measure in non_chemical_measures:
                st.markdown(f"• {measure}")

        if safety_precautions:
            with st.expander("🛡️ Agro-Ecological Safety Notes", expanded=True):
                for p in safety_precautions:
                    st.markdown(f"• {p}")

        if evidence_gaps:
            with st.expander("📋 Evidence Gap Details", expanded=False):
                for gap in evidence_gaps:
                    st.markdown(f"• {gap}")

        st.caption(f"⚖️ *{disclaimer}*")
        return

    # 3. Uncertain Prediction Warning Scenario (<60% Confidence)
    if status == "UNCERTAIN_DIAGNOSIS_WARNING":
        st.warning(f"⚠️ **Uncertain Diagnosis Warning**: {summary_msg}")
        st.info(
            "ℹ️ **Chemical Withholding Policy**: Because visual classification confidence is marginal (<60.0%), "
            "disease-specific chemical fungicides are withheld. Confirm foliar lesions in the field or consult your "
            "local Krishi Vigyan Kendra (KVK) before chemical application."
        )

        if non_chemical_measures:
            st.markdown("### 🚜 Recommended Cultural & Preventative Practices")
            for measure in non_chemical_measures:
                st.markdown(f"• {measure}")

        if safety_precautions:
            with st.expander("🛡️ Agro-Ecological Safety Notes", expanded=True):
                for p in safety_precautions:
                    st.markdown(f"• {p}")

        if evidence_gaps:
            with st.expander("📋 Evidence Gap Details", expanded=False):
                for gap in evidence_gaps:
                    st.markdown(f"• {gap}")

        st.caption(f"⚖️ *{disclaimer}*")
        return

    # 4. Supported Crop Without a Verified Treatment Registry (e.g. Maize/Wheat/Rice)
    # Show farmer-facing treatment steps directly; hide technical status label.
    if status == "VERIFIED_RECOMMENDATION_UNAVAILABLE":
        st.markdown("## 🌾 Recommended Crop Care Steps")
        st.info(summary_msg)

        if non_chemical_measures:
            st.markdown("### ✅ What to Do Now")
            for measure in non_chemical_measures:
                st.markdown(f"• {measure}")

        if safety_precautions:
            with st.expander("🛡️ Safety & Precaution Notes", expanded=True):
                for p in safety_precautions:
                    st.markdown(f"• {p}")

        st.caption(f"⚖️ *{disclaimer}*")
        return

    # 5. Unsupported Crop or Disease Scenario
    if not is_supported or status in ("UNSUPPORTED_CROP", "UNSUPPORTED_DISEASE", "INSUFFICIENT_EVIDENCE"):
        st.warning(f"⚠️ **Verified Treatment Notice**: {summary_msg}")
        if evidence_gaps:
            with st.expander("📋 Evidence Gap Details", expanded=True):
                for gap in evidence_gaps:
                    st.markdown(f"• {gap}")
        st.caption(f"⚠️ *{disclaimer}*")
        return

    # 6. Active Disease with Verified Treatment Options (Early Blight / Late Blight)
    st.info(f"📋 **Clinical Management Guidance**: {summary_msg}")

    # Farmer-facing non-chemical cultural measures (shown prominently for all crops)
    if non_chemical_measures:
        st.markdown("### ✅ What to Do Now - Crop Care Steps")
        for measure in non_chemical_measures:
            st.markdown(f"• {measure}")
        st.markdown("---")

    if is_healthy:
        st.info(
            f"🌿 **Simple guidance**: Your **{crop}** plant foliage is healthy. No disease-specific "
            f"chemical treatment is needed. Continue routine field monitoring and the preventive steps above."
        )

    # Section 1: Chemical & Biological Treatment Options (Potato/Tomato only)
    if chemical_options:
        st.markdown("### 🧪 Verified Treatment & Fungicide Formulations (CIBRC Registered)")
        st.caption("Fungicides registered under the Insecticides Act (CIBRC) and endorsed by ICAR-CPRI / TNAU:")

        for idx, opt in enumerate(chemical_options, start=1):
            category = opt.get("category", "Fungicide")
            ai = opt.get("active_ingredient", "Unspecified Active Ingredient")
            formulation = opt.get("formulation", "")
            rate = opt.get("application_rate", "Consult registered container label")
            timing = opt.get("timing_and_frequency", "Follow extension timing")
            directions = opt.get("application_directions", "")
            phi = opt.get("pre_harvest_interval_phi", "Refer to label")
            rei = opt.get("re_entry_interval_rei", "24 hours")
            opt_source = opt.get("evidence_source_title", "CIBRC / Extension Registry")
            opt_url = opt.get("evidence_source_url", "https://ppqs.gov.in")

            with st.container():
                # Simple farmer-facing "what to do" action line (plain language)
                fa = opt.get("farmer_action", "")
                if fa:
                    st.markdown(f"### 👉 **Farmer Action**: {fa}")
                    st.markdown("---")

                category = opt.get("category", "Fungicide")
                ai = opt.get("active_ingredient", "Unspecified Active Ingredient")
                formulation = opt.get("formulation", "")
                rate = opt.get("application_rate", "Consult registered container label")
                timing = opt.get("timing_and_frequency", "Follow extension timing")
                directions = opt.get("application_directions", "")
                phi = opt.get("pre_harvest_interval_phi", "Refer to label")
                rei = opt.get("re_entry_interval_rei", "24 hours")
                opt_source = opt.get("evidence_source_title", "CIBRC / Extension Registry")
                opt_url = opt.get("evidence_source_url", "https://ppqs.gov.in")

                st.markdown(f"#### **Option {idx}: {ai} {formulation}**")
                st.markdown(f"🏷️ **Category**: `{category}` | **Status**: `Verified CIBRC / Extension`")

                col_det1, col_det2 = st.columns(2)
                with col_det1:
                    st.markdown(f"• **Verified Dosage / Rate**: `{rate}`")
                    st.markdown(f"• **Application Timing**: {timing}")
                    if directions:
                        st.markdown(f"• **Application Directions**: {directions}")
                with col_det2:
                    st.markdown(f"• **Pre-Harvest Interval (PHI)**: {phi}")
                    st.markdown(f"• **Re-Entry Interval (REI)**: {rei}")
                    st.markdown(f"• **Registered Evidence Reference**: [{opt_source}]({opt_url})")

                st.markdown("---")

    # Section 2: Non-Chemical Alternatives & Cultural Measures
    # (Already shown above prominently; this block kept for backward compatibility)
    if non_chemical_measures and "What to Do Now - Crop Care Steps" not in str(st.session_state.get("_rendered_sections", "")):
        pass  # Already displayed above

    # Section 3: Safety & PPE Precautions
    if safety_precautions:
        with st.expander("🛡️ Applicator Safety, PPE & Environmental Precautions", expanded=True):
            st.markdown("Critical safety protocols when preparing and applying crop protection products:")
            for sp in safety_precautions:
                st.markdown(f"• {sp}")

    # Section 4: Evidence Gaps & Commercial Brand Notice
    if evidence_gaps:
        with st.expander("🔍 Verification Status & Evidence Gaps Notice", expanded=False):
            st.markdown(
                "To prevent misinformation, AI Crop Doctor does NOT invent commercial trade names, "
                "unregistered rates, or proprietary brand formulations. The following details are omitted "
                "or require local verification:"
            )
            for gap in evidence_gaps:
                st.markdown(f"• **Omitted/Unverified**: {gap}")
            st.markdown(
                "Farmers must inspect the physical container label approved by CIBRC at the point of "
                "authorized agro-dealer purchase."
            )

    # Section 5: Clickable Authoritative Sources
    if evidence_sources:
        with st.expander("🔗 Official Regulatory & Extension Citations", expanded=False):
            for s_idx, src in enumerate(evidence_sources, start=1):
                org = src.get("source_organization", "Official Authority")
                title = src.get("document_title", "Regulatory Document")
                url = src.get("url_reference", "https://ppqs.gov.in")
                dtype = src.get("document_type", "treatment_guide")
                st.markdown(f"**{s_idx}. [{title}]({url})**")
                st.markdown(f"• **Organization**: {org} | **Type**: `{dtype}`")
                st.markdown(f"• **Official URL**: [{url}]({url})")
                st.markdown("---")

    # Statutory Disclaimer Footer
    st.caption(f"⚖️ *{disclaimer}*")
