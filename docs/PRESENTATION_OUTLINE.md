# AI Crop Doctor — Presentation Outline

Suggested length: 12–15 minutes plus demo and questions.

---

## Slide 1 — Title

**AI Crop Doctor**  
Intelligent Crop Disease, Risk, and Agricultural Knowledge System

- Final-year B.Tech major project
- Leaf image → validated diagnosis → weather risk → grounded guidance → safe treatment

---

## Slide 2 — Problem statement

- Crop diseases reduce yield and farmer income.
- Smallholder farmers often lack immediate agronomist access.
- Generic image classifiers are unsafe: they can hallucinate confidence on non-leaf
  inputs and unrelated crops.
- Advice must be crop-specific, sourced, and safety-gated.

**Goal:** decision support, not autonomous pesticide prescription.

---

## Slide 3 — Project objectives

1. Validate that the upload is a usable leaf image.
2. Identify the crop species.
3. Route supported crops to validated disease specialists.
4. Predict disease and confidence.
5. Explain the prediction with Grad-CAM.
6. Retrieve crop-compatible verified knowledge.
7. Assess weather-driven disease risk.
8. Support What-If weather scenarios.
9. Withhold unsafe or unsupported treatment advice.

---

## Slide 4 — Implemented system

```text
Image upload
  → leaf validation
  → crop identification / manual routing
  → disease specialist
  → confidence + Grad-CAM
  → actual weather + risk
  → crop-safe RAG + citations
  → safety-gated treatment
  → What-If simulation
```

Single Streamlit application: `app.py`.

---

## Slide 5 — Supported crop matrix

| Crop | Identification | Disease specialist | RAG knowledge | Treatments |
|---|---|---|---|---|
| Potato | Yes | 3 classes | Yes | Verified, safety-gated |
| Tomato | Yes | 10 classes | Yes | Verified, safety-gated |
| Maize | Yes | 4 classes | No | Withheld (`VERIFIED_RECOMMENDATION_UNAVAILABLE`) |
| Wheat | Yes | 3 classes | No | Withheld (`VERIFIED_RECOMMENDATION_UNAVAILABLE`) |
| Rice | Yes | 4 classes | No | Withheld (`VERIFIED_RECOMMENDATION_UNAVAILABLE`) |
| Cotton / others | Registry entries | No | No | Withheld |

Unsupported crops never receive fabricated predictions or cross-crop evidence.

---

## Slide 6 — Models and explainability

- Plant identifier: ResNet-18, 15 classes (14 PlantVillage species + Wheat from public field dataset).
- Potato specialist: ResNet-18, 3 classes.
- Tomato specialist: ResNet-18, 10 classes.
- Maize specialist: ResNet-18, 4 classes (held-out laboratory test accuracy **97.32%**).
- Wheat specialist: ResNet-18, 3 classes (held-out laboratory test accuracy **89.13%**).
- Grad-CAM highlights image regions supporting the predicted disease class.
- Open-set gates:
  - confidence thresholds,
  - feature-prototype similarity,
  - energy score,
  - ambiguity checks.

**Important:** PlantVillage test accuracy is not field accuracy.

**Measured field results (PlantDoc, real-world photographs):**
- Potato field disease accuracy: **68.85%** (lab benchmark **100.0%**, domain drop **31.15%**).
- Tomato field disease accuracy: **43.9%** (lab benchmark **98.36%**, domain drop **54.46%**).
- Maize field disease accuracy: **42.69%** over 171 evaluable images (PlantDoc corn
  labels are coarse/approximate; specialist routing recall **61.54%**, precision **100%**).
- Treatment safety compliance on field set: **100%** (zero leaked chemicals).

---

## Slide 7 — Safety architecture

1. Leaf verification runs before every routing mode.
2. Manual unsupported crop → `UNSUPPORTED_CROP`, confidence `0.0`.
3. OOD / low confidence → `UNKNOWN_SPECIES`.
4. Chemical treatments withheld for:
   - unsupported crops,
   - unknown species,
   - healthy plants,
   - rejected predictions,
   - uncertain predictions.
5. Thresholds are not tuned to improve reported results.

---

## Slide 8 — Crop-safe RAG

Problem found during inspection:

- A filtered ChromaDB query with zero results retried **without** a filter.
- This could return Tomato evidence for a Maize/Wheat query.

Fix:

1. Exactly one retrieval query.
2. Canonical crop names through `CropRegistry`.
3. ChromaDB crop filter plus Python-side revalidation.
4. Missing/mismatched crop metadata rejected.
5. Disease compatibility post-filtering for combined labels.
6. Zero compatible results → insufficient-knowledge response.

Validated against the real 82-chunk store.

---

## Slide 9 — Weather, risk, and simulation

- Actual weather: Open-Meteo, based on confirmed field location.
- Risk engine: rule-based temperature, humidity, precipitation, and leaf-wetness
  indicators.
- What-If simulation: hypothetical weather inputs.
- Actual and simulated weather remain clearly separated in the UI and payload.

---

## Slide 10 — Treatment recommendations

- Verified treatment registry with source citations.
- Crop-specific and disease-specific.
- Safety gates suppress chemicals when diagnosis is unsupported, uncertain, rejected,
  or healthy.
- Withholding reason is shown to the user.

---

## Slide 11 — Engineering quality

- Modular package structure (`modules/`, `ui/`, `scripts/`, `tests/`).
- Portable dataset manifests:
  - project-relative paths for project data,
  - `~/` paths for Kaggle cache.
- `.gitignore` protects datasets, checkpoints, vector store, nested repos, and temps.
- Explicit dependencies in `requirements.txt`.
- Reproducible test commands documented.

---

## Slide 12 — Test results

| Check | Result |
|---|---|
| Full suite | 201 tests passed |
| Crop-safe RAG | 13 passed |
| Routing / open-set / registry | 43 passed |
| Portable dataset paths | 5 passed |
| Real ChromaDB check | Passed |
| App import | Passed |
| Streamlit health | HTTP 200 `ok` |
| Field benchmark (PlantDoc) | 2,572 images; 0 duplicates vs PlantVillage |

Commands and warnings are in `docs/TEST_REPORT.md`.

### 12a — Measured results snapshot

| Claim | Value |
|---|---|
| 15-class identifier held-out test accuracy | **98.52%** |
| Maize specialist held-out laboratory test accuracy | **97.32%** |
| Wheat specialist held-out laboratory test accuracy | **89.13%** |
| PlantDoc Potato field disease accuracy | **68.85%** |
| PlantDoc Tomato field disease accuracy | **43.9%** |
| PlantDoc Maize field disease accuracy (approx. labels) | **42.69%** |
| Field treatment safety compliance | **100%** |

---

## Slide 13 — Limitations and honest claims

- Maize/Wheat chemical treatment registries are not yet populated, so treatments are
  withheld with `VERIFIED_RECOMMENDATION_UNAVAILABLE` even for supported diagnoses.
- PlantVillage accuracy is not field accuracy; the PlantDoc field evaluation is being
  re-run with the new Maize cohort mapping.
- Field-adapted checkpoints are experimental, not production defaults.
- Gemini answers require API availability; offline fallback is conservative.

These are documented limitations, not hidden workarounds.

---

## Slide 14 — Demo script

1. Show supported Potato leaf diagnosis.
2. Show confidence and Grad-CAM.
3. Show actual weather and risk factors.
4. Show crop-safe RAG answer and sources.
5. Show treatment recommendations or withholding reason.
6. Run a What-If scenario and contrast it with actual weather.
7. Select an unsupported crop (e.g. Cotton) and show safe blocking.
8. Upload a blank/non-leaf image and show rejection.

---

## Slide 15 — Future work

- Validated Wheat identification and Maize/Wheat disease specialists.
- Independent field validation protocol.
- Expanded verified knowledge and treatment registry.
- Calibration study for field confidence.
- Deployment and offline packaging for low-connectivity regions.

---

## Closing statement

AI Crop Doctor demonstrates an end-to-end, safety-first agricultural decision-support
pipeline. Its main engineering contribution is not only prediction accuracy, but
**crop-safe evidence handling, explicit uncertainty, and treatment suppression** when
the system cannot support a reliable recommendation.
