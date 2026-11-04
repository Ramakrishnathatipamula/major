# AI Crop Doctor — Architecture and Workflow

This document describes the implemented architecture, request flow, data contracts,
safety gates, and extension points. It reflects the current code, not an aspirational
design.

---

## 1. System context

AI Crop Doctor is a single-machine Streamlit application. It accepts a leaf image and
a field location, then produces a decision-support payload containing:

- image and leaf-validation status,
- identified crop and routing decision,
- disease prediction and confidence,
- Grad-CAM explanation,
- actual weather and environmental risk,
- crop-safe grounded knowledge,
- treatment recommendations or withholding reasons,
- optional hypothetical What-If weather results.

The system does **not** control machinery, prescribe unverified pesticides, or claim
field-level accuracy from laboratory test splits.

---

## 2. Component map

| Layer | Module | Responsibility |
|---|---|---|
| UI | `app.py`, `ui/components/` | Sidebar inputs, image upload, result views, What-If controls |
| Orchestration | `modules/intelligence/orchestrator.py` | Coordinates diagnosis, weather, risk, RAG, treatment, simulation |
| Context | `modules/intelligence/context_builder.py` | Normalizes confidence units and builds retrieval context |
| Query generation | `modules/intelligence/query_generator.py` | Builds crop/disease-aware RAG queries |
| Leaf validation | `modules/identification/leaf_verifier.py` | Rejects blank, corrupt, non-leaf, and synthetic non-foliar inputs |
| Plant identification | `modules/identification/plant_identifier.py` | 15-class ResNet-18 species classifier (14 PlantVillage species + Wheat from public field dataset), prototypes, OOD/energy gates |
| Routing | `modules/identification/router.py` | Manual/auto crop resolution and specialist gating |
| Diagnosis | `modules/diagnosis/pipeline.py` | Runs routing, specialist inference, Grad-CAM, weather, and risk |
| Disease inference | `modules/disease_predictor.py` | Potato, Tomato, Maize, and Wheat specialist checkpoints |
| Explainability | `modules/explainability/gradcam.py` | Grad-CAM heatmap generation |
| Weather | `modules/weather/` | Open-Meteo retrieval and location hierarchy/resolution |
| Risk | `modules/risk/` | Rule-based environmental disease-risk indicators |
| RAG | `modules/rag/` | Document loading, chunking, embeddings, ChromaDB, retrieval, grounded answers |
| Treatment | `modules/treatment/` | Verified treatment registry and safety-gated recommendations |
| Simulation | `modules/simulation/` | Hypothetical weather scenarios |
| Crop registry | `modules/crops/crop_registry.py` | Canonical crop identities, aliases, support status, display names |
| Path portability | `modules/dataset_paths.py` | Resolves project-relative and `~`-based manifest paths |

> **Implementation note:** the current code does not use LangChain. RAG is implemented
> directly with `sentence-transformers`, ChromaDB, and `google-genai`. Documentation
> should not describe a LangChain pipeline that is not present.

---

## 3. End-to-end workflow

```text
1. User selects/upload image, crop mode, and confirms a field location.
2. UI validates the image container and writes uploads to data/processed/temp_uploads/.
3. Intelligence layer calls the diagnosis pipeline.
4. Router resolves manual/auto crop mode.
5. LeafVerifier validates the image for every mode.
   - blank -> REJECTED_BLANK
   - non-leaf -> REJECTED_NON_LEAF
   - corrupt/missing -> INVALID_INPUT
6. If the manual crop is unsupported:
   - return UNSUPPORTED_CROP with confidence 0.0 and verified leaf evidence.
7. If auto-detect:
   - PlantIdentifier predicts species, confidence, prototype similarity, energy.
   - OOD/low-confidence -> UNKNOWN_SPECIES.
   - Supported Potato/Tomato/Maize/Wheat -> specialist routing with confidence status.
   - Other recognized species -> UNSUPPORTED_CROP.
8. Supported crop -> DiseasePredictor runs the crop-specific ResNet-18.
9. Grad-CAM explains the predicted class.
10. Weather client retrieves actual conditions for the confirmed location.
11. Risk engine computes environmental risk indicators.
12. RAG retrieves crop-compatible, disease-compatible verified evidence.
13. Answer layer generates a grounded Gemini answer or deterministic fallback.
14. Treatment layer returns verified recommendations only when safety gates pass.
15. UI renders diagnosis, treatment, weather, risk, knowledge, and What-If views.
```

What-If simulation reruns the analysis with hypothetical weather parameters while
keeping the original image diagnosis. Actual and simulated weather are stored and
displayed separately.

---

## 4. Crop identity and routing contract

`RoutingDecision` (`modules/identification/router.py`) contains:

| Field | Meaning |
|---|---|
| `target_crop` | Canonical crop id used downstream, e.g. `potato`, `tomato`, `maize` |
| `display_name` | User-facing name |
| `route_to_specialist` | Whether a production disease specialist may run |
| `routing_status` | `ROUTED_TO_*`, `UNSUPPORTED_CROP`, `UNKNOWN_SPECIES`, `REJECTED_*`, `INVALID_INPUT` |
| `confidence` | Identification/specialist confidence; `0.0` when no model assessed the crop |
| `message` | User-facing explanation |
| `leaf_verification` | LeafVerifier evidence |
| `plant_identification` | PlantIdentifier evidence when auto-detection ran |
| `warning` | Mismatch or uncertainty warning |

### Canonical propagation

The intelligence layer resolves crop identity in this order:

1. `detected_crop` from the diagnosis result,
2. routing decision target crop,
3. the originally requested crop.

The original selection is retained as `requested_crop` for audit. This prevents an
unsupported manual selection from silently borrowing evidence from another crop.

### Manual unsupported-crop rule

Every manual selection now passes through leaf verification first. A valid leaf with
an unsupported crop returns `UNSUPPORTED_CROP` and `confidence = 0.0`; a blank,
non-leaf, or corrupt input returns the corresponding rejection status instead. No
path returns a fabricated 100% confidence without image verification.

---

## 5. Disease specialists

| Crop | Architecture | Classes | Checkpoint |
|---|---|---|---|
| Potato | ResNet-18 | 3 | `models/disease/potato_disease_model.pth` |
| Tomato | ResNet-18 | 10 | `models/disease/tomato_disease_model.pth` |
| Maize | ResNet-18 | 4 | `models/disease/maize_disease_model.pth` |
| Wheat | ResNet-18 | 3 | `models/disease/wheat_disease_model.pth` |

Classification confidence is normalized to percentage units `[0.0, 100.0]`.
Threshold semantics are shared with the UI and safety layers:

- `< 40.0%`: rejected / low confidence,
- `40.0%` to `< 60.0%`: uncertain,
- `>= 60.0%`: confident.

These thresholds are product-safety controls and must not be loosened to improve
reported metrics.

Maize and Wheat are diagnostically supported only while their specialist checkpoints
exist (`models/disease/maize_disease_model.pth`, `models/disease/wheat_disease_model.pth`).
The corresponding treatment registries are not yet populated, so supported Maize/Wheat
diagnoses return `VERIFIED_RECOMMENDATION_UNAVAILABLE` with zero chemical recommendations.

Experimental `*_field_adapted_model.pth` checkpoints are research artifacts. They are
not selected by the production `DiseasePredictor`.

---

## 6. Open-set and leaf safety

`PlantIdentifier` combines a closed-set classifier with:

- feature-prototype cosine similarity from training embeddings,
- an energy score,
- confidence thresholds,
- ambiguity checks.

`LeafVerifier` runs before neural inference and rejects:

- blank/uniform images,
- corrupt or missing files,
- non-foliar colour distributions,
- images without natural leaf texture/contours.

The ordering is deliberate: a closed-set classifier can hallucinate high confidence
for arbitrary objects, so leaf verification and OOD gates protect the specialist
models.

---

## 7. Crop-safe RAG

### Ingestion

`scripts/ingest_knowledge_base.py` loads verified extension documents from
`data/knowledge/`, chunks them, embeds them with `BAAI/bge-small-en-v1.5`, and writes
to the persistent ChromaDB collection.

Current verified store: **82 chunks** (45 Potato, 37 Tomato).

### Retrieval

`modules/rag/retriever.py` enforces:

1. **Crop canonicalization** through `CropRegistry`, so `Corn` maps to `Maize` and
   `potato` maps to `Potato`.
2. **ChromaDB crop filter** using the canonical display name.
3. **Python-side crop revalidation** rejecting missing, empty, `none`, `unknown`, or
   mismatched crop metadata.
4. **Exactly one query.** There is deliberately no unfiltered retry. Zero compatible
   chunks returns an empty list.
5. **Disease compatibility post-filtering.** Disease metadata is heterogeneous
   (`Late Blight`, `Late Blight / Early Blight`, `Multiple (...)`), so exact equality
   would discard valid Tomato evidence. The retriever uses conservative token and
   alternative matching instead.

When retrieval returns nothing compatible, `RAGAnswerer` returns the deterministic
insufficient-knowledge message rather than cross-crop evidence.

### Answer grounding

`RAGAnswerer` cites retrieved source metadata (`source_organization`,
`document_title`, `url_reference`). Gemini generation is optional. The offline
fallback is conservative and remains grounded in the retrieved chunks.

---

## 8. Weather, risk, and simulation

### Actual weather

`modules/weather/weather_client.py` retrieves current and hourly conditions from
Open-Meteo. The location must be resolved and confirmed before diagnosis because
microclimate risk depends on real coordinates.

### Risk engine

`modules/risk/risk_engine.py` applies crop- and disease-aware rules to temperature,
humidity, precipitation, and leaf-wetness duration indicators. It reports a risk
level, index, and contributing factors. Risk is decision support, not a diagnosis.

### What-If simulation

`modules/simulation/simulator.py` accepts hypothetical weather parameters. The
result is labelled as simulated and never overwrites actual weather observations.
The UI keeps the two concepts visually distinct.

---

## 9. Treatment safety

`modules/treatment/` contains verified treatment records and safety gates. Chemical
recommendations are suppressed when:

- the crop has no production specialist,
- the species is unknown,
- the leaf/image is rejected,
- the prediction is uncertain or low confidence,
- the plant is healthy (preventive guidance only),
- the disease is outside the verified registry.

When recommendations are withheld, the payload includes the reason. Source records
cite CIBRC/ICAR/extension authorities where applicable.

---

## 10. Data and checkpoint boundaries

The following are intentionally not committed:

- raw and field datasets,
- nested PlantDoc checkout (`data/field_eval/raw_plantdoc/`),
- ChromaDB persistence,
- model checkpoints and prototypes,
- temporary uploads,
- generated evaluation reports.

Manifests use portable paths:

- project-owned data: `data/...`,
- Kaggle cache: `~/.cache/kagglehub/...`.

`resolve_dataset_path()` accepts portable paths and legacy absolute inputs, so
existing tooling remains backward compatible.

---

## 11. Failure behavior

| Condition | Behavior |
|---|---|
| Missing image / corrupt bytes | `INVALID_INPUT`, no diagnosis, no chemicals |
| Blank image | `REJECTED_BLANK` |
| Non-leaf image | `REJECTED_NON_LEAF` |
| Unknown species / OOD | `UNKNOWN_SPECIES`, no specialist, no RAG cross-crop evidence |
| Unsupported crop (manual or auto) | `UNSUPPORTED_CROP`, confidence `0.0` for manual, no specialist |
| Missing specialist checkpoint | Safe `MISSING_CHECKPOINT` response, no crash |
| No compatible RAG evidence | Insufficient-knowledge response, empty sources |
| Gemini unavailable | Deterministic offline fallback |
| Weather unavailable | UI reports the failure; diagnosis remains available |

---

## 12. Extension points

To add a new supported crop safely:

1. Add or validate a canonical `CropSpec` entry.
2. Build a validated disease-specialist dataset with leakage controls.
3. Train and evaluate a checkpoint; keep PlantVillage vs field metrics separate.
4. Register the checkpoint and class list in `DiseasePredictor`.
5. Add verified crop-specific knowledge documents.
6. Re-ingest and validate crop-safe retrieval.
7. Add treatment records only after authority verification.
8. Add routing, RAG, treatment, and end-to-end regression tests.
9. Leave the crop unsupported until every step above is complete.

Adding an identifier class alone is **not** sufficient to claim disease support.
