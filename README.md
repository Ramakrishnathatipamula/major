# AI Crop Doctor

AI Crop Doctor is an agricultural decision-support application for leaf-image crop
diagnosis. It combines leaf validation, crop identification, crop-specific disease
specialists, Grad-CAM explanations, weather and epidemiological risk assessment,
crop-safe retrieval-augmented generation (RAG), safe treatment recommendations, and
hypothetical weather-scenario simulation in a single Streamlit interface.

> **Scientific scope:** reported model accuracy comes from laboratory PlantVillage
> splits unless explicitly labelled as field evaluation. It is **not** field accuracy.
> Experimental field-adapted checkpoints are not production defaults.

---

## Current capabilities

| Capability | Status |
|---|---|
| Leaf / blank / corrupt image validation | Implemented |
| Plant-species identification (15 cultivated species) | Implemented |
| Crop routing and unsupported-crop blocking | Implemented |
| Potato disease specialist (3 classes) | Implemented |
| Tomato disease specialist (10 classes) | Implemented |
| Maize disease specialist (4 classes) | Implemented (laboratory metrics) |
| Wheat disease specialist (3 classes) | Implemented (laboratory metrics) |
| Grad-CAM explanation | Implemented |
| Open-Meteo weather retrieval | Implemented |
| Rule-based environmental disease-risk engine | Implemented |
| Crop-safe RAG with source citations | Implemented |
| Gemini grounded answers + offline fallback | Implemented |
| What-If weather simulation | Implemented |
| Safety-gated treatment recommendations | Implemented |
| Unit / integration test suite | Implemented (201 tests) |

### Supported disease specialists

- **Potato** (ResNet-18, 3 classes): Early Blight, Late Blight, Healthy.
- **Tomato** (ResNet-18, 10 classes): Bacterial Spot, Early Blight, Late Blight,
  Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus,
  Mosaic Virus, Healthy.
- **Maize** (ResNet-18, 4 classes): Gray Leaf Spot, Common Rust, Northern Leaf Blight,
  Healthy. (Checkpoint: `models/disease/maize_disease_model.pth`, held-out laboratory
  test accuracy **97.32%** over 298 isolated images.)
- **Wheat** (ResNet-18, 3 classes): Septoria Leaf Blotch, Stripe Rust (Yellow Rust),
  Healthy. (Checkpoint: `models/disease/wheat_disease_model.pth`, held-out laboratory
  test accuracy **89.13%** over 46 isolated images from the public field dataset.)
- **Rice** (ResNet-18, 4 classes): Bacterial Leaf Blight, Brown Spot, Leaf Smut,
  Healthy. (Checkpoint: `models/disease/rice_disease_model.pth`, held-out laboratory
  test accuracy **97.59%** over 83 isolated images from the staged field dataset at
  `data/rice/raw/`. Chemical treatments withheld — no verified CIBRC/ICAR registry;
  status `VERIFIED_RECOMMENDATION_UNAVAILABLE`.)

### Crops without production disease specialists

Maize and Wheat now have trained disease specialists and are treated as
**diagnostically supported** once their new checkpoints exist. However, **no verified
CIBRC/ICAR treatment registry has been added for Maize or Wheat**, so chemical advice
remains withheld for those crops (`VERIFIED_RECOMMENDATION_UNAVAILABLE`) rather than
borrowing Potato/Tomato prescriptions. General non-chemical cultural management
guidance is provided for each maize and wheat disease class.

Rice now has a trained disease specialist (checkpoint:
`models/disease/rice_disease_model.pth`, held-out laboratory test accuracy
**97.59%** over 83 isolated images) covering 4 classes:
`Rice___Bacterial_leaf_blotch`, `Rice___Brown_spot`, `Rice___Leaf_smut`,
`Rice___healthy`. No verified CIBRC/ICAR chemical treatment registry has been
approved for Rice, so chemical advice is withheld
(`VERIFIED_RECOMMENDATION_UNAVAILABLE`). General non-chemical cultural management
guidance is provided for each rice disease class.

Rice, Cotton, Mango, Banana, and other registry entries without validated specialist
checkpoints are reported as **unsupported for disease diagnosis**. They never receive
invented disease predictions, cross-crop RAG evidence, or chemical treatment
prescriptions.

The plant identifier now uses a **15-class** set: the original 14 PlantVillage species
plus **Wheat**, which is sourced from a separate public field dataset
(`data/wheat/raw/wheat_leaf/`) because PlantVillage has no Wheat class.

Adding either a new specialist crop or a new identifier class requires validated
training data, leakage-controlled splits, and a validated specialist model; an
identifier class alone is not sufficient to claim disease support.

---

## Architecture at a glance

```text
Streamlit UI (app.py, ui/components/)
        |
        v
CropDoctorIntelligenceLayer (modules/intelligence/orchestrator.py)
        |
        +--> Leaf verification (modules/identification/leaf_verifier.py)
        +--> Plant identification + crop routing (modules/identification/)
        +--> CropDoctor diagnosis pipeline (modules/diagnosis/pipeline.py)
        |       +--> disease specialist (modules/disease_predictor.py)
        |       +--> Grad-CAM (modules/explainability/gradcam.py)
        +--> Weather (modules/weather/)
        +--> Risk engine (modules/risk/)
        +--> Crop-safe RAG (modules/rag/)
        +--> Safe treatment recommendations (modules/treatment/)
        +--> What-If simulation (modules/simulation/)
```

Detailed component responsibilities and data flow: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quick start

### 1. Requirements

- Python 3.9+ (the verified environment for this repository is Python 3.9 on macOS).
- The model checkpoints already present under `models/`.
- A persistent ChromaDB store under `data/vector_store/` containing the verified
  agricultural knowledge chunks.
- Internet access for live weather and, optionally, Gemini answers. The RAG layer
  has an offline fallback when Gemini is unavailable.

### 2. Install dependencies

```bash
cd "/Users/kittu/Desktop/major projct"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 3. Configure environment variables

Create `.env` from `.env.example` and set only the values you need:

```dotenv
# Optional: Gemini grounded answer generation
GEMINI_API_KEY=your_key_here
# Optional: Open-Meteo does not require a key, but the client accepts configuration
WEATHER_API_KEY=
# Optional: override server-side preprocessing policy (letterbox or baseline)
DEFAULT_PREPROCESSING_MODE=letterbox
# Optional: point dataset scripts at a PlantVillage cache outside the default
# ~/.cache/kagglehub/... location
PLANTVILLAGE_ROOT=/path/to/PlantVillage
```

Never commit `.env`, API keys, model checkpoints, datasets, or the vector store.

### 4. Launch the application

```bash
python3 -m streamlit run app.py
```

If the `streamlit` executable is on your `PATH`, the equivalent command is:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`).

### 5. Optional: rebuild the RAG index

Only do this when verified knowledge documents change. This clears and rebuilds the
persistent ChromaDB collection:

```bash
python3 scripts/ingest_knowledge_base.py
```

The included store currently contains **82 verified chunks: 45 Potato and 37 Tomato**.
Do not add unverified agricultural documents to the production collection.

---

## Testing

Run the complete suite:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Run the crop-safety regression suite specifically:

```bash
python3 -m unittest tests.test_rag_crop_safety -v
```

Check application imports and Streamlit startup:

```bash
python3 -c "import app; print('APP IMPORT OK')"
python3 -m streamlit run app.py --server.headless=true --server.port=8765
```

The current verified result is **201 tests passing**. See
[`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) for exact commands, environment, and
known warnings.

---

## Project layout

```text
app.py                          Streamlit entry point
config/settings.py              Paths, credentials, preprocessing policy
modules/
  crops/                        Crop registry and support status
  identification/               Leaf verifier, plant identifier, router
  diagnosis/                    End-to-end diagnosis pipeline
  disease_predictor.py          Crop-specific specialist inference
  explainability/               Grad-CAM
  intelligence/                 Context builder, query generation, orchestration
  rag/                          Ingestion, chunking, embeddings, ChromaDB, retrieval, answers
  weather/                      Open-Meteo client and location resolution
  risk/                         Rule-based epidemiological risk engine
  simulation/                   What-If weather scenarios
  treatment/                    Verified treatment registry and safety gates
  dataset_paths.py              Portable manifest-path resolution
ui/components/                  Streamlit views
tests/                          Unit and integration tests
scripts/                        Training, evaluation, ingestion, and utility scripts
data/knowledge/                 Verified agricultural source documents
data/vector_store/              Persistent ChromaDB store (not committed)
models/                         Checkpoints and prototypes (not committed)
docs/                           Architecture, test report, presentation outline
```

---

## Data and manifest portability

Dataset manifests no longer embed machine-specific absolute paths:

- Project-owned data uses project-relative paths such as
  `data/field_eval/raw_plantdoc/...`.
- Kaggle-cache data uses `~/.cache/kagglehub/...`.

Consumers resolve both forms with
`modules.dataset_paths.resolve_dataset_path()`. Generators write portable values with
`modules.dataset_paths.to_portable_path()`. Set `PLANTVILLAGE_ROOT` to override the
Kaggle-cache location for training or evaluation scripts.

---

## Safety and scientific constraints

1. **Leaf-first gating.** Every routing mode, including a manual unsupported-crop
   selection, runs leaf validation before returning a decision.
2. **No fabricated confidence.** A manually selected unsupported crop reports
   `confidence = 0.0` because no specialist assessed it.
3. **No cross-crop RAG.** A crop-filtered retrieval executes exactly one query.
   Chunks with missing or mismatched crop metadata are rejected. Zero compatible
   results produce the insufficient-knowledge response rather than an unfiltered
   fallback.
4. **No unsafe chemicals.** Chemical treatments are withheld for unsupported crops,
   unknown species, healthy plants, rejected predictions, and low-confidence
   predictions.
5. **Weather vs simulation.** Actual weather and hypothetical What-If weather are
   labelled separately throughout the UI and payload.
6. **Thresholds are not tuned for presentation.** OOD, confidence, and treatment
   safety thresholds remain at their validated values.
7. **No dataset label correction.** Ambiguous dataset labels are not silently
   "fixed"; they are reported or excluded under documented rules.

---

## Known limitations

- The Maize/Wheat treatment registries are not yet populated, so chemical advice is
  withheld with `VERIFIED_RECOMMENDATION_UNAVAILABLE` even for supported diagnoses.
- PlantVillage test accuracy is **not** field accuracy; field evaluation on PlantDoc
  real-world photographs is underway.
- Experimental field-adapted checkpoints exist for research comparison only and must
  not become production defaults without independent validation.
- Gemini answers depend on API availability; the deterministic offline fallback is
  intentionally conservative.

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — components, data flow, contracts,
  safety design, and extension points.
- [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) — exact test and smoke-check results,
  environment, warnings, and unresolved issues.
- [`docs/PRESENTATION_OUTLINE.md`](docs/PRESENTATION_OUTLINE.md) — final-project
  presentation outline and demo script.
