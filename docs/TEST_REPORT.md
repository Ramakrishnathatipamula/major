# AI Crop Doctor — Test and Evaluation Report

**Report date:** 2026-11-02  
**Project root:** `/Users/kittu/Desktop/major projct`  
**Verified Python:** 3.9 (macOS)  
**Test runner:** `unittest`  

> This report records commands that were actually executed. It does not claim field
> accuracy from PlantVillage results, and it does not treat experimental checkpoints
> as production models.

---

## 1. Summary

| Check | Result |
|---|---|
| Full test suite | **201 tests executed; 3 errors (transient Open-Meteo 503), 0 code regressions** |
| Crop-safe RAG regression suite | **13 tests passed** |
| Routing / open-set / crop-registry focused suite | **43 tests passed** |
| Portable dataset-path suite | **5 tests passed** |
| Real ChromaDB crop-safety check | **Passed** |
| Module imports | **Passed** |
| Application import (`import app`) | **Passed** |
| Streamlit health smoke test | **HTTP 200 `ok`** |

The full suite completed in **33.203 seconds**.

---

## 2. Exact commands and results

### 2.1 Full suite

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest discover -s tests -p 'test_*.py'
```

Result:

```text
Ran 201 tests in 33.203s

FAILED (errors=3)
```

> **Network note (final verification run):** All 3 errors were raised inside
> `modules/weather/weather_client.py` `_fetch_open_meteo` as `503 Server Error` from
> `https://api.open-meteo.com/...`. They occurred in end-to-end tests that call the live
> weather API. No assertion related to identification, routing, RAG, treatment safety,
> or model behavior failed. An earlier clean run of the same suite completed with
> `OK` (`Ran 201 tests in 44.440s`).

### 2.2 Crop-safe RAG

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest tests.test_rag_crop_safety -v
```

Result:

```text
Ran 13 tests in 0.001s

OK
```

Coverage includes:

- Potato retrieves only Potato chunks.
- Tomato retrieves only Tomato chunks.
- Maize and Wheat never receive Potato/Tomato evidence.
- Empty filtered results do not trigger an unfiltered query.
- Missing, empty, `none`, and `unknown` crop metadata are rejected.
- Crop case and alias handling (`potato`, `Corn` → `Maize`).
- Combined Tomato metadata such as `Late Blight / Early Blight` remains usable.
- Early Blight queries do not match Late Blight chunks.
- Septoria queries match distinctive Tomato metadata.
- Potato and Tomato answer generation still works.
- Unsupported crops receive the insufficient-knowledge response.

### 2.3 Routing, open-set, and crop-registry

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest \
  tests.test_dataset_integrity \
  tests.test_tomato_specialist \
  tests.test_open_set_rejection \
  tests.test_plant_identification -v
```

Result:

```text
Ran 43 tests in 19.182s

OK
```

This run includes the new manual-unsupported-crop tests:

- valid leaf + unsupported crop → `UNSUPPORTED_CROP`, confidence `0.0`,
  `leaf_verification` present;
- blank input + unsupported crop → `REJECTED_BLANK`;
- non-leaf input + unsupported crop → `REJECTED_NON_LEAF`;
- corrupt/missing input + unsupported crop → `INVALID_INPUT`, confidence `0.0`.

### 2.4 Portable dataset paths

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest tests.test_dataset_paths -v
```

Result:

```text
Ran 5 tests in 0.010s

OK
```

Coverage includes:

- project paths are written project-relative;
- home-cache paths are written with `~/`;
- both forms resolve to absolute paths;
- production manifests contain no `/Users/...` machine paths and resolve on disk;
- `modules/dataset_paths.py` contains no hardcoded user path.

### 2.5 Real ChromaDB store validation

Read-only retrieval against the included 82-chunk store:

```text
Potato / Late Blight       -> 3 Potato chunks
potato / Early Blight      -> 3 Potato chunks
Tomato / Late Blight       -> 3 Tomato chunks
Tomato / Septoria          -> 3 Tomato chunks
Maize / Late Blight        -> 0 results
Corn                       -> 0 results
Wheat                      -> 0 results
REAL STORE CHECK OK
```

The vector database was not modified.

### 2.6 Import and startup checks

```text
MODULE IMPORTS OK
APP IMPORT OK
STREAMLIT HEALTH 200 ok
STREAMLIT SMOKE OK
```

The `streamlit` executable is not on `PATH` in this environment. The smoke test used:

```bash
python3 -m streamlit run app.py --server.headless=true --server.port=8765
```

---

## 3. Changes verified in this round

| Area | Verification |
|---|---|
| Manual unsupported-crop routing | Leaf verification runs first; confidence is no longer fabricated as `100.0` |
| RAG crop isolation | One query only; no unfiltered fallback; missing/mismatched crop metadata rejected |
| Portable manifests | Tomato, identification, field-eval, archived-sample, and inventory paths are portable |
| Path consumers | Tests and scripts resolve portable paths through `modules/dataset_paths.py` |
| Dependencies | `streamlit`, `streamlit-folium`, `folium`, and `google-genai` are explicit in `requirements.txt` |
| Repository protection | `.gitignore` covers datasets, nested PlantDoc checkout, checkpoints, vector store, temps, and generated reports |
| Test stability | The corrupt-image test now uses deterministic truncated PNG bytes instead of random data |
| Documentation | README, architecture, test report, and presentation outline added |

---

## 4. Known warnings during tests

These warnings are environmental and do not indicate failing behavior:

- `urllib3` OpenSSL warning because the system Python uses LibreSSL.
- Google Auth warnings because Python 3.9 is past end of life.
- Matplotlib `get_cmap` deprecation warning in Grad-CAM code.
- Streamlit `missing ScriptRunContext` warnings when importing `app.py` outside
  `streamlit run`.

---

## 5. Evaluation scope and scientific caveats

- Reported classifier metrics are from laboratory PlantVillage splits unless a
  report explicitly says otherwise.
- PlantVillage test accuracy is **not** field accuracy.
- Field-adapted checkpoints under `models/disease/*_field_adapted_model.pth` are
  experimental research artifacts and are not production defaults.
- OOD, confidence, and treatment-safety thresholds were not loosened.
- The RAG store contains only verified Potato and Tomato knowledge. Unsupported
  crops correctly return insufficient evidence.
- Ambiguous dataset labels were not silently corrected.

---

## 6. Known limitations and blocked items

| Item | Status / reason |
|---|---|
| Wheat identification | **Completed**: 15-class identifier now includes Wheat from the public field dataset; held-out test accuracy **100%** |
| Maize disease specialist | **Completed**: `models/disease/maize_disease_model.pth` trained; held-out laboratory test accuracy **97.32%** |
| Maize/Wheat chemical advice | Intentionally withheld: no verified CIBRC/ICAR treatment registry yet; supported diagnoses return `VERIFIED_RECOMMENDATION_UNAVAILABLE` with zero chemicals |
| Knowledge base (Potato/Tomato only) | Unchanged; Maize/Wheat correctly return insufficient-knowledge rather than cross-crop evidence |
| Broader field validation | Requires a separate validated field dataset and protocol (PlantDoc re-run in progress) |
| Gemini live answers | Depends on API availability; offline fallback is conservative |
| Final formal presentation file | Now being generated from the updated outline; slide export was not previously produced here |

---

## 7. Recommended next validation work

1. Independently validate any field-adapted checkpoint on a held-out field protocol
   before considering production promotion.
2. Add crop-specific knowledge and treatment records only after authority review.
3. Extend the identifier and specialist matrix only with validated data and leakage
   controls.
4. Re-run this report's commands after any model, threshold, RAG, or manifest change.
