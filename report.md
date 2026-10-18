# Face Search Corrective Validation Report

## Outcome

The low detector precision/recall was primarily caused by a SCRFD decoding bug. The decoder placed each anchor center at `(grid + 0.5) * stride`; this export uses `grid * stride`. The half-cell offset displaced boxes and landmarks by 4, 8, or 16 pixels depending on the feature stride, which caused valid detections to miss the IoU match and created false positives. The corrected decoder now passes a synthetic box/landmark sanity check.

The official InsightFace SCRFD decoder also uses grid centers multiplied directly by stride, without the half-cell offset: [official decoder](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/scrfd.py#L3329-L3347).

## Validation data

- WIDER FACE validation: 3,226 images and 39,123 valid ground-truth faces.
- Ground-truth faces below short-side 20 pixels: 23,051, or 58.919%.
- WIDER short-side quantiles: q10=6, q25=9, q50=16, q75=31, q90=59 pixels.
- VGGFace2 subset: 100 identities and 2,000 images, split into 100 queries and 1,900 gallery images.
- Smoke event: 80 gallery photos from 10 identities and 10 query selfies.

## Corrective changes

- `src/face_encoder.py` now decodes SCRFD anchor centers as `grid * stride` for both boxes and landmarks.
- `scripts/tune.py` evaluates 75 detector configurations: sizes 640/960/1280, thresholds 0.30–0.70, and minimum face sizes 0/8/12/16/20.
- Every detector row includes full precision, recall, F1, FP/image, latency, ground-truth size counts, eligible TP/FN counts, and size-restricted recall. The complete grid is in `storage/tuning_detector.csv`.
- Production selection is restricted to the native-compatible 640 export, with precision at least 0.80 and FP/image at most 3.0; full recall is then maximized.
- VGGFace2 uses one primary face per image, selected by largest bounding box. Production event indexing still stores every detected face.

## Detector results

The old 640/0.30/min-size-20 row is shown only as a historical comparison. Its metrics are invalid as a final detector result because they used the incorrect decoder.

| Configuration | Status | Precision | Recall | F1 | FP/image | Latency |
|---|---|---:|---:|---:|---:|---:|
| Old 640 / 0.30 / 20 | invalid old decoder | 0.4987 | 0.2134 | 0.2989 | 2.6017 | 138.12 ms |
| New 640 / 0.30 / 0 | selected production | 0.8356 | 0.6497 | 0.7310 | 1.5505 | 110.02 ms |
| New 640 / 0.30 / 20 | diagnostic | 0.9027 | 0.3775 | 0.5323 | 0.4935 | 110.02 ms |
| New 960 / 0.30 / 0 | diagnostic, incompatible | 0.8247 | 0.7520 | 0.7867 | 1.9386 | 262.74 ms |
| New 1280 / 0.30 / 0 | diagnostic, incompatible | 0.8171 | 0.7849 | 0.8007 | 2.1311 | 488.49 ms |

Selected native-640 WIDER counts are TP=25,417, FP=5,002, and FN=13,706. With min face size 0, no ground-truth faces are excluded and full recall equals size-restricted recall. For the min-size-20 diagnostic, full-GT matching yields TP=14,768, but the restricted numerator is the separately matched eligible-GT count: eligible TP=14,219, eligible FN=1,853, and eligible GT=16,072. Therefore restricted recall is exactly 14,219 / 16,072 = 0.8847063. The full TP value must not be paired with the eligible denominator; the filter also excludes 23,051 tiny faces before evaluation.

The 960/1280 rows were retained for diagnosis but not selected. The pinned ONNX export declares native-640 output sizes; larger inputs produced repeated ONNX Runtime output-shape warnings and are not production-compatible for this model. The larger sizes therefore cannot be treated as a safe recall improvement.

## Retrieval grid

- 2,000 records examined; 10 had no detectable face and 0 failed.
- 1,725 images had one detection; 265 had multiple detections.
- 1,990 embeddings cached: 100 query and 1,890 gallery vectors.
- 61 similarity thresholds from 0.20 through 0.80 were evaluated.
- Selected threshold: **0.23**, the highest-recall threshold satisfying precision at least 0.95.

This threshold sweep was rerun after production retrieval changed to threshold-complete FAISS range search. The existing corrected cache was reused unchanged. All 61 threshold TP/FP/FN totals matched an in-memory `IndexFlatIP.range_search` cross-check with no top-k truncation, so the selected threshold and metrics remain unchanged.

Selected retrieval metrics:

- Pair precision: 0.9606.
- Pair recall: 0.8392.
- F1: 0.8958.
- F2: 0.8609.
- TP=1,586, FP=65, FN=304.
- Recall@1/5/10: 0.0492 / 0.2429 / 0.4822.
- Hit@1/5/10: 0.93 / 0.94 / 0.96.

Recall@K is the fraction of relevant gallery images returned, not the fraction of queries having one correct result. With about 18.9 gallery images per identity, Recall@1 is naturally capped near 0.053 even when the top result is correct; Hit@1=0.93 is the corresponding query-level result.

Production search now uses FAISS `IndexFlatIP.range_search` at the final similarity threshold, then deduplicates photo paths and sorts by score. The configured `top_k=100` is retained for compatibility/display purposes but no longer caps the number of valid production matches.

## Final configuration

`storage/best_config.json` now contains:

```json
{
  "det_size": 640,
  "det_threshold": 0.3,
  "min_face_size": 0,
  "similarity_threshold": 0.23,
  "embedding_dim": 512,
  "top_k": 100,
  "detector_model": "scrfd_10g.onnx",
  "embedding_model": "arcface_w600k_r50.onnx"
}
```

The retrieve-all validation rerun does not alter `storage/best_config.json`; the similarity threshold remains 0.23. Detector tuning was not rerun, and the corrected VGGFace2 embedding cache was reused.

## Event index and smoke validation

The real index was rebuilt from `data/event_photos/` after tuning:

- 80 images processed, 0 failed, 1 no-face image.
- 92 face vectors indexed.
- Reload check: FAISS `ntotal=92`, metadata entries=92.
- 10 smoke queries returned 66 correct results from 80 possible gallery images.
- Smoke precision: 100%; gallery recall: 82.5%; top-1 accuracy: 90%.
- Wrong-person results: 0; duplicate result paths: 0; queries with matches: 9/10.
- Mean query latency: 0.387 seconds; maximum cold-start latency: 2.946 seconds.

Temporary FastAPI integration checks also passed without altering the real event directory: a known query returned HTTP 200 with matches, an unrelated VGGFace2 image returned HTTP 200 with zero matches, a blank query returned HTTP 400, and a multi-face WIDER image produced 10 event vectors. Re-uploading the same content was skipped as a duplicate, the no-face upload was accepted and reported, the multi-face upload was searchable using one primary query face, and the temporary index reloaded with 102/102 FAISS-to-metadata consistency.

## Root-cause assessment

1. **Primary defect:** the SCRFD anchor-center convention was wrong. This explains the large precision and recall failure and is fixed.
2. **Dataset effect:** WIDER contains many faces smaller than 20 pixels. A min-size-20 filter is a deliberate recall trade-off, not a neutral setting; the selected configuration removes that filter.
3. **Model/export limitation:** the larger input sizes are not compatible with the pinned static-output export, so their raw WIDER scores are diagnostic only.
4. **Retrieval interpretation:** low Recall@1 was a denominator effect from many relevant gallery images per query; query-level Hit@1 remains high.

## Updated artifacts

- `src/face_encoder.py`
- `scripts/tune.py`
- `storage/tuning_detector.csv`
- `storage/tuning_retrieval.csv`
- `storage/validation_embeddings.npz`
- `storage/best_config.json`
- `storage/faces.faiss`
- `storage/metadata.json`
- `storage/tuned_metrics.json`
- `storage/tuned_search.txt`
- `src/search_engine.py`
- `src/vector_store.py`
- `tests/test_search_all_matches.py`
