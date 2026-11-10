# External LFW evaluation

This evaluation ran the production SCRFD detector, landmark alignment, and ArcFace embedder from `models/` without changing `storage/best_config.json` or the FAISS search pipeline. The source dataset was [marcelohaps/lfw on Hugging Face](https://huggingface.co/datasets/marcelohaps/lfw), whose card documents 13,233 LFW images, 5,749 identities, and the official 6,000 labeled verification pairs.

The reproducible sample used a fixed seed (`20261110`) to select 300 same-identity and 300 different-identity pairs. It contained 1,141 unique images. SCRFD/ArcFace produced embeddings for 1,140 images; one image (`Morgan_Freeman_0002.jpg`) had no detected face, so 599 pairs were scored. No image download failed.

## Production threshold

The current production threshold stayed at `0.23`:

| Metric | Result |
|---|---:|
| True positives | 297 |
| True negatives | 300 |
| False positives | 0 |
| False negatives | 2 |
| Precision | 1.0000 |
| Recall / TPR | 0.9933 |
| Specificity | 1.0000 |
| FPR | 0.0000 |
| FNR | 0.0067 |
| Accuracy | 0.9967 |
| Balanced accuracy | 0.9967 |
| F1 | 0.9966 |
| ROC-AUC | 0.9970 |

The threshold sweep found the same best F1 (`0.9966`) at `0.18`; `0.18–0.23` was a flat region on this sample. Raising the threshold to `0.35` would reduce recall to `0.9866` while still having no false positives in this sample. No threshold change was made because the requested production parameters remain unchanged.

Positive-pair cosine scores had median `0.6769` and 5th percentile `0.4803`. Negative-pair scores had median `0.0008` and 95th percentile `0.0914`; the maximum negative score was `0.1721`. This shows strong separation on LFW, but it does not prove the same separation on the unlabeled Drive event album.

## Drive transport check

Sixty LFW JPEGs were uploaded to a temporary child folder of the linked `Database` folder. A real Drive import against that child folder processed 52 images and indexed 75 faces. Eight final downloads were rejected by Google Drive/gdown while retrieving their public links; the app reported those per-file failures and continued. This was a Drive access/rate-limit issue, not an embedding failure. All 60 uploaded test files were deleted afterward; the temporary Drive folder is empty because the connector cannot delete an empty folder through its delete endpoint.

The downloaded LFW images, temporary dataset directory, model cache, and temporary import album were removed from the local machine. The full pair-level JSON is in `codex/hf-lfw-eval-20261110.json`; the reusable evaluator is `codex/hf_lfw_eval.py`.

## Interpretation

The model is strong on a standard, labeled verification benchmark and the current `0.23` threshold is not intrinsically too low there. The earlier Drive behavior cannot be assigned a true precision/recall value because the 38 event images have no identity labels. Its false-positive impression is therefore a domain/data-quality question—different lighting, glasses, group collages, small faces, and multiple people—not something this LFW score can disprove. The production detector, embedder, threshold, and retrieval code were left unchanged.
