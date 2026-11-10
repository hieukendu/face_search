"""Evaluate the production SCRFD/ArcFace pipeline on a labeled LFW subset.

The manifest is produced by the Hugging Face download step. Images are encoded
with the same detector, alignment, and ArcFace settings used by the app; pair
scores are then evaluated without changing the production index or threshold.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_encoder import FaceEncoder


_ENCODER = None
_ROOT = None


def _init_worker(root: str, detector: str, embedder: str, det_size: int, det_threshold: float, min_face_size: float):
    global _ENCODER, _ROOT
    _ROOT = Path(root)
    _ENCODER = FaceEncoder(
        detector,
        embedder,
        det_size=(det_size, det_size),
        det_threshold=det_threshold,
        min_face_size=min_face_size,
    )


def _encode_one(relative_path: str):
    path = _ROOT / "dataset" / "train" / relative_path
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return relative_path, None, 0, "image could not be decoded"
    try:
        faces = _ENCODER.encode_image(image)
        if not faces:
            return relative_path, None, 0, "no face detected"
        selected = max(
            faces,
            key=lambda face: max(0.0, face["bbox"][2] - face["bbox"][0])
            * max(0.0, face["bbox"][3] - face["bbox"][1]),
        )
        return relative_path, selected["embedding"].tolist(), len(faces), None
    except Exception as exc:  # keep one bad image from stopping the evaluation
        return relative_path, None, len(faces) if "faces" in locals() else 0, repr(exc)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positives) * len(negatives)))


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= float(threshold)
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & negative))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": 1.0 - specificity,
        "fnr": 1.0 - recall,
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": f1,
    }


def evaluate(root: Path, output: Path, workers: int) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((PROJECT_ROOT / "storage" / "best_config.json").read_text(encoding="utf-8"))
    detector = str(PROJECT_ROOT / "models" / config["detector_model"])
    embedder = str(PROJECT_ROOT / "models" / config["embedding_model"])
    image_paths = sorted({pair[key] for pair in manifest["pairs"] for key in ("image_a_path", "image_b_path")})
    started = time.perf_counter()
    context = mp.get_context("spawn")
    results = {}
    face_counts = {}
    failures = {}
    with context.Pool(
        processes=max(1, int(workers)),
        initializer=_init_worker,
        initargs=(str(root), detector, embedder, int(config["det_size"]), float(config["det_threshold"]), float(config["min_face_size"])),
    ) as pool:
        for index, (relative_path, embedding, face_count, error) in enumerate(pool.imap_unordered(_encode_one, image_paths, chunksize=4), 1):
            face_counts[relative_path] = face_count
            if error:
                failures[relative_path] = error
            else:
                results[relative_path] = np.asarray(embedding, dtype=np.float32)
            if index % 100 == 0 or index == len(image_paths):
                print(f"ENCODED {index}/{len(image_paths)}", flush=True)

    pair_rows = []
    for pair in manifest["pairs"]:
        first = results.get(pair["image_a_path"])
        second = results.get(pair["image_b_path"])
        if first is None or second is None:
            continue
        pair_rows.append({
            "pair_id": int(pair["pair_id"]),
            "fold_id": int(pair["fold_id"]),
            "is_same": int(pair["is_same"]),
            "image_a": pair["image_a_path"],
            "image_b": pair["image_b_path"],
            "score": float(np.dot(first, second)),
        })
    labels = np.asarray([row["is_same"] for row in pair_rows], dtype=np.int8)
    scores = np.asarray([row["score"] for row in pair_rows], dtype=np.float32)
    thresholds = np.round(np.arange(0.10, 0.901, 0.01), 2)
    sweep = [_metrics(labels, scores, threshold) for threshold in thresholds]
    best_f1 = max(sweep, key=lambda row: (row["f1"], row["precision"], row["recall"], -row["threshold"]))
    current = _metrics(labels, scores, float(config["similarity_threshold"]))
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    report = {
        "dataset": manifest["repo"],
        "sample": {
            "pairs_requested": len(manifest["pairs"]),
            "positive_pairs_requested": int(sum(int(pair["is_same"]) for pair in manifest["pairs"])),
            "negative_pairs_requested": int(sum(not int(pair["is_same"]) for pair in manifest["pairs"])),
            "unique_images": len(image_paths),
            "valid_images": len(results),
            "failed_images": len(failures),
            "valid_pairs": len(pair_rows),
            "skipped_pairs": len(manifest["pairs"]) - len(pair_rows),
        },
        "production_config": {
            "det_size": config["det_size"],
            "det_threshold": config["det_threshold"],
            "min_face_size": config["min_face_size"],
            "similarity_threshold": config["similarity_threshold"],
            "detector_model": config["detector_model"],
            "embedding_model": config["embedding_model"],
        },
        "current_threshold_metrics": current,
        "best_f1_threshold_metrics": best_f1,
        "roc_auc": _auc(labels, scores),
        "positive_score_quantiles": dict(zip(["min", "p05", "p25", "median", "p75", "p95", "max"], np.quantile(positive_scores, [0, .05, .25, .5, .75, .95, 1]).round(6).tolist())),
        "negative_score_quantiles": dict(zip(["min", "p05", "p25", "median", "p75", "p95", "max"], np.quantile(negative_scores, [0, .05, .25, .5, .75, .95, 1]).round(6).tolist())),
        "threshold_sweep": sweep,
        "failed_images": failures,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "pair_scores": pair_rows,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    result = evaluate(args.root, args.output, args.workers)
    print(json.dumps({key: result[key] for key in ("sample", "current_threshold_metrics", "best_f1_threshold_metrics", "roc_auc", "positive_score_quantiles", "negative_score_quantiles", "elapsed_seconds")}, ensure_ascii=False, indent=2))
