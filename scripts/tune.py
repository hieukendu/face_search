"""Validation and parameter search for the face-search MVP."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_encoder import FaceEncoder


def load_validation_dataset(dataset_path):
    """
    Load validation dataset.

    Dataset cần biết:
        - ảnh nào
        - người nào trong ảnh
        - ground truth identity
    """
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Validation dataset does not exist: {dataset_path}")

    metadata_csv = root / "metadata.csv"
    if metadata_csv.is_file():
        records = []
        with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                relative_file = row.get("file") or row.get("image") or row.get("path")
                identity = row.get("identity") or row.get("label")
                if not relative_file or identity is None:
                    continue
                file_path = Path(relative_file)
                if not file_path.is_absolute():
                    file_path = root / file_path
                records.append(
                    {
                        "path": str(file_path),
                        "file": str(relative_file),
                        "identity": str(identity),
                        "split": row.get("split", ""),
                    }
                )
        split = {}
        split_path = root / "split.json"
        if split_path.is_file():
            try:
                with split_path.open("r", encoding="utf-8") as handle:
                    split = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Unable to read split metadata: {exc}") from exc
        return {"kind": "recognition", "root": str(root), "records": records, "split": split}

    annotation_candidates = [
        root / "annotations" / "wider_face_val_bbx_gt.txt",
        root / "wider_face_val_bbx_gt.txt",
    ]
    annotation_candidates.extend(sorted(root.rglob("*bbx_gt.txt")))
    annotation_path = next((path for path in annotation_candidates if path.is_file()), None)
    if annotation_path is None:
        raise FileNotFoundError(
            f"Could not find metadata.csv or WIDER FACE annotations under {dataset_path}"
        )

    records = []
    with annotation_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    cursor = 0
    while cursor < len(lines):
        relative_file = lines[cursor]
        cursor += 1
        if cursor >= len(lines):
            break
        try:
            box_count = int(lines[cursor])
        except ValueError:
            continue
        cursor += 1
        boxes = []
        for _ in range(box_count):
            if cursor >= len(lines):
                break
            values = lines[cursor].split()
            cursor += 1
            if len(values) < 4:
                continue
            x, y, width, height = [float(value) for value in values[:4]]
            invalid = int(values[7]) if len(values) > 7 else 0
            if invalid:
                continue
            boxes.append([x, y, x + width, y + height])
        relative_path = Path(relative_file)
        candidates = [
            root / "images" / relative_path,
            root / relative_path,
            annotation_path.parent / relative_path,
        ]
        image_path = next((path for path in candidates if path.is_file()), candidates[0])
        records.append(
            {
                "path": str(image_path),
                "file": str(relative_file),
                "boxes": boxes,
            }
        )
    return {"kind": "detection", "root": str(root), "records": records}


def generate_detector_configs():
    """
    Sinh các combination detector.

    Ví dụ:

        det_size:
            640
            960
            1280

        det_threshold:
            .3
            .4
            .5

        min_face_size:
            0
            8
            12
            16
            20
    """
    return [
        {
            "det_size": det_size,
            "det_threshold": det_threshold,
            "min_face_size": min_face_size,
        }
        for det_size, det_threshold, min_face_size in itertools.product(
            (640, 960, 1280),
            (0.30, 0.40, 0.50, 0.60, 0.70),
            (0, 8, 12, 16, 20),
        )
    ]


def _image_path(record):
    return record.get("path") or record.get("file") or record.get("image")


def _face_area(face):
    """Return a face bounding-box area for primary-face selection."""
    bbox = face.get("bbox", [])
    if len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _match_detections(detections, ground_truth):
    """Greedily match scored detections to ground-truth boxes at IoU 0.5."""
    matched_ground_truth = set()
    true_positives = 0
    for detection in sorted(detections, key=lambda item: item.get("score", 0), reverse=True):
        overlaps = [
            (_box_iou(detection["bbox"], box), index)
            for index, box in enumerate(ground_truth)
            if index not in matched_ground_truth
        ]
        best_iou, best_index = max(overlaps, default=(0.0, None))
        if best_index is not None and best_iou >= 0.5:
            matched_ground_truth.add(best_index)
            true_positives += 1
    return true_positives


def detector_decode_sanity_check():
    """Verify the SCRFD decoder's anchor-center and landmark geometry."""
    probe = object.__new__(FaceEncoder)
    bbox = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    score = np.asarray([[0.9]], dtype=np.float32)
    landmarks = np.arange(10, dtype=np.float32).reshape(1, 10)
    decoded_boxes, decoded_scores, decoded_landmarks = probe._decode_level(
        bbox,
        score,
        landmarks,
        input_height=8,
        input_width=8,
    )
    expected_box = np.asarray([[-8.0, -16.0, 24.0, 32.0]], dtype=np.float32)
    expected_landmarks = (landmarks.reshape(1, 5, 2) * 8.0).astype(np.float32)
    if not np.allclose(decoded_boxes, expected_box):
        raise AssertionError(f"SCRFD box decode sanity check failed: {decoded_boxes}")
    if not np.allclose(decoded_landmarks, expected_landmarks):
        raise AssertionError(f"SCRFD landmark decode sanity check failed: {decoded_landmarks}")
    if not np.allclose(decoded_scores, score.reshape(-1)):
        raise AssertionError(f"SCRFD score decode sanity check failed: {decoded_scores}")
    return {
        "box": decoded_boxes[0].tolist(),
        "landmark_first": decoded_landmarks[0, 0].tolist(),
    }


def evaluate_detector(
    encoder,
    dataset
):
    """
    Đánh giá detector.

    Metrics:
        detection recall
        false positives
        runtime
    """
    records = dataset.get("records", dataset) if isinstance(dataset, dict) else dataset
    total_ground_truth = 0
    true_positives = 0
    false_positives = 0
    processed = 0
    failed = 0
    elapsed = 0.0
    for record in records:
        ground_truth = record.get("boxes")
        if ground_truth is None:
            continue
        image = cv2.imread(str(_image_path(record)), cv2.IMREAD_COLOR)
        if image is None:
            failed += 1
            continue
        started = time.perf_counter()
        try:
            detections = encoder.detect_faces(image)
        except Exception:
            failed += 1
            continue
        elapsed += time.perf_counter() - started
        processed += 1
        total_ground_truth += len(ground_truth)
        matched_ground_truth = set()
        for detection in sorted(detections, key=lambda item: item.get("score", 0), reverse=True):
            overlaps = [
                (_box_iou(detection["bbox"], box), index)
                for index, box in enumerate(ground_truth)
                if index not in matched_ground_truth
            ]
            best_iou, best_index = max(overlaps, default=(0.0, None))
            if best_index is not None and best_iou >= 0.5:
                matched_ground_truth.add(best_index)
                true_positives += 1
            else:
                false_positives += 1

    false_negatives = max(0, total_ground_truth - true_positives)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / total_ground_truth if total_ground_truth else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "images": processed,
        "failed_images": failed,
        "ground_truth_faces": total_ground_truth,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "detection_precision": precision,
        "detection_recall": recall,
        "f1": f1,
        "false_positives_per_image": false_positives / processed if processed else 0.0,
        "latency_ms_per_image": (elapsed * 1000.0) / processed if processed else 0.0,
    }


def _collect_detector_runs(encoder, dataset):
    """Run one detector configuration and retain detections for threshold sweeps."""
    records = dataset.get("records", dataset) if isinstance(dataset, dict) else dataset
    runs = []
    for record in records:
        ground_truth = record.get("boxes")
        if ground_truth is None:
            continue
        image = cv2.imread(str(_image_path(record)), cv2.IMREAD_COLOR)
        if image is None:
            runs.append({"ground_truth": ground_truth, "detections": None, "elapsed": 0.0})
            continue
        started = time.perf_counter()
        try:
            detections = encoder.detect_faces(image)
        except Exception:
            detections = None
        runs.append(
            {
                "ground_truth": ground_truth,
                "detections": detections,
                "elapsed": time.perf_counter() - started,
            }
        )
    return runs


def _metrics_from_detector_runs(runs, det_threshold, min_face_size):
    """Score cached detector outputs without rerunning ONNX inference."""
    total_ground_truth = 0
    true_positives = 0
    false_positives = 0
    ground_truth_below_min_face_size = 0
    size_restricted_ground_truth = 0
    size_restricted_true_positives = 0
    processed = 0
    failed = 0
    elapsed = 0.0
    for run in runs:
        ground_truth = run["ground_truth"]
        detections = run["detections"]
        if detections is None:
            failed += 1
            continue
        processed += 1
        elapsed += float(run["elapsed"])
        total_ground_truth += len(ground_truth)
        eligible_ground_truth = []
        for box in ground_truth:
            short_side = min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
            if short_side < float(min_face_size):
                ground_truth_below_min_face_size += 1
            else:
                eligible_ground_truth.append(box)
        size_restricted_ground_truth += len(eligible_ground_truth)
        filtered = []
        for detection in detections:
            if float(detection.get("score", 0.0)) < float(det_threshold):
                continue
            bbox = detection.get("bbox", [])
            if len(bbox) != 4:
                continue
            if min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])) < float(min_face_size):
                continue
            filtered.append(detection)
        image_true_positives = _match_detections(filtered, ground_truth)
        true_positives += image_true_positives
        false_positives += len(filtered) - image_true_positives
        size_restricted_true_positives += _match_detections(
            filtered,
            eligible_ground_truth,
        )
    false_negatives = max(0, total_ground_truth - true_positives)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / total_ground_truth if total_ground_truth else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    if size_restricted_true_positives > size_restricted_ground_truth:
        raise RuntimeError(
            "Restricted true positives cannot exceed eligible ground-truth faces"
        )
    size_restricted_false_negatives = max(
        0,
        size_restricted_ground_truth - size_restricted_true_positives,
    )
    size_restricted_recall = (
        size_restricted_true_positives / size_restricted_ground_truth
        if size_restricted_ground_truth
        else 0.0
    )
    return {
        "images": processed,
        "failed_images": failed,
        "ground_truth_faces": total_ground_truth,
        "gt_faces_below_min_face_size": ground_truth_below_min_face_size,
        "gt_faces_below_min_face_size_pct": (
            ground_truth_below_min_face_size / total_ground_truth
            if total_ground_truth
            else 0.0
        ),
        "gt_faces_at_or_above_min_face_size": size_restricted_ground_truth,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "eligible_ground_truth_faces": size_restricted_ground_truth,
        "eligible_true_positives": size_restricted_true_positives,
        "eligible_false_negatives": size_restricted_false_negatives,
        "size_restricted_true_positives": size_restricted_true_positives,
        "size_restricted_recall": size_restricted_recall,
        "detection_precision": precision,
        "detection_recall": recall,
        "f1": f1,
        "false_positives_per_image": false_positives / processed if processed else 0.0,
        "latency_ms_per_image": (elapsed * 1000.0) / processed if processed else 0.0,
    }


def cache_embeddings(
    encoder,
    dataset
):
    """
    Sau khi chọn detector tốt cho VGGFace2:

        detect
        align
        ArcFace

    rồi cache đúng một embedding primary cho mỗi ảnh.

    Nhờ vậy khi tune similarity threshold
    không cần chạy ArcFace lại.
    """
    records = dataset.get("records", dataset) if isinstance(dataset, dict) else dataset
    embeddings = []
    labels = []
    image_paths = []
    splits = []
    skipped = 0
    failed = 0
    no_face = 0
    one_detection = 0
    multiple_detections = 0
    for record in records:
        identity = record.get("identity")
        image_path = _image_path(record)
        if identity is None or image_path is None:
            skipped += 1
            failed += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            failed += 1
            continue
        try:
            faces = encoder.encode_image(image)
        except Exception:
            skipped += 1
            failed += 1
            continue
        if not faces:
            skipped += 1
            no_face += 1
            continue
        if len(faces) == 1:
            one_detection += 1
        else:
            multiple_detections += 1

        # VGGFace2 supplies one target identity per image. Keep exactly one
        # primary face for validation, even when the detector finds multiple
        # faces. Production event indexing intentionally keeps every face and
        # does not use this validation-only path.
        primary_face = max(faces, key=_face_area)
        embeddings.append(np.asarray(primary_face["embedding"], dtype=np.float32))
        labels.append(str(identity))
        image_paths.append(str(image_path))
        splits.append(str(record.get("split", "")))

    if embeddings:
        embedding_array = np.vstack(embeddings).astype(np.float32)
    else:
        embedding_array = np.empty((0, 512), dtype=np.float32)
    return {
        "embeddings": embedding_array,
        "labels": np.asarray(labels, dtype=str),
        "image_paths": np.asarray(image_paths, dtype=str),
        "splits": np.asarray(splits, dtype=str),
        "total_records": len(records),
        "images_with_no_face": no_face,
        "images_with_one_detection": one_detection,
        "images_with_multiple_detections": multiple_detections,
        "images_failed": failed,
        "skipped": skipped,
    }


def evaluate_similarity_threshold(
    threshold,
    embeddings,
    labels=None,
    splits=None,
):
    """
    Thử một cosine similarity threshold.

    Tính:
        query-to-gallery Precision, Recall, F1, and F2
        true per-query Recall@K
        binary Hit@K
    """
    if isinstance(embeddings, dict):
        cache = embeddings
        labels = cache.get("labels", labels)
        splits = cache.get("splits", splits)
        embeddings = cache.get("embeddings")
    if labels is None or splits is None:
        raise ValueError("retrieval validation requires identity labels and query/gallery splits")
    vectors = np.asarray(embeddings, dtype=np.float32)
    labels_array = np.asarray(labels).astype(str)
    splits_array = np.asarray(splits).astype(str)
    if (
        vectors.ndim != 2
        or len(vectors) != len(labels_array)
        or len(vectors) != len(splits_array)
    ):
        raise ValueError("embeddings, labels, and splits must describe the same 2-D array")
    query_mask = splits_array == "query"
    gallery_mask = splits_array == "gallery"
    if not np.any(query_mask) or not np.any(gallery_mask):
        raise ValueError("retrieval validation requires query and gallery records")

    query_vectors = vectors[query_mask]
    gallery_vectors = vectors[gallery_mask]
    query_labels = labels_array[query_mask]
    gallery_labels = labels_array[gallery_mask]
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    gallery_norms = np.linalg.norm(gallery_vectors, axis=1, keepdims=True)
    query_norms[query_norms <= 1e-12] = 1.0
    gallery_norms[gallery_norms <= 1e-12] = 1.0
    query_vectors = query_vectors / query_norms
    gallery_vectors = gallery_vectors / gallery_norms
    similarity = query_vectors @ gallery_vectors.T
    query_count = len(query_vectors)
    gallery_count = len(gallery_vectors)
    if query_count == 0 or gallery_count == 0:
        return {
            "threshold": float(threshold),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "query_count": query_count,
            "gallery_count": gallery_count,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "hit_at_1": 0.0,
            "hit_at_5": 0.0,
            "hit_at_10": 0.0,
        }

    positives = query_labels[:, None] == gallery_labels[None, :]
    predicted = similarity >= float(threshold)
    true_positive = int(np.sum(predicted & positives))
    false_positive = int(np.sum(predicted & ~positives))
    false_negative = int(np.sum(~predicted & positives))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5.0 * precision * recall / (4.0 * precision + recall) if 4.0 * precision + recall else 0.0

    ranking_recall = {1: [], 5: [], 10: []}
    ranking_hit = {1: [], 5: [], 10: []}
    for query_index in range(query_count):
        relevant = positives[query_index]
        relevant_count = int(np.sum(relevant))
        if relevant_count == 0:
            continue
        order = np.argsort(-similarity[query_index], kind="stable")
        for k in ranking_recall:
            top_indices = order[:k]
            relevant_retrieved = int(np.sum(relevant[top_indices]))
            ranking_recall[k].append(relevant_retrieved / relevant_count)
            ranking_hit[k].append(float(relevant_retrieved > 0))

    ranking_metrics = {}
    for k in ranking_recall:
        ranking_metrics[f"recall_at_{k}"] = (
            float(np.mean(ranking_recall[k])) if ranking_recall[k] else 0.0
        )
        ranking_metrics[f"hit_at_{k}"] = (
            float(np.mean(ranking_hit[k])) if ranking_hit[k] else 0.0
        )

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "query_count": query_count,
        "gallery_count": gallery_count,
        **ranking_metrics,
    }


def grid_search_threshold(
    embeddings,
    labels=None,
    splits=None,
):
    """
    Sweep threshold.

    Ví dụ:

        .30
        .31
        .32
        ...
        .70

    Chọn threshold tốt nhất.
    """
    thresholds = np.round(np.arange(0.20, 0.80 + 0.001, 0.01), 2)
    return [
        evaluate_similarity_threshold(float(threshold), embeddings, labels, splits)
        for threshold in thresholds
    ]


def select_best_config(results):
    """
    Chọn hyperparameter tốt nhất.

    Ví dụ objective:

        maximize F2

    hoặc:

        maximize Recall
        subject to Precision >= 95%
    """
    if not results:
        raise ValueError("No tuning results available")
    precision_constrained = [result for result in results if float(result.get("precision", 0.0)) >= 0.95]
    if precision_constrained:
        return dict(
            max(
                precision_constrained,
                key=lambda result: (
                    float(result.get("recall", 0.0)),
                    float(result.get("f2", 0.0)),
                    float(result.get("precision", 0.0)),
                    -float(result.get("threshold", 1.0)),
                ),
            )
        )
    return dict(
        max(
            results,
            key=lambda result: (
                float(result.get("f2", 0.0)),
                float(result.get("recall", 0.0)),
                float(result.get("precision", 0.0)),
                -float(result.get("threshold", 1.0)),
            ),
        )
    )


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_detector_config(results, compatible_sizes=(640,)):
    if not results:
        raise ValueError("Detector grid produced no results")
    compatible = [
        result
        for result in results
        if int(result.get("det_size", -1)) in set(compatible_sizes)
    ]
    if not compatible:
        raise ValueError("Detector grid produced no compatible production configurations")

    # Production selection is restricted to the native compatible input size.
    # Within that pool, keep a reasonable precision/false-positive envelope,
    # then favor full-WIDER recall and F1. The restricted-size recall remains a
    # diagnostic only and is never used as a replacement for full recall.
    quality_pool = [
        result
        for result in compatible
        if float(result.get("detection_precision", 0.0)) >= 0.80
        and float(result.get("false_positives_per_image", float("inf"))) <= 3.0
    ]
    candidates = quality_pool or compatible
    selected = max(
        candidates,
        key=lambda result: (
            float(result.get("detection_recall", 0.0)),
            float(result.get("f1", 0.0)),
            float(result.get("detection_precision", 0.0)),
            -float(result.get("false_positives_per_image", 0.0)),
            -float(result.get("latency_ms_per_image", 0.0)),
            -float(result.get("det_threshold", 1.0)),
            -float(result.get("min_face_size", 1_000_000.0)),
        ),
    )
    return {
        **dict(selected),
        "selection_pool": "quality-constrained-compatible"
        if quality_pool
        else "compatible-fallback",
        "selection_precision_floor": 0.80,
        "selection_fp_per_image_ceiling": 3.0,
    }


def _make_encoder(args, config):
    size = int(config["det_size"])
    return FaceEncoder(
        args.detector_path,
        args.embedder_path,
        det_size=(size, size),
        det_threshold=float(config["det_threshold"]),
        min_face_size=float(config["min_face_size"]),
    )


def main():
    """
    Pipeline tuning:

        validation data
        -> tune detector
        -> freeze detector
        -> cache embeddings
        -> tune similarity threshold
        -> print best config
    """
    parser = argparse.ArgumentParser(description="Tune detector and retrieval thresholds")
    parser.add_argument("--wider-path", default="data/validation/wider_face")
    parser.add_argument("--vgg-path", default="data/validation/vggface2_subset")
    parser.add_argument("--detector-path", default="models/scrfd_10g.onnx")
    parser.add_argument("--embedder-path", default="models/arcface_w600k_r50.onnx")
    parser.add_argument("--storage", default="storage")
    parser.add_argument("--max-detector-images", type=int, default=None)
    parser.add_argument("--max-recognition-images", type=int, default=None)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Rebuild VGGFace2 embeddings and rerun retrieval tuning using the existing detector config",
    )
    args = parser.parse_args()

    storage = Path(args.storage)
    storage.mkdir(parents=True, exist_ok=True)
    old_detector = None
    old_config_path = storage / "best_config.json"
    if old_config_path.is_file():
        try:
            with old_config_path.open("r", encoding="utf-8") as handle:
                old_config = json.load(handle)
            old_detector = {
                key: old_config[key]
                for key in ("det_size", "det_threshold", "min_face_size")
                if key in old_config
            }
        except (OSError, json.JSONDecodeError):
            old_detector = None
    if old_detector:
        print(f"old detector config: {old_detector}", flush=True)
    if args.retrieval_only:
        config_path = storage / "best_config.json"
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                existing_config = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read existing detector config {config_path}: {exc}") from exc
        required_detector_keys = ("det_size", "det_threshold", "min_face_size")
        missing = [key for key in required_detector_keys if key not in existing_config]
        if missing:
            raise ValueError(f"Existing config is missing detector keys: {', '.join(missing)}")
        best_detector = {
            "det_size": int(existing_config["det_size"]),
            "det_threshold": float(existing_config["det_threshold"]),
            "min_face_size": int(existing_config["min_face_size"]),
        }
        print(f"using existing detector config: {best_detector}", flush=True)
    else:
        wider = load_validation_dataset(args.wider_path)
        if args.max_detector_images is not None:
            wider = {**wider, "records": wider["records"][: args.max_detector_images]}

        detector_sizes = (640, 960, 1280)
        detector_thresholds = (0.30, 0.40, 0.50, 0.60, 0.70)
        detector_min_face_sizes = (0, 8, 12, 16, 20)
        detector_configs = [
            {
                "det_size": size,
                "det_threshold": threshold,
                "min_face_size": min_face_size,
            }
            for size, threshold, min_face_size in itertools.product(
                detector_sizes,
                detector_thresholds,
                detector_min_face_sizes,
            )
        ]
        detector_results = []
        # Confidence and minimum-size settings are post-processing filters. Cache
        # one low-threshold, zero-size-filter detector pass per image size so all
        # threshold x min-face points reuse the same expensive ONNX inference.
        detector_runs_by_size = {}
        for size_index, det_size in enumerate(detector_sizes, start=1):
            base_config = {"det_size": det_size, "det_threshold": 0.30, "min_face_size": 0}
            print(f"detector inference pass {size_index}/{len(detector_sizes)}: {base_config}", flush=True)
            encoder = _make_encoder(args, base_config)
            detector_runs_by_size[det_size] = _collect_detector_runs(encoder, wider)
            for config in [item for item in detector_configs if item["det_size"] == det_size]:
                metrics = _metrics_from_detector_runs(
                    detector_runs_by_size[det_size],
                    config["det_threshold"],
                    config["min_face_size"],
                )
                detector_results.append(
                    {
                        **config,
                        "production_compatible": int(det_size == 640),
                        **metrics,
                    }
                )
                print(
                    f"  config={config['det_threshold']:.2f}/{config['min_face_size']} "
                    f"recall={metrics['detection_recall']:.4f} "
                    f"precision={metrics['detection_precision']:.4f} "
                    f"restricted_recall={metrics['size_restricted_recall']:.4f} "
                    f"latency_ms={metrics['latency_ms_per_image']:.2f}",
                    flush=True,
                )

        _write_csv(storage / "tuning_detector.csv", detector_results)
        best_detector = _select_detector_config(detector_results)
        print(f"best detector: {best_detector}", flush=True)

        # The detector/configuration change invalidates every dependent
        # recognition artifact. Remove only these generated cache/results;
        # source images and production event photos remain untouched.
        for stale_path in (
            storage / "validation_embeddings.npz",
            storage / "tuning_retrieval.csv",
        ):
            if stale_path.is_file():
                stale_path.unlink()
                print(f"invalidated: {stale_path}", flush=True)
        sanity = detector_decode_sanity_check()
        print(f"detector decode sanity: passed {sanity}", flush=True)

    recognition = load_validation_dataset(args.vgg_path)
    if args.max_recognition_images is not None:
        recognition = {**recognition, "records": recognition["records"][: args.max_recognition_images]}
    frozen_encoder = _make_encoder(args, best_detector)
    cache = cache_embeddings(frozen_encoder, recognition)
    np.savez_compressed(
        storage / "validation_embeddings.npz",
        embeddings=cache["embeddings"],
        identity=cache["labels"],
        image_path=cache["image_paths"],
        split=cache["splits"],
    )
    print(f"recognition records: {cache['total_records']}", flush=True)
    print(f"images with no face: {cache['images_with_no_face']}", flush=True)
    print(f"images with one detection: {cache['images_with_one_detection']}", flush=True)
    print(f"images with multiple detections: {cache['images_with_multiple_detections']}", flush=True)
    print(f"images failed: {cache['images_failed']}", flush=True)
    print(f"cached embeddings: {len(cache['embeddings'])}", flush=True)
    retrieval_results = grid_search_threshold(
        cache["embeddings"],
        cache["labels"],
        cache["splits"],
    )
    _write_csv(storage / "tuning_retrieval.csv", retrieval_results)
    best_retrieval = select_best_config(retrieval_results)
    best_config = {
        "det_size": int(best_detector["det_size"]),
        "det_threshold": float(best_detector["det_threshold"]),
        "min_face_size": int(best_detector["min_face_size"]),
        "similarity_threshold": float(best_retrieval["threshold"]),
        "embedding_dim": 512,
        "top_k": 100,
        "detector_model": Path(args.detector_path).name,
        "embedding_model": Path(args.embedder_path).name,
    }
    with (storage / "best_config.json").open("w", encoding="utf-8") as handle:
        json.dump(best_config, handle, ensure_ascii=False, indent=2)
    print(f"best retrieval: {best_retrieval}", flush=True)
    print(f"best config written to {storage / 'best_config.json'}", flush=True)


if __name__ == "__main__":
    main()
