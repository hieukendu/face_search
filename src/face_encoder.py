"""Face detection, alignment, and ArcFace embedding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised when setup is incomplete
    ort = None


class FaceEncoder:
    """Run the SCRFD -> alignment -> ArcFace pipeline."""

    _ARC_FACE_TEMPLATE = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        detector_path: str,
        embedder_path: str,
        det_size=(640, 640),
        det_threshold=0.5,
        min_face_size=20,
    ):
        """
        Load:
            - SCRFD face detector
            - ArcFace embedding model

        Đồng thời lưu các parameter detector.
        """
        if ort is None:
            raise ImportError(
                "onnxruntime is required. Install dependencies from requirements.txt."
            )

        detector_file = Path(detector_path)
        embedder_file = Path(embedder_path)
        if not detector_file.is_file() or detector_file.stat().st_size == 0:
            raise FileNotFoundError(f"Detector model does not exist or is empty: {detector_path}")
        if not embedder_file.is_file() or embedder_file.stat().st_size == 0:
            raise FileNotFoundError(f"Embedder model does not exist or is empty: {embedder_path}")

        if len(det_size) != 2:
            raise ValueError("det_size must be a (height, width) pair")
        self.det_size = (int(det_size[0]), int(det_size[1]))
        if min(self.det_size) <= 0:
            raise ValueError("det_size values must be positive")
        self.det_threshold = float(det_threshold)
        if not 0.0 <= self.det_threshold <= 1.0:
            raise ValueError("det_threshold must be between 0 and 1")
        self.min_face_size = float(min_face_size)
        if self.min_face_size < 0:
            raise ValueError("min_face_size must be non-negative")

        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            self.detector_session = ort.InferenceSession(
                str(detector_file), providers=providers
            )
            self.embedder_session = ort.InferenceSession(
                str(embedder_file), providers=providers
            )
        except Exception as exc:  # expose a useful model-load error
            raise RuntimeError(f"Unable to load ONNX models: {exc}") from exc

        detector_inputs = self.detector_session.get_inputs()
        embedder_inputs = self.embedder_session.get_inputs()
        if not detector_inputs or not embedder_inputs:
            raise RuntimeError("ONNX model has no input tensor")
        self._detector_input_name = detector_inputs[0].name
        self._embedder_input_name = embedder_inputs[0].name
        self._detector_input_shape = tuple(detector_inputs[0].shape)
        self._embedder_input_shape = tuple(embedder_inputs[0].shape)

    @staticmethod
    def _as_bgr_image(image: np.ndarray) -> np.ndarray:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Image is empty or invalid")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must have shape H x W x 3")

        if image.dtype != np.uint8:
            if not np.issubdtype(image.dtype, np.number):
                raise ValueError("Image must contain numeric pixel values")
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _model_detector_size(self) -> tuple[int, int]:
        """Return the configured size, respecting a static ONNX input shape."""
        shape = self._detector_input_shape
        if len(shape) >= 4:
            height, width = shape[-2], shape[-1]
            if isinstance(height, (int, np.integer)) and isinstance(width, (int, np.integer)):
                if int(height) > 0 and int(width) > 0:
                    return int(height), int(width)
        return self.det_size

    def _detector_geometry(self, image_shape: tuple[int, ...]):
        source_height, source_width = int(image_shape[0]), int(image_shape[1])
        target_height, target_width = self._model_detector_size()
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, min(target_width, int(round(source_width * scale))))
        resized_height = max(1, min(target_height, int(round(source_height * scale))))
        pad_x = (target_width - resized_width) // 2
        pad_y = (target_height - resized_height) // 2
        return (
            target_height,
            target_width,
            scale,
            resized_width,
            resized_height,
            pad_x,
            pad_y,
        )

    def _preprocess_detector(self, image: np.ndarray):
        """
        Tiền xử lý ảnh trước khi đưa vào SCRFD.

        Công việc:
            - resize ảnh
            - padding nếu cần
            - normalize pixel
            - chuyển sang tensor đúng shape

        Đây là preprocessing dành RIÊNG cho detector.
        """
        image = self._as_bgr_image(image)
        (
            target_height,
            target_width,
            _scale,
            resized_width,
            resized_height,
            pad_x,
            pad_y,
        ) = self._detector_geometry(image.shape)
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

        # SCRFD expects the detector's own mean/std and RGB channel order.
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 128.0,
            size=(target_width, target_height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        if blob.dtype != np.float32 or blob.ndim != 4:
            raise RuntimeError("Invalid SCRFD preprocessing output")
        return np.ascontiguousarray(blob)

    @staticmethod
    def _squeeze_model_output(output: Any) -> np.ndarray:
        array = np.asarray(output)
        if array.size == 0:
            raise RuntimeError("SCRFD returned an empty output")
        array = np.squeeze(array)
        if array.ndim == 0:
            raise RuntimeError("SCRFD returned a scalar output")
        if array.ndim == 2 and array.shape[0] in (1, 4, 10, 15) and array.shape[1] > array.shape[0]:
            array = array.T
        return np.asarray(array, dtype=np.float32)

    @staticmethod
    def _output_rows(output: np.ndarray, feature_sizes: tuple[int, ...]) -> tuple[np.ndarray, int]:
        array = FaceEncoder._squeeze_model_output(output)
        if array.ndim == 1:
            return array.reshape(-1, 1), int(array.size)
        if array.ndim != 2:
            raise RuntimeError(f"Unexpected SCRFD output shape: {array.shape}")
        if array.shape[1] in feature_sizes:
            return array, int(array.shape[0])
        if array.shape[0] in feature_sizes:
            return array.T, int(array.shape[1])
        return array, int(array.shape[0])

    @staticmethod
    def _sigmoid_if_needed(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("SCRFD returned non-finite confidence scores")
        if np.min(scores) < 0.0 or np.max(scores) > 1.0:
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))
        return scores

    @staticmethod
    def _infer_stride(num_anchors: int, input_height: int, input_width: int) -> int:
        candidates = (8, 16, 32, 64)
        for stride in candidates:
            locations = (input_height // stride) * (input_width // stride)
            if locations > 0 and locations * num_anchors == num_anchors * locations:
                # This branch only documents the relationship; the actual
                # selection is made from the output row count below.
                continue
        return 0

    def _decode_level(
        self,
        bbox_output: np.ndarray,
        score_output: np.ndarray,
        landmark_output: np.ndarray,
        input_height: int,
        input_width: int,
    ):
        bbox = np.asarray(bbox_output, dtype=np.float32)
        scores = self._sigmoid_if_needed(score_output)
        landmarks = np.asarray(landmark_output, dtype=np.float32)
        if bbox.ndim != 2 or bbox.shape[1] != 4:
            raise RuntimeError(f"Invalid SCRFD bbox output shape: {bbox.shape}")
        if landmarks.ndim != 2 or landmarks.shape[1] != 10:
            raise RuntimeError(f"Invalid SCRFD landmark output shape: {landmarks.shape}")
        if len(scores) != len(bbox) or len(landmarks) != len(bbox):
            raise RuntimeError("SCRFD output row counts do not match")
        if not np.all(np.isfinite(bbox)) or not np.all(np.isfinite(landmarks)):
            raise RuntimeError("SCRFD returned non-finite boxes or landmarks")

        row_count = len(bbox)
        stride = None
        anchors = None
        for candidate_stride in (8, 16, 32, 64):
            locations = (input_height // candidate_stride) * (input_width // candidate_stride)
            if locations > 0 and row_count % locations == 0:
                candidate_anchors = row_count // locations
                if 1 <= candidate_anchors <= 4:
                    stride = candidate_stride
                    anchors = candidate_anchors
                    break
        if stride is None or anchors is None:
            # Dynamic/variant exports can have a different anchor count. Pick
            # the closest standard stride and validate the resulting grid.
            possible = []
            for candidate_stride in (8, 16, 32, 64):
                locations = (input_height // candidate_stride) * (input_width // candidate_stride)
                if locations:
                    possible.append(
                        (
                            abs(row_count / locations - round(row_count / locations)),
                            candidate_stride,
                            max(1, int(round(row_count / locations))),
                        )
                    )
            if not possible:
                raise RuntimeError("Cannot infer SCRFD output stride")
            _error, stride, anchors = min(possible, key=lambda item: item[0])
            locations = (input_height // stride) * (input_width // stride)
            if locations * anchors != row_count:
                raise RuntimeError("Cannot align SCRFD outputs to an anchor grid")

        grid_height = input_height // stride
        grid_width = input_width // stride
        centers_y, centers_x = np.mgrid[0:grid_height, 0:grid_width]
        centers = np.stack(
            [centers_x.reshape(-1) * stride, centers_y.reshape(-1) * stride],
            axis=1,
        )
        centers = np.repeat(centers, anchors, axis=0).astype(np.float32)
        if len(centers) != row_count:
            raise RuntimeError("SCRFD anchor grid size does not match model output")

        distances = bbox * float(stride)
        decoded_boxes = np.column_stack(
            [
                centers[:, 0] - distances[:, 0],
                centers[:, 1] - distances[:, 1],
                centers[:, 0] + distances[:, 2],
                centers[:, 1] + distances[:, 3],
            ]
        )
        decoded_landmarks = landmarks.reshape(row_count, 5, 2) * float(stride)
        decoded_landmarks += centers[:, None, :]
        return decoded_boxes, scores, decoded_landmarks

    def _parse_detector_outputs(self, outputs, input_height: int, input_width: int):
        if not outputs:
            raise RuntimeError("SCRFD returned no outputs")

        combined = []
        bbox_outputs = []
        score_outputs = []
        landmark_outputs = []
        for raw_output in outputs:
            array = self._squeeze_model_output(raw_output)
            if array.ndim == 2 and array.shape[1] in (15, 16):
                combined.append(array)
                continue
            if array.ndim == 1:
                score_outputs.append(array.reshape(-1, 1))
                continue
            if array.ndim != 2:
                raise RuntimeError(f"Unexpected SCRFD output shape: {array.shape}")
            feature_count = array.shape[1]
            if feature_count == 4:
                bbox_outputs.append(array)
            elif feature_count == 10:
                landmark_outputs.append(array)
            elif feature_count == 1:
                score_outputs.append(array)
            elif array.shape[0] in (1, 4, 10, 15, 16):
                transposed = array.T
                if transposed.shape[1] == 4:
                    bbox_outputs.append(transposed)
                elif transposed.shape[1] == 10:
                    landmark_outputs.append(transposed)
                elif transposed.shape[1] == 1:
                    score_outputs.append(transposed)
                else:
                    raise RuntimeError(f"Unsupported SCRFD output shape: {array.shape}")
            else:
                raise RuntimeError(f"Unsupported SCRFD output shape: {array.shape}")

        decoded_boxes = []
        decoded_scores = []
        decoded_landmarks = []
        if combined:
            for output in combined:
                if output.shape[1] < 15:
                    raise RuntimeError("Combined SCRFD output must contain boxes, scores, and landmarks")
                boxes, scores, landmarks = self._decode_level(
                    output[:, :4],
                    output[:, 4:5],
                    output[:, 5:15],
                    input_height,
                    input_width,
                )
                decoded_boxes.append(boxes)
                decoded_scores.append(scores)
                decoded_landmarks.append(landmarks)
        else:
            if not bbox_outputs or not score_outputs or not landmark_outputs:
                raise RuntimeError(
                    "SCRFD output must provide bbox, confidence, and five-landmark tensors"
                )
            used_scores = set()
            used_landmarks = set()
            for bbox_output in bbox_outputs:
                row_count = len(bbox_output)
                score_index = next(
                    (
                        index
                        for index, candidate in enumerate(score_outputs)
                        if index not in used_scores and len(candidate) == row_count
                    ),
                    None,
                )
                landmark_index = next(
                    (
                        index
                        for index, candidate in enumerate(landmark_outputs)
                        if index not in used_landmarks and len(candidate) == row_count
                    ),
                    None,
                )
                if score_index is None or landmark_index is None:
                    raise RuntimeError("SCRFD output levels have incompatible row counts")
                used_scores.add(score_index)
                used_landmarks.add(landmark_index)
                boxes, scores, landmarks = self._decode_level(
                    bbox_output,
                    score_outputs[score_index],
                    landmark_outputs[landmark_index],
                    input_height,
                    input_width,
                )
                decoded_boxes.append(boxes)
                decoded_scores.append(scores)
                decoded_landmarks.append(landmarks)

        if not decoded_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
            )
        return (
            np.concatenate(decoded_boxes, axis=0),
            np.concatenate(decoded_scores, axis=0),
            np.concatenate(decoded_landmarks, axis=0),
        )

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.4):
        if len(boxes) == 0:
            return []
        order = np.argsort(-scores)
        kept = []
        while len(order):
            current = int(order[0])
            kept.append(current)
            if len(order) == 1:
                break
            remaining = order[1:]
            xx1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
            yy1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
            xx2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
            yy2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
            intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_current = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(
                0.0, boxes[current, 3] - boxes[current, 1]
            )
            remaining_boxes = boxes[remaining]
            area_remaining = np.maximum(0.0, remaining_boxes[:, 2] - remaining_boxes[:, 0]) * np.maximum(
                0.0, remaining_boxes[:, 3] - remaining_boxes[:, 1]
            )
            union = area_current + area_remaining - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = remaining[iou <= threshold]
        return kept

    def detect_faces(self, image: np.ndarray):
        """
        Detect tất cả khuôn mặt trong ảnh.

        Input:
            image H x W x 3

        Output:
            list các face:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": confidence,
                    "landmarks": 5 điểm
                },
                ...
            ]

        SCRFD làm bước này.
        """
        image = self._as_bgr_image(image)
        blob = self._preprocess_detector(image)
        input_height, input_width = int(blob.shape[2]), int(blob.shape[3])
        try:
            outputs = self.detector_session.run(None, {self._detector_input_name: blob})
        except Exception as exc:
            raise RuntimeError(f"SCRFD inference failed: {exc}") from exc

        boxes, scores, landmarks = self._parse_detector_outputs(
            outputs, input_height, input_width
        )
        (
            _target_height,
            _target_width,
            scale,
            _resized_width,
            _resized_height,
            pad_x,
            pad_y,
        ) = self._detector_geometry(image.shape)
        boxes = np.asarray(boxes, dtype=np.float32)
        landmarks = np.asarray(landmarks, dtype=np.float32)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - float(pad_x)) / float(scale)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - float(pad_y)) / float(scale)
        landmarks[:, :, 0] = (landmarks[:, :, 0] - float(pad_x)) / float(scale)
        landmarks[:, :, 1] = (landmarks[:, :, 1] - float(pad_y)) / float(scale)

        height, width = image.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, width - 1)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, height - 1)

        valid = []
        for index, (box, score, points) in enumerate(zip(boxes, scores, landmarks)):
            x1, y1, x2, y2 = [float(value) for value in box]
            if not np.isfinite(score) or not np.all(np.isfinite(points)):
                continue
            if float(score) < self.det_threshold:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            if min(x2 - x1, y2 - y1) < self.min_face_size:
                continue
            if points.shape != (5, 2):
                raise RuntimeError("SCRFD landmarks must have shape (5, 2)")
            valid.append(index)

        if not valid:
            return []
        valid_boxes = boxes[valid]
        valid_scores = scores[valid]
        keep_relative = self._nms(valid_boxes, valid_scores)
        keep = [valid[index] for index in keep_relative]
        keep.sort(key=lambda index: float(scores[index]), reverse=True)
        faces = []
        for index in keep:
            faces.append(
                {
                    "bbox": [float(value) for value in boxes[index]],
                    "score": float(scores[index]),
                    "landmarks": np.asarray(landmarks[index], dtype=np.float32).reshape(5, 2),
                }
            )
        return faces

    def align_face(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Dùng 5 facial landmarks:

            mắt trái
            mắt phải
            mũi
            khóe miệng trái
            khóe miệng phải

        để rotate + scale + translate khuôn mặt
        về template chuẩn.

        Output:
            aligned face 112 x 112
        """
        image = self._as_bgr_image(image)
        points = np.asarray(landmarks, dtype=np.float32)
        if points.shape != (5, 2) or not np.all(np.isfinite(points)):
            raise ValueError("landmarks must have shape (5, 2) and finite values")
        transform, _inliers = cv2.estimateAffinePartial2D(
            points,
            self._ARC_FACE_TEMPLATE,
            method=cv2.LMEDS,
        )
        if transform is None or transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
            raise RuntimeError("Unable to estimate face alignment transform")
        aligned = cv2.warpAffine(
            image,
            transform,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        if aligned.shape != (112, 112, 3):
            raise RuntimeError(f"Aligned face has invalid shape: {aligned.shape}")
        return np.ascontiguousarray(aligned)

    def _preprocess_arcface(
        self,
        aligned_face: np.ndarray
    ) -> np.ndarray:
        """
        Tiền xử lý face 112x112 trước ArcFace.

        Thường gồm:
            - convert color format nếu cần
            - float32
            - normalize pixel
            - HWC -> CHW
            - thêm batch dimension

        Output thường:
            1 x 3 x 112 x 112
        """
        face = self._as_bgr_image(aligned_face)
        if face.shape[:2] != (112, 112):
            raise ValueError("aligned_face must have shape 112 x 112")
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalized = (rgb - 127.5) / 127.5
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        if tensor.shape != (1, 3, 112, 112):
            raise RuntimeError(f"Invalid ArcFace preprocessing output: {tensor.shape}")
        return tensor

    def embed_face(
        self,
        aligned_face: np.ndarray
    ) -> np.ndarray:
        """
        Đưa face đã alignment vào ArcFace.

        Output:
            embedding vector shape (512,)

        Sau đó thực hiện L2 normalization.
        """
        tensor = self._preprocess_arcface(aligned_face)
        try:
            outputs = self.embedder_session.run(None, {self._embedder_input_name: tensor})
        except Exception as exc:
            raise RuntimeError(f"ArcFace inference failed: {exc}") from exc
        if not outputs:
            raise RuntimeError("ArcFace returned no output")
        embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if embedding.shape != (512,):
            raise RuntimeError(f"ArcFace embedding must have shape (512,), got {embedding.shape}")
        if not np.all(np.isfinite(embedding)):
            raise RuntimeError("ArcFace returned a non-finite embedding")
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-12:
            raise RuntimeError("ArcFace returned a zero-norm embedding")
        embedding = embedding / norm
        if not np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5):
            raise RuntimeError("ArcFace embedding normalization failed")
        return np.ascontiguousarray(embedding, dtype=np.float32)

    def encode_image(
        self,
        image: np.ndarray
    ):
        """
        Xử lý một ảnh event hoàn chỉnh.

        Pipeline:
            image
            -> detect_faces()
            -> align_face()
            -> embed_face()

        Output:
            [
                {
                    "bbox": ...,
                    "score": ...,
                    "embedding": vector 512-D
                },
                ...
            ]

        Một ảnh có N người thì trả N embeddings.
        """
        image = self._as_bgr_image(image)
        encoded = []
        for face in self.detect_faces(image):
            aligned = self.align_face(image, face["landmarks"])
            encoded.append(
                {
                    "bbox": list(face["bbox"]),
                    "score": float(face["score"]),
                    "embedding": self.embed_face(aligned),
                }
            )
        return encoded

    def encode_query(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Dùng cho selfie query.

        Pipeline:
            selfie
            -> detect face
            -> chọn face chính
            -> align
            -> embedding

        Nếu ảnh có nhiều người:
            ưu tiên face lớn nhất hoặc gần center nhất.

        Output:
            query embedding 512-D
        """
        image = self._as_bgr_image(image)
        faces = self.detect_faces(image)
        if not faces:
            raise ValueError("No face detected in query image")

        def area(face):
            x1, y1, x2, y2 = face["bbox"]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        selected = max(faces, key=area)
        aligned = self.align_face(image, selected["landmarks"])
        embedding = self.embed_face(aligned)
        if embedding.shape != (512,):
            raise RuntimeError("Query embedding must have shape (512,)")
        return embedding
