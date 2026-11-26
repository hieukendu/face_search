"""Query-time retrieval and result shaping."""

from __future__ import annotations

import numpy as np


class SearchEngine:

    def __init__(
        self,
        face_encoder,
        vector_store,
        similarity_threshold=0.5,
        top_k=100,
        raw_similarity_threshold=None,
        confidence_alpha=0.0,
    ):
        """
        Nhận vào:

            FaceEncoder
            VectorStore

        và các parameter search.
        """
        self.encoder = face_encoder
        self.store = vector_store
        self.threshold = float(similarity_threshold)
        self.raw_threshold = (
            float(raw_similarity_threshold)
            if raw_similarity_threshold is not None
            else self.threshold
        )
        self.confidence_alpha = float(confidence_alpha)
        self.confidence_enabled = raw_similarity_threshold is not None
        if self.confidence_alpha < 0.0:
            raise ValueError("confidence_alpha must be non-negative")
        self.top_k = int(top_k)
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")

    def filter_by_threshold(
        self,
        results
    ):
        """
        Chỉ giữ candidate có:

            similarity >= threshold

        Ví dụ:

            .91 -> giữ
            .82 -> giữ
            .32 -> bỏ
        """
        if results is None:
            return []
        return [
            result
            for result in results
            if float(result.get("score", float("-inf"))) >= self.threshold
        ]

    def deduplicate_photos(
        self,
        results
    ):
        """
        Đảm bảo một photo chỉ xuất hiện một lần.

        Ví dụ nếu có duplicate photo path,
        giữ result có similarity cao nhất.
        """
        best_by_photo = {}
        for result in results or []:
            photo_path = result.get("photo_path")
            if photo_path is None:
                continue
            score = float(result.get("score", float("-inf")))
            previous = best_by_photo.get(photo_path)
            if previous is None or score > float(previous.get("score", float("-inf"))):
                best_by_photo[photo_path] = result
        return sorted(
            best_by_photo.values(),
            key=lambda result: float(result.get("score", float("-inf"))),
            reverse=True,
        )

    def search(
        self,
        query_image,
        *,
        is_face_crop=False,
    ):
        """
        API chính của search system.

        Pipeline:

            query image
            -> FaceEncoder.encode_query_with_scores()
            -> query embeddings for every detected face
            -> FAISS range search at the raw cosine threshold
            -> optional detector-confidence conjunction
            -> threshold filter
            -> deduplicate
            -> matching photos

        Output:
            danh sách ảnh match.
        """
        encode_with_scores = getattr(self.encoder, "encode_query_with_scores", None)
        if encode_with_scores is None:
            query_embeddings = (
                self.encoder.encode_query(query_image, is_face_crop=True)
                if is_face_crop else self.encoder.encode_query(query_image)
            )
            query_embeddings = np.atleast_2d(query_embeddings)
            query_scores = np.ones(len(query_embeddings), dtype=np.float32)
        else:
            query_embeddings, query_scores = encode_with_scores(
                query_image,
                is_face_crop=is_face_crop,
            )
            query_embeddings = np.atleast_2d(query_embeddings)
            query_scores = np.asarray(query_scores, dtype=np.float32).reshape(-1)
            if len(query_embeddings) != len(query_scores):
                raise RuntimeError("Query embeddings and detector scores must have equal length")
        # top_k remains a compatibility/display hint, but production
        # completeness is defined by the threshold: do not truncate valid
        # matches before filtering and photo-path deduplication.
        results = []
        for embedding, query_score in zip(query_embeddings, query_scores):
            threshold = self.raw_threshold if self.confidence_enabled else self.threshold
            candidates = self.store.search(embedding, score_threshold=threshold)
            if not self.confidence_enabled:
                results.extend(candidates)
                continue
            query_score = float(np.clip(query_score, 1e-6, 1.0))
            for candidate in candidates:
                raw_score = float(candidate.get("score", float("-inf")))
                gallery_score = float(candidate.get("detector_score", 1.0))
                gallery_score = float(np.clip(gallery_score, 1e-6, 1.0))
                weighted_score = raw_score * (
                    query_score * gallery_score
                ) ** self.confidence_alpha
                if weighted_score >= self.threshold:
                    candidate = dict(candidate)
                    candidate["raw_score"] = raw_score
                    candidate["weighted_score"] = weighted_score
                    # Rank and display by the score that admitted the result.
                    candidate["score"] = weighted_score
                    results.append(candidate)
        filtered = self.filter_by_threshold(results)
        return self.deduplicate_photos(filtered)
