"""Query-time retrieval and result shaping."""

from __future__ import annotations


class SearchEngine:

    def __init__(
        self,
        face_encoder,
        vector_store,
        similarity_threshold=0.5,
        top_k=100,
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
        query_image
    ):
        """
        API chính của search system.

        Pipeline:

            query image
            -> FaceEncoder.encode_query()
            -> query embedding
            -> FAISS range search at the configured threshold
            -> threshold filter
            -> deduplicate
            -> matching photos

        Output:
            danh sách ảnh match.
        """
        query_embedding = self.encoder.encode_query(query_image)
        # top_k remains a compatibility/display hint, but production
        # completeness is defined by the threshold: do not truncate valid
        # matches before filtering and photo-path deduplication.
        results = self.store.search(
            query_embedding,
            score_threshold=self.threshold,
        )
        filtered = self.filter_by_threshold(results)
        return self.deduplicate_photos(filtered)
