# src/face_encoder.py

import numpy as np


class FaceEncoder:

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
        pass


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
        pass


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
        pass


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
        pass


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
        pass


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
        pass


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
        pass


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
        pass

# src/vector_store.py

import numpy as np


class VectorStore:

    def __init__(
        self,
        dimension: int = 512
    ):
        """
        Khởi tạo FAISS IndexFlatIP.

        Vì embedding đã L2-normalize:

            inner product
            =
            cosine similarity
        """
        pass


    def add_face(
        self,
        embedding: np.ndarray,
        photo_path: str,
        bbox
    ):
        """
        Add một embedding vào FAISS.

        Đồng thời lưu metadata:

            vector ID
            -> photo_path
            -> bbox

        Ví dụ:

            FAISS ID 51
            -> IMG_0012.jpg
            -> [x1,y1,x2,y2]
        """
        pass


    def add_faces(
        self,
        embeddings,
        metadata
    ):
        """
        Add nhiều face cùng lúc.

        Dùng khi build index cho cả event.

        Nhanh hơn gọi add_face() hàng nghìn lần.
        """
        pass


    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100
    ):
        """
        Search query embedding trong FAISS.

        FAISS tính similarity giữa query
        với embeddings trong index.

        Output ví dụ:

        [
            {
                "id": 124,
                "score": 0.91,
                "photo_path": "...",
                "bbox": [...]
            },
            ...
        ]
        """
        pass


    def save(
        self,
        index_path: str,
        metadata_path: str
    ):
        """
        Save:

            FAISS index
                -> faces.faiss

            metadata
                -> metadata.json
        """
        pass


    def load(
        self,
        index_path: str,
        metadata_path: str
    ):
        """
        Load FAISS index và metadata
        đã build trước đó.
        """
        pass


    def __len__(self):
        """
        Trả về số face embeddings
        hiện đang nằm trong index.
        """
        pass
# src/search_engine.py


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
        self.threshold = similarity_threshold
        self.top_k = top_k


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
        pass


    def deduplicate_photos(
        self,
        results
    ):
        """
        Đảm bảo một photo chỉ xuất hiện một lần.

        Ví dụ nếu có duplicate photo path,
        giữ result có similarity cao nhất.
        """
        pass


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
            -> VectorStore.search()
            -> threshold filter
            -> deduplicate
            -> matching photos

        Output:
            danh sách ảnh match.
        """
        pass
