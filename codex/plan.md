Face Search Project — End-to-End Execution Plan
0. Goal
Hoàn thiện dự án face-search/ từ specification đến một ứng dụng chạy được:
Event photos
    -> detect faces
    -> align faces
    -> ArcFace embeddings
    -> FAISS index

User selfie
    -> detect face
    -> align
    -> ArcFace query embedding
    -> vector retrieval
    -> threshold filtering
    -> matching event photos
Mục tiêu của tuning là tối ưu độ chính xác và trải nghiệm người dùng, không phải làm benchmark nghiên cứu.
Nguyên tắc UX:
- Hạn chế ảnh sai người xuất hiện trong kết quả.
- Đồng thời cố gắng tìm được càng nhiều ảnh thật của user càng tốt.
- Ưu tiên cấu hình có Recall cao nhưng phải giữ Precision ở mức chấp nhận được.
- Đo cả tốc độ indexing và query để tránh đổi accuracy lấy latency quá lớn.
1. Documentation reading order
Agent PHẢI đọc toàn bộ specification trước khi code.
Đọc theo đúng thứ tự:
1. structure.md
2. src.md
3. scripts.md
4. app.md
5. plan.md
1.1 structure.md
Source of truth cho:
- folder structure;
- tên file;
- model được sử dụng;
- data flow;
- storage;
- kiến trúc tổng thể.
Không tự ý thêm layer/module/database/framework nếu structure.md không yêu cầu.
1.2 src.md
Source of truth cho:
src/
├── face_encoder.py
├── vector_store.py
└── search_engine.py
Agent phải implement đúng:
- function/class names;
- parameters;
- return values;
- behavior;
- edge cases.
1.3 scripts.md
Source of truth cho:
scripts/
├── build_index.py
├── search.py
└── tune.py
1.4 app.md
Source of truth cho behavior của user interface và API.
1.5 Conflict resolution
Nếu docs mâu thuẫn:
architecture/folder ownership
    -> structure.md

function behavior inside src/
    -> src.md

CLI/script behavior
    -> scripts.md

app/UI behavior
    -> app.md

execution order
    -> plan.md
Không silently invent behavior mới.
2. Required project structure
Kiểm tra project có structure tối thiểu:
face-search/
│
├── data/
│   ├── event_photos/
│   ├── queries/
│   └── validation/
│
├── models/
│   ├── scrfd_10g.onnx
│   └── arcface_w600k_r50.onnx
│
├── storage/
│   ├── faces.faiss
│   ├── metadata.json
│   └── best_config.json
│
├── src/
│   ├── face_encoder.py
│   ├── vector_store.py
│   └── search_engine.py
│
├── scripts/
│   ├── build_index.py
│   ├── search.py
│   └── tune.py
│
├── app.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
Không tạo các module như:
database.py
alignment.py
preprocessing.py
config.py
service/
repository/
trừ khi specification hiện tại yêu cầu.
Giữ MVP lean.
3. Environment setup
3.1 Python environment
Tạo virtual environment:
python -m venv .venv
Cài tối thiểu:
numpy
opencv-python
onnxruntime
faiss-cpu
huggingface_hub
datasets
python-dotenv
tqdm
pandas
scikit-learn
fastapi
uvicorn
python-multipart
jinja2
Nếu máy có NVIDIA GPU và ONNX Runtime GPU hoạt động ổn định thì có thể dùng:
onnxruntime-gpu
nhưng CPU phải vẫn chạy được.
3.2 .env
Expected:
HF_TOKEN=...
Load bằng python-dotenv.
Rules:
- không print token;
- không log token;
- không ghi token vào source;
- không ghi token vào report.md;
- .env bắt buộc nằm trong .gitignore.
Nếu HF_TOKEN không tồn tại, stop với lỗi rõ ràng.
4. Download pretrained models
Dùng huggingface_hub + HF_TOKEN.
Repository:
globalnebula/insightface-buffalo-l-onnx
Tải đúng hai file:
scrfd_10g.onnx
arcface_w600k_r50.onnx
Lưu:
face-search/models/scrfd_10g.onnx
face-search/models/arcface_w600k_r50.onnx
Expected roles:
SCRFD-10G
    input  : full image
    output : face bbox + confidence + 5 landmarks

ArcFace w600k_r50
    input  : aligned 112x112 face
    output : 512-D face embedding
Sau download verify:
- file exists;
- file size > 0;
- ONNX Runtime load được;
- output shapes đúng.
Không train lại model.
5. Implement core CV pipeline
Implement theo src.md.
5.1 src/face_encoder.py
Required flow:
raw image
    ↓
SCRFD preprocessing
    ↓
SCRFD inference
    ↓
bbox + confidence + 5 landmarks
    ↓
face alignment
    ↓
112x112 face
    ↓
ArcFace preprocessing
    ↓
ArcFace inference
    ↓
512-D embedding
    ↓
L2 normalization
Required sanity checks:
landmarks shape      == (5, 2)
aligned face         == 112x112
embedding shape      == (512,)
L2 norm              ~= 1.0
Important:
- detector preprocessing và ArcFace preprocessing là hai preprocessing khác nhau;
- không tách thêm module chỉ để chứa preprocessing;
- alignment nằm trong FaceEncoder theo architecture hiện tại.
5.2 Error behavior
Handle:
- image không đọc được;
- không detect được face;
- nhiều face trong query;
- corrupted file;
- invalid ONNX output.
Query selfie có nhiều face:
chọn largest valid face
trừ khi src.md quy định khác.
6. Implement vector retrieval
6.1 src/vector_store.py
Baseline:
FAISS IndexFlatIP
dimension = 512
Vì embeddings L2-normalized:
inner product == cosine similarity
Vector store lưu:
FAISS ID
    -> embedding

metadata.json:
FAISS ID
    -> photo path
    -> bbox
Không cluster identities trước.
Không dùng:
IVF
PQ
HNSW
ở MVP.
6.2 Persistence
Verify:
save()
restart process
load()
same index count
same search behavior
FAISS count và metadata count phải khớp.
7. Implement search engine
7.1 src/search_engine.py
Required query flow:
query image
    ↓
FaceEncoder.encode_query()
    ↓
query embedding
    ↓
VectorStore search
    ↓
similarity filtering
    ↓
deduplicate photo paths
    ↓
sort score descending
    ↓
matching event photos
CRITICAL:
query embedding must NOT be added to FAISS
Event-photo embeddings mới được persist.
8. Validation strategy — product-oriented
Không dùng CelebA làm dataset validation chính.
Lý do:
- phần lớn ảnh face-centric;
- không mô phỏng tốt group/event images;
- không đủ sát failure mode "mặt nhỏ giữa nhiều người".
Không dùng UCCS cho automated setup:
- đúng use case;
- nhưng official dataset bị restricted license;
- bản đầy đủ không phù hợp cho một agent tự động tải và setup.
Không dùng PIPA làm automated primary dataset:
- rất phù hợp group-photo + identity;
- nhưng original image distribution không còn ổn định;
- hiện metadata còn, nhưng ảnh phải crawl từ Flickr và một số ảnh đã mất.
Thay vào đó dùng một validation suite gồm 2 dataset bổ sung nhau.
9. Validation Dataset A — WIDER FACE validation
Purpose:
optimize full-scene face detection
Dataset:
CUHK-CSE/wider_face
validation split
Properties:
- full-scene images;
- nhiều ảnh có nhiều faces;
- scale variation lớn;
- small faces;
- pose variation;
- occlusion;
- blur/illumination variation;
- bounding-box ground truth.
Validation split:
3,226 images
~363 MB image archive
Download bằng Hugging Face với HF_TOKEN.
Store:
data/validation/wider_face/
Expected:
data/validation/wider_face/
├── images/
└── annotations/
Không copy dataset này vào queries/.
Không dùng identity matching trên WIDER FACE vì dataset không cung cấp repeated person identity suitable cho retrieval.
10. Validation Dataset B — VGGFace2 test subset
Purpose:
optimize ArcFace similarity threshold
optimize retrieval behavior
Repository:
ProgramComputer/VGGFace2
Không tải full 40 GB.
Use test split / streaming loader.
Target subset:
~100 identities
~15–20 images per identity
~1,500–2,000 face images total
Selection rules:
- identity phải có ít nhất 15 usable images;
- deterministic random seed;
- lưu identity mapping;
- không thay subset giữa các grid-search runs.
Store:
data/validation/vggface2_subset/
├── images/
├── metadata.csv
└── split.json
metadata.csv tối thiểu:
file
identity
split.json ghi:
query image(s) per identity
gallery images per identity
random seed
11. Why two datasets are necessary
Pipeline có hai failure modes độc lập:
A. detector không nhìn thấy face
B. detector thấy face nhưng embedding matching sai
WIDER FACE giải quyết A.
VGGFace2 giải quyết B.
Không ép một dataset không phù hợp phải đánh giá cả hai.
Final config là combination:
best detector config from WIDER FACE
+
best similarity config from VGGFace2
Sau đó test end-to-end bằng smoke event.
12. Build a local smoke event
Tạo fixture từ VGGFace2 subset để verify toàn bộ app flow.
Chọn:
10–20 identities
Per identity:
1 clear image
    -> query

5–10 other images
    -> event gallery
Store:
data/queries/smoke_queries/
data/event_photos/smoke_event/
Ý nghĩa:
event_photos/
    = ảnh thuộc event gallery
    = được detect/embed/index vào FAISS

queries/
    = ảnh selfie/query
    = được embed rồi dùng search
    = KHÔNG được add vào FAISS

validation/
    = dataset dùng để tune parameter
    = không phải production gallery
Smoke event này dùng để verify software flow, không dùng thay thế WIDER detector validation.
13. Implement and test scripts/build_index.py
Run on:
data/event_photos/smoke_event/
Required flow:
for each image
    -> detect all faces
    -> align
    -> embed each face
    -> add each embedding to FAISS
    -> save metadata
After indexing print:
number of images processed
number of images failed
number of images with no faces
number of faces indexed
indexing time
Save:
storage/faces.faiss
storage/metadata.json
Verify:
FAISS vector count == metadata count
14. Implement and run scripts/search.py
For each query in:
data/queries/smoke_queries/
Run:
query
    -> FaceEncoder
    -> FAISS
    -> filter
    -> ranked photo paths
Output should include:
rank
similarity score
photo path
Smoke success criteria:
- same-identity images rank near the top;
- unrelated identities should not dominate results;
- no crash on no-face query;
- no duplicate photo path in result list.
Do this BEFORE grid search to establish a baseline.
Store baseline metrics/results for report.md.
15. Grid search Stage A — detector optimization
Dataset:
WIDER FACE validation
Tune:
det_size:
    640
    960
    1280

det_threshold:
    0.30
    0.40
    0.50
    0.60
    0.70
Optional second pass:
min_face_size:
    20
    30
    40
Do not immediately grid all combinations if runtime is excessive.
Recommended strategy:
Pass 1:
    det_size × det_threshold

Pass 2:
    optimize min_face_size around best Pass-1 config
Metrics:
IoU >= 0.5 detection match

Detection Recall
Precision
F1
False Positives / Image
Latency / Image
Product selection objective:
maximize Detection Recall
while keeping false positives and latency reasonable
Because a face missed by detector can never be recovered by ArcFace.
Persist all results:
storage/tuning_detector.csv
16. Freeze detector config
After Stage A:
best_det_size
best_det_threshold
best_min_face_size
Freeze these values before recognition threshold tuning.
Do not rerun detector grid while testing every ArcFace threshold.
17. Cache recognition embeddings
Dataset:
VGGFace2 subset
For each image:
detect
-> align
-> ArcFace
-> L2 normalize
Run once.
Cache:
storage/validation_embeddings.npz
Include:
embedding
identity
image path
This makes threshold grid search fast.
18. Grid search Stage B — retrieval optimization
Use cached embeddings.
For each identity:
query image
    vs
all gallery embeddings
Sweep:
similarity_threshold:
    0.20 to 0.80
    step 0.01
Also evaluate if specified in scripts.md:
top_k
query strategy
Metrics:
Precision
Recall
F1
F2
Recall@K
Primary UX objective:
maximize Recall
subject to Precision >= 95%
Reason:
- user wants as many of their real event photos as possible;
- but result gallery should not be polluted by many wrong people.
Fallback if no config achieves Precision >= 95%:
choose highest F2
Persist:
storage/tuning_retrieval.csv
19. Create best_config.json
After tuning:
storage/best_config.json
Example schema:
{
  "det_size": 960,
  "det_threshold": 0.4,
  "min_face_size": 20,
  "similarity_threshold": 0.XX,
  "embedding_dim": 512,
  "detector_model": "scrfd_10g.onnx",
  "embedding_model": "arcface_w600k_r50.onnx"
}
IMPORTANT:
0.XX must come from grid search.
Do not copy a threshold from this document.
20. Rebuild and rerun smoke test
Delete/rebuild smoke index using best_config.json.
Run:
python scripts/build_index.py ...
python scripts/search.py ...
Compare with baseline:
before tuning
vs
after tuning
Record:
Precision
Recall
wrong-person results
query latency
indexing latency
If tuned config is unexpectedly worse on smoke data:
- inspect failure cases;
- verify coordinate scaling;
- verify alignment;
- verify L2 normalization;
- verify similarity direction;
- verify metadata mapping;
- verify threshold logic.
Do not proceed to UI until core pipeline is stable.
21. Generate final report
Create outside project:
report.md
face-search/
NOT:
face-search/report.md
Required sections:
# Project Summary

# Final Architecture

# Environment

# Models
- source
- filenames
- sizes
- ONNX providers

# Validation Data
- WIDER FACE validation
- VGGFace2 subset
- subset construction

# Baseline

# Detector Grid Search
- search space
- metrics
- best detector config

# Retrieval Grid Search
- search space
- metrics
- best threshold

# Final Configuration

# Smoke Event Test
- baseline vs tuned

# Performance
- indexing time
- query latency
- storage size

# Failure Cases
- small faces
- extreme pose
- heavy occlusion
- severe blur
- false matches

# Product Notes
- user experience implications
- recommended selfie guidance

# How To Run
This is a project report, not a research paper.
Do not include:
novelty claims
paper-style contribution claims
acceptance claims
research limitations wording
HF_TOKEN
Use terminology like:
product constraint
known failure case
deployment note
UX trade-off
22. Setup app.py after tuning is complete
Do not finalize UI before best_config.json exists.
App must load:
models/
storage/best_config.json
storage/faces.faiss
storage/metadata.json
Use final tuned parameters automatically.
23. User Interface — two main flows
23.1 Flow A — Add Event Photos
UI:
[ Add Event Photos ]
Allow one or multiple images.
Behavior:
upload event image
    ↓
save original file
    ↓
FaceEncoder.encode_image()
    ↓
0..N face embeddings
    ↓
VectorStore.add_face()
    ↓
persist FAISS + metadata
Return:
photos uploaded
faces detected
faces indexed
failed photos
Newly uploaded event photos must become searchable without manual rebuild.
24. User Interface — Find My Photos
UI:
[ Find My Photos ]
User uploads one selfie.
Behavior:
selfie
    ↓
FaceEncoder.encode_query()
    ↓
query embedding
    ↓
VectorStore retrieval
    ↓
best similarity threshold
    ↓
deduplicate results
    ↓
photo gallery
Query image:
must NOT be added to event FAISS index
UI should show:
matching photo thumbnails
similarity score optional/debug only
number of matches
For normal users, similarity score does not need to be emphasized.
25. UX error handling
App must handle:
No face in selfie
Show:
Không tìm thấy khuôn mặt rõ trong ảnh.
Hãy thử một ảnh selfie chính diện và đủ sáng hơn.
Multiple faces in selfie
Use largest face and optionally show:
Hệ thống sử dụng khuôn mặt lớn nhất trong ảnh.
No matches
Show:
Chưa tìm thấy ảnh phù hợp trong event này.
Hãy thử selfie khác với góc mặt hoặc ánh sáng khác.
Event photo has no face
Do not fail upload.
Store the photo if desired, but report:
0 searchable faces
26. App implementation scope
Keep MVP simple.
Use:
FastAPI
Jinja2 / simple HTML
minimal CSS
Do not add:
React
Next.js
Redis
Celery
PostgreSQL
Docker orchestration
microservices
unless current specs explicitly require them.
27. Final integration tests
Agent must verify all of these:
[ ] model download using HF_TOKEN
[ ] SCRFD loads
[ ] ArcFace loads
[ ] image -> face detection
[ ] landmarks -> alignment
[ ] face -> 512-D embedding
[ ] embedding L2 norm ~= 1
[ ] add event photo
[ ] add group photo with multiple faces
[ ] add photo with zero faces
[ ] restart app and reload index
[ ] query known identity
[ ] query unrelated identity
[ ] query image with no face
[ ] query image with multiple faces
[ ] duplicate event upload behavior
[ ] newly uploaded photo becomes searchable
[ ] FAISS count == metadata count
[ ] grid search completed
[ ] best_config.json exists
[ ] app uses best_config.json
[ ] report.md generated outside face-search/
28. Definition of Done
Project is complete only when:
1. All specification files were followed.
2. Core source files are implemented.
3. Required ONNX models are downloaded.
4. WIDER FACE validation data is available.
5. VGGFace2 identity subset is available.
6. Baseline smoke search runs.
7. Detector grid search completes.
8. Recognition/retrieval threshold grid search completes.
9. best_config.json is generated from real results.
10. Smoke index is rebuilt with tuned config.
11. search.py passes end-to-end.
12. report.md is generated outside face-search/.
13. app.py provides:
      - Add Event Photos
      - Find My Photos
14. New event uploads are automatically embedded/indexed.
15. Query selfies are embedded only for search, not persisted in the event index.
16. Saved index survives application restart.
17. No secret/token is committed.
29. Execution order summary
Agent must execute in this order:
READ SPECS
    ↓
SETUP ENVIRONMENT
    ↓
DOWNLOAD SCRFD + ARCFACE
    ↓
IMPLEMENT FaceEncoder
    ↓
SANITY TEST CV OUTPUTS
    ↓
IMPLEMENT VectorStore
    ↓
IMPLEMENT SearchEngine
    ↓
DOWNLOAD WIDER FACE VALIDATION
    ↓
DOWNLOAD VGGFACE2 SUBSET
    ↓
CREATE SMOKE EVENT + QUERIES
    ↓
IMPLEMENT/RUN build_index.py
    ↓
IMPLEMENT/RUN search.py BASELINE
    ↓
RUN DETECTOR GRID SEARCH
    ↓
FREEZE DETECTOR
    ↓
CACHE ARCFACE EMBEDDINGS
    ↓
RUN RETRIEVAL GRID SEARCH
    ↓
SAVE best_config.json
    ↓
REBUILD SMOKE INDEX
    ↓
RERUN search.py
    ↓
WRITE ../report.md
    ↓
FINALIZE app.py
    ↓
RUN INTEGRATION TESTS
    ↓
DONE
30. Data-source notes
Models
Hugging Face:
https://huggingface.co/globalnebula/insightface-buffalo-l-onnx
Files:
scrfd_10g.onnx
arcface_w600k_r50.onnx
WIDER FACE
Official/Hugging Face:
https://huggingface.co/datasets/CUHK-CSE/wider_face
Use validation split.
VGGFace2
Hugging Face:
https://huggingface.co/datasets/ProgramComputer/VGGFace2
Use a deterministic subset of the test identities.
31. Final product architecture
                    EVENT INGESTION

Event Photos
     ↓
SCRFD
     ↓
Faces + Landmarks
     ↓
Alignment
     ↓
ArcFace
     ↓
512-D normalized embeddings
     ↓
FAISS + metadata
     ↓
Persistent event index


                    USER SEARCH

Selfie
     ↓
SCRFD
     ↓
Largest face
     ↓
Alignment
     ↓
ArcFace
     ↓
512-D query embedding
     ↓
FAISS similarity search
     ↓
best tuned threshold
     ↓
deduplicate photo paths
     ↓
Matching event photos
Core distinction:
Event photo:
    embed + ADD

User selfie:
    embed + SEARCH
The query embedding is temporary and must not become part of the event gallery.