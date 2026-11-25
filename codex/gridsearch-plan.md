# Plan setup dataset và grid search cho face-search

## Material Passport

- Ngày: 2026-11-25.
- Workflow: academic-research-suite / experiment-agent, mode plan.
- Trạng thái: PLAN_ONLY; chưa tải dataset, chưa chạy grid search, chưa xác định cấu hình thắng.
- Mục tiêu: tìm cấu hình tốt nhất trong dải được thử cho tác vụ một ảnh query tìm các ảnh chứa cùng người trong gallery nhiều mặt.
- Working directory: `D:\face_search`.

## 1. Phạm vi và tiêu chí đầu ra

Chạy trực tiếp trên ảnh local:

```text
Dataset ảnh + ground truth
  -> SCRFD detect
  -> align 5 landmarks
  -> ArcFace 512-D, L2 normalize
  -> gallery: lưu mọi face hợp lệ vào FAISS tạm
  -> query: chọn face lớn nhất như encode_query() hiện tại
  -> cosine similarity >= threshold
  -> gộp theo photo_path, giữ score lớn nhất
  -> đối chiếu danh sách ảnh với ground truth
```

Không gọi API nhập Drive, không chạy UI hay download ZIP trong thí nghiệm này. Dataset có thể được tải từ nguồn phát hành, nhưng toàn bộ inference/grid search dùng file local. Embedding phải do model thật tạo từ ảnh; không dùng vector giả hoặc embedding tải sẵn làm kết quả chính.

Đầu ra gồm baseline, toàn bộ điểm grid search, cấu hình ứng viên tốt nhất trên dev, kết quả test độc lập và danh sách ca sai để xem lại. “Tốt nhất” luôn gắn với dataset, dải grid và mục tiêu đã định; không hứa có một threshold tối ưu cho mọi album.

Giữ SCRFD + ArcFace hiện tại trong lượt đầu. Không train model. Chỉ thêm công cụ thí nghiệm và output riêng; chưa ghi đè `storage/best_config.json`, `faces.faiss` hoặc metadata production.

## 2. Baseline và những điểm phải xử lý trước khi chạy

| Thành phần | Baseline hiện tại | Cách dùng trong thí nghiệm |
|---|---|---|
| Detector | `models/scrfd_10g.onnx` | Dùng model local, ghi SHA256 |
| Embedder | `models/arcface_w600k_r50.onnx` | Dùng model local, ghi SHA256 |
| Detector input | `640 x 640` cấu hình | Ghi cả input tensor thực tế |
| `det_threshold` | `0.30` | Có grid search |
| `min_face_size` | `0` | Có grid search; đơn vị pixel trên ảnh nguồn sau map bbox về |
| NMS IoU | `0.40`, hardcode | Cố định trong lượt đầu |
| Similarity | `0.23` | Có grid search |
| Query selection | bbox diện tích lớn nhất | Giữ như runtime |
| Gallery | tất cả detection hợp lệ | Giữ như runtime |
| Retrieval | `IndexFlatIP.range_search` | Không cắt top-k trước threshold/gộp ảnh |

Các điểm cụ thể của code hiện tại:

1. `FaceEncoder._model_detector_size()` có thể lấy input shape cố định của ONNX thay cho `det_size` yêu cầu. Phải kiểm tra tensor thực sự trước khi coi 960/1280 là các cấu hình khác nhau. Lượt chính chỉ dùng 640. Không suy ra model hỗ trợ kích thước lớn từ việc sửa JSON.
2. `scripts/tune.py` chọn một primary face mỗi ảnh recognition. Cần evaluator gallery nhiều mặt riêng để đo đúng tác vụ này.
3. Không dùng điểm tốt nhất trên dev làm kết quả test; bộ test chỉ mở sau khi khóa cấu hình.
4. Cache detection của script cũ nằm sau NMS. Khi đổi minimum face size, lọc lại cache sau NMS có thể khác chạy filter-size rồi NMS như runtime. Cache mới cần bbox/score/landmarks trước filter-size và trước NMS, sau đó replay đúng thứ tự.
5. Không chọn detector chỉ theo recall WIDER rồi coi đó là cấu hình retrieval tốt nhất. Phải so cả tổ hợp detector và similarity threshold trên dev retrieval.

## 3. Bộ dữ liệu và thứ tự setup

### 3.1 WIDER FACE: kiểm tra detector

- Nguồn: [trang dự án WIDER FACE](https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/), chuyển tới trang tác giả.
- Tải validation images, bbox annotations và official evaluation files/code. Chưa cần tải train cho model đã có sẵn.
- Dùng toàn bộ 3.226 ảnh validation nếu tải được đủ; chia nội bộ dev/test 70/30 theo ảnh, stratify theo event và độ đông người, seed `20261125`.
- Giữ nguyên annotations gốc, cờ invalid/ignore và định nghĩa Easy/Medium/Hard. Khi báo AP chuẩn dùng official evaluator, không biến ảnh/box ignore thành negative.
- Đây là split nội bộ từ validation, phải ghi đúng tên; không gọi đó là official WIDER test.
- Đo AP theo official evaluator và recall tại operating point của cấu hình; báo riêng face nhỏ, đông người, pose và occlusion nếu annotation cho phép.
- WIDER không có identity ground truth dùng cho tác vụ này: không dùng WIDER để tự tạo kết luận đúng/sai người.

### 3.2 PIPA: ưu tiên cho retrieval trên ảnh album thực

- Nguồn: [metadata/crawler của tác giả](https://github.com/coallaoh/PIPA_dataset), [bài PIPA CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Zhang_Beyond_Frontal_Faces_2015_CVPR_paper.html).
- Đã xác minh trang metadata cảnh báo các link ảnh gốc không còn hoạt động; có thể thiếu ảnh Flickr và khác resolution. PIPA là nguồn ưu tiên có điều kiện, không giả định clone metadata là đã có dataset ảnh.
- Tải metadata, split và ảnh còn truy cập được qua nguồn được chỉ dẫn; ghi đủ requested/downloaded/missing và resolution. Dùng photo ID để nối ảnh với annotation.
- Xác minh tọa độ bbox khớp resolution tải về. Nếu không có kích thước gốc hoặc phép quy đổi tin cậy, không tự đoán tỷ lệ; giữ ảnh đó ngoài phần đánh giá cần bbox và báo số lượng.
- Giữ ảnh full scene làm gallery. Ground truth là identity theo từng head instance; không coi head bbox là face bbox khi tính face IoU.
- Dùng official validation identities cho dev, official test identities cho test. Giữ official day/album split nếu có đủ ảnh; chọn protocol chính trước khi chạy, đề xuất day split.
- Trong mỗi partition, tạo query từ head crop của ảnh phía 0, gallery full scene từ phía 1; có thể đảo chiều theo protocol. Sau mọi bước phải bảo đảm query source photo không nằm trong gallery cùng lượt.
- Crop query là ảnh đại diện người cần tìm, phải detect lại bằng encoder thật; không đưa landmarks/embedding ground truth vào model.
- Một ảnh full scene có thể chứa người ở nhiều partition. Chia theo connected components của quan hệ photo-identity để tránh cùng ảnh/identity xuất hiện giữa dev và test; nếu phải loại ảnh gây giao nhau, ghi rõ số ảnh/identity còn lại. Không tuyên bố giữ nguyên official protocol nếu đã đổi split.
- PIPA có cả người quay lưng. Gán thuộc tính `face_visibility` từ ground truth hoặc kiểm tra thủ công độc lập với detector. Báo riêng: ảnh có mặt nhìn thấy và toàn bộ person instances; không loại ca khó chỉ vì SCRFD không phát hiện.
- Đây là protocol photo retrieval dựng trên PIPA; không so trực tiếp điểm này với published PIPA person-classification accuracy.

### 3.3 Event local có nhãn: phương án chính nếu PIPA không đủ ảnh

Dùng ảnh event local trực tiếp; không cần import lại từ Drive. Bộ hiện có trong `data/event_photos` có thể làm pilot sau khi gán nhãn. Không coi 38 ảnh hoặc số detection đã biết là ground truth accuracy.

- Mục tiêu ban đầu: ít nhất 30 identity có trong gallery, 3 query độc lập/người và vài trăm ảnh nhóm; đây là mục tiêu thu thập, không phải số liệu đã có.
- Thêm ít nhất 100 query người không có trong gallery nếu có nguồn phù hợp. Pilot nhỏ hơn vẫn chạy được nhưng kết luận phải ghi rõ quy mô thực tế.
- Gán nhãn photo -> identity có mặt, bbox/head box nếu cần phân tích detection, visibility, size, blur, pose. Hai người kiểm tra lại các ca mơ hồ khi có thể.
- Query nên là selfie/portrait khác lần chụp. Không lấy crop từ chính ảnh gallery trong lượt đó. Crop từ cùng event nhưng khác ảnh chỉ là một track phụ.
- Chia dev/test theo nhóm identity và scene/event; cùng ảnh hoặc burst/near-duplicate phải nằm cùng partition. Nếu chỉ một event và không chia độc lập được, dùng làm pilot, không tuyên bố cross-event generalization.
- Khóa toàn bộ label trước khi xem dự đoán grid. Instance chưa xác định identity phải đánh dấu unknown annotation; không tính thành negative chắc chắn.

Nếu PIPA không đủ dữ liệu và event local chưa có label, chỉ hoàn thành được tuning detector và recognition phụ. Không thay thế âm thầm bằng LFW rồi tuyên bố đã tối ưu retrieval group photo.

### 3.4 Recognition phụ: TinyFace, LFW, VGGFace2

- [TinyFace official](https://qmul-tinyface.github.io/): tải ảnh và protocol từ link tác giả để stress-test mặt độ phân giải thấp. Bộ này thiết kế cho 1:N low-resolution recognition. Chạy detector thật trên crop để đo toàn pipeline; báo thêm track recognition dùng crop như input chuẩn nếu cần, ghi rõ khác protocol và số ca detector thất bại. Không upscale rồi ghi kích thước native đã thay đổi.
- LFW: giữ benchmark phụ, chạy đủ official 6.000 pairs/folds nếu có ảnh và pair files xác minh được. Với cấu hình khóa trước, báo fixed-threshold metrics. Threshold CV theo official folds, nếu chạy, phải là kết quả riêng. Không tiếp tục chọn thông số dựa trên 600 pairs cũ đã xem.
- [VGGFace2 official](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/): hiện trang gốc không còn link tải. Chỉ dùng nếu có bản local/nguồn khôi phục xác minh được; tách identity cho dev/test, không buộc setup phải phụ thuộc bộ này.
- Không dùng embedding có sẵn của các bộ trên thay cho ảnh local. Không cộng pair counts của các bộ với photo counts của PIPA/event thành một precision chung.
- Ghi nguồn training checkpoint nếu biết để kiểm tra khả năng identity/image overlap với benchmark; nếu chưa rõ thì ghi unknown, không gọi benchmark là hoàn toàn chưa từng xuất hiện trong training.

### 3.5 Setup local và kiểm tra đầu vào

```text
data/validation/gridsearch/
  raw/{wider,pipa,tinyface,lfw,event_local}/
  manifests/
    datasets.json
    images.jsonl
    instances.jsonl
    queries.jsonl
    splits.json
    missing_files.csv
storage/gridsearch/<run_id>/
  cache/
  results/
  report.md
```

Các thư mục này nằm trong nhánh validation/storage đã ignore hiện tại. Giữ ảnh gốc; lưu crop riêng kèm đường dẫn nguồn và tọa độ.

Schema tối thiểu:

| File | Trường bắt buộc |
|---|---|
| `datasets.json` | dataset, source URL/revision, đường dẫn, counts thực tế, phiên bản annotations |
| `images.jsonl` | dataset, image_id, path, width, height, sha256, scene_or_burst_group, partition |
| `instances.jsonl` | image_id, instance_id, identity_id, bbox_type, bbox, face_visibility, annotation_complete |
| `queries.jsonl` | query_id, identity_id, path, source_image_id, crop_bbox nếu có, partition, known_or_unknown |
| `splits.json` | seed, rules, photo/identity lists, query/gallery membership, gallery scenarios |

Namespace identity bằng dataset để tránh ID trùng. Phân biệt unknown person annotation với unknown query được xác nhận không thuộc gallery. Kiểm tra ảnh decode được, đường dẫn có thật, box nằm trong ảnh, hash/near-duplicate giữa partition, query source không nằm trong gallery, positive identity có ít nhất một gallery photo. Lưu danh sách thiếu/corrupt trước khi chạy model; mẫu không được phụ thuộc prediction.

## 4. Grid search đề xuất

### 4.1 Grid chính, tương thích encoder hiện tại

| Tham số | Giá trị |
|---|---|
| Effective detector input | `640 x 640` |
| `det_threshold` | `0.20, 0.30, 0.40, 0.50, 0.60` |
| `min_face_size` | `0, 8, 12, 16, 20, 32` |
| NMS IoU | `0.40` |
| `similarity_threshold` | `0.10..0.90`, bước `0.02`, thêm `0.23` |

Tổng 30 cấu hình detector x 42 thresholds = 1.260 tổ hợp coarse. Cả query và gallery dùng cùng detector settings như app. Không tune `top_k` vì production trả tất cả match qua threshold.

Baseline `640 / 0.30 / 0 / 0.23` luôn có trong kết quả. Nếu vùng tốt nằm sát biên similarity grid, mở rộng dev grid trước khi khóa cấu hình.

### 4.2 Fine grid chỉ trên dev

- Lấy tối đa 5 tổ hợp detector tốt theo quy tắc ở mục 7.
- Quanh similarity threshold coarse của từng tổ hợp, sweep khoảng +/-0.02, bước 0.002, giữ trong miền cosine [-1, 1].
- Bổ sung detector confidence +/-0.05, bước 0.025 và min size lân cận nếu dev cho thấy cần. Ghi lại toàn bộ grid mở rộng và tính lại lựa chọn trên dev.
- Không dùng test để chọn vùng fine hoặc đổi tiêu chí.

### 4.3 Nhánh mở rộng sau lượt chính

Chỉ thử input 960/1280 khi model export hỗ trợ thật, output decode hợp lệ, không warning shape và effective input khác 640. Nếu phải đổi model/export, coi đó là thí nghiệm khác, cache khác và baseline riêng. NMS/tiling/TTA/model khác cũng nằm ngoài grid chính; không trộn lẫn lợi ích của chúng vào kết luận tuning threshold hiện tại.

## 5. Cache để chạy nhanh nhưng tương đương runtime

1. Chạy SCRFD một lần/ảnh/effective input, lưu decoded boxes, scores, landmarks trước lọc kích thước và NMS. Có thể bỏ score < 0.20 vì đó là biên thấp nhất của grid chính.
2. Với từng cấu hình detector, replay đúng runtime: map/clip -> score filter -> size filter -> NMS -> confidence sort.
3. Gallery giữ mọi face còn lại. Query chọn largest bbox từ chính danh sách của cấu hình đó. Lựa chọn query có thể đổi theo grid, không cache cố định một primary embedding cho mọi cấu hình.
4. Align/embed các face được dùng; cache theo image hash + bbox/landmarks + model/preprocessing hash. Một detection được dùng ở nhiều cấu hình chỉ cần ArcFace một lần.
5. Dựng FAISS trong RAM/thư mục run theo từng cấu hình detector. Similarity sweep tái dùng embeddings, không gọi ArcFace lại.
6. Có thể cache ma trận score theo block rồi threshold/gộp ảnh cho nhanh. Kiểm tra một mẫu score và kết quả ở baseline, threshold biên, cấu hình min-size khác bằng `VectorStore`/`SearchEngine` thật; kết quả phải tương đương.
7. Không cache giữ NMS output ở min-size=0 rồi lọc lại cho mọi min-size. Box bị loại vì size có thể từng suppress box khác, nên cần chạy lại NMS sau size filter.

Cache key gồm model SHA256, ảnh/manifest hash, effective input, preprocessing/alignment/decoder version, dtype. Đổi geometry/model phải invalid cache liên quan. Không rebuild toàn bộ embedding chỉ vì đổi similarity threshold.

## 6. Thiết kế evaluation

### 6.1 Gallery và ground truth

- Mỗi dataset tạo gallery riêng, identity label riêng. Chỉ trộn distractor giữa nguồn khi đã kiểm tra identity overlap và ghi thành scenario riêng.
- Dùng các quy mô gallery lồng nhau 100/500/1.000/5.000 ảnh nếu dữ liệu đủ. Chốt danh sách bằng seed trước grid, giữ positive photos và tăng distractors; không nhân bản ảnh để đạt quota.
- Scenario chính: gallery lớn nhất khả dụng đã định trước. Báo quy mô ảnh, face vectors, số mặt/ảnh; luôn có scenario group photo từ 3 người trở lên nếu đủ nhãn.
- Photo được coi là relevant nếu ground truth xác nhận target có trong ảnh theo track visibility đã định. Nếu detector bỏ sót target thì ảnh vẫn là relevant, và sẽ tính FN khi không được trả về.
- Với query fail/no-face hoặc selected face sai người: tính là truy vấn thất bại trong end-to-end recall/Hit và báo coverage; không loại ra để điểm đẹp hơn. Query crop phải xác nhận đúng target; phát hiện sai selection bằng kiểm tra box/label hoặc review.
- Unknown query có zero relevant gallery photo; đo tỷ lệ query này trả ít nhất một ảnh và số ảnh sai/query.
- Báo hai track PIPA/event: visible-face retrieval và all-person coverage. Mẫu số của mỗi track phải khóa từ label trước inference.

### 6.2 Metrics

| Metric | Định nghĩa/mục đích |
|---|---|
| Photo precision | TP photos / (TP + FP photos), sau gộp theo đường dẫn |
| Photo recall | TP photos / tất cả GT relevant photos, gồm target bị detector bỏ sót |
| Macro recall | Trung bình recall trên positive query; sau đó theo identity để người nhiều query không lấn át |
| False photos/query | Tổng FP photos / số queries |
| Unknown false-return rate | Số unknown queries trả >=1 ảnh / tất cả unknown queries |
| Hit@1/5/10 | Positive queries có ít nhất một đúng trong top K ảnh trả về; fail query tính 0 |
| Recall@K | Tỷ lệ relevant photos tìm được trong top K, không nhầm với Hit@K |
| Coverage | Query encode thành công, gallery photos/vectors, GT faces missed |
| Detection | Official WIDER AP, operating-point recall/precision, FP/image, theo kích thước |
| Runtime | Cold load riêng; warm encode/query và search p50/p95; peak RAM; vectors/giây |

Precision khi không trả gì ghi undefined ở query đó, đồng thời vẫn báo zero return, recall=0 cho positive query; không cho hệ thống trả rỗng đạt pass chỉ nhờ precision quy ước. Báo micro precision, macro recall và số TP/FP/FN cùng nhau.

Pair TAR/FAR/AUC của LFW hoặc rank metric TinyFace chỉ là bảng bổ sung. Không đổi tên pair accuracy thành photo retrieval accuracy.

## 7. Quy tắc chọn cấu hình tốt nhất

Chốt mục tiêu vận hành trước khi chạy. Đề xuất cho app tìm ảnh cá nhân:

1. Precision ảnh trên dev retrieval >= 0.98.
2. Unknown false-return rate <= 0.01 khi có đủ unknown labels; luôn báo numerator/denominator và độ bất định. Thiếu unknown queries thì trạng thái tiêu chí này là NOT_EVALUATED, không tự coi pass.
3. Trong cấu hình đạt điều kiện, maximize macro recall theo identity trên dataset event/PIPA chính.
4. Tie-break: ít FP/query hơn -> p95 query thấp hơn -> cấu hình đơn giản hơn.
5. Khi có cả PIPA và local event, báo riêng hai nguồn, precision guard áp dụng từng nguồn; dùng trung bình đều macro recall hai nguồn để xếp hạng, tránh bộ lớn áp đảo. WIDER và TinyFace không được cộng vào mẫu số này.

Các mốc 0.98/0.01 là mục tiêu đề xuất, không phải chỉ số đã đo. Nếu không có cấu hình thỏa, xuất `NO_FEASIBLE_CONFIG` và Pareto precision-recall-latency; không hạ yêu cầu âm thầm hoặc ghi winner đạt chuẩn. Vẫn xuất best-effort candidate, kèm điều kiện thất bại.

Lấy khoảng tin cậy theo identity/scene groups trên bộ test khi số nhóm cho phép; query cùng người/burst không coi độc lập. Với unknown ít mẫu hoặc zero FP, ghi count và bound phù hợp, không suy ra FPR thật bằng 0. Chọn winner trên dev; uncertainty test dùng mô tả, không dùng quay lại retune.

## 8. Trình tự triển khai và chạy

### A. Setup, khóa dữ liệu

- [ ] Kiểm tra models, ONNX input/output thực tế và dependencies từ `requirements.txt`.
- [ ] Setup WIDER validation; setup PIPA khả dụng; thêm TinyFace/LFW nếu truy cập được.
- [ ] Nếu PIPA không đủ, setup event local có nhãn theo mục 3.3; ghi chính xác phạm vi khả dụng.
- [ ] Tạo manifests/splits; xem một mẫu ảnh + annotation để xác minh bbox/identity/resolution.
- [ ] Tạo gallery scenarios/known/unknown lists và khóa trước grid.

### B. Pilot correctness và tốc độ

- [ ] Chạy vài chục ảnh gồm single-face, group, tiny/blurred, no-face.
- [ ] Đối chiếu cache replay với encoder/runtime ở một số cấu hình, nhất là min-size khác 0.
- [ ] Xác minh query không được add vào index, match gộp theo ảnh, largest-face selection đúng.
- [ ] Chạy baseline trên dev; ước lượng thời gian/RAM để chọn worker count và chunk size.

### C. Grid search dev

- [ ] Build cache detection/embedding; xử lý theo block, ghi progress và resume được.
- [ ] Chạy đủ 1.260 tổ hợp coarse trên cùng tập dev và gallery scenarios.
- [ ] Xuất kết quả từng nguồn, slices, Pareto; chạy fine grid theo mục 4.2.
- [ ] Xem các FP/FN dev: missed detection, wrong query face, poor embedding, threshold, label ambiguity.
- [ ] Khóa winner/best-effort và hash config; ghi tiêu chí pass/fail trước khi mở test.

### D. Test độc lập và handoff

- [ ] Chạy baseline và winner đã khóa trên test, cùng manifests/scenarios.
- [ ] Báo chênh lệch precision/recall/unknown false-return/latency và số mẫu thực tế.
- [ ] Chạy WIDER/TinyFace/LFW track test tương ứng với cấu hình đã khóa; không chọn lại threshold.
- [ ] Export candidate config tương thích loader hiện tại, chỉ vào thư mục run.
- [ ] Kết luận có đủ bằng chứng cải thiện không; nếu test không đạt thì báo kết quả đó. Một vòng tuning tiếp theo cần test mới để giữ tính độc lập.

## 9. Công cụ và command dự kiến cần viết

Chưa có các script dưới đây trong repo. Đây là interface triển khai dự kiến, không phải lệnh đã sẵn sàng chạy.

```powershell
python -B codex/setup_gridsearch.py --root data/validation/gridsearch --datasets wider pipa tinyface lfw --seed 20261125
python -B codex/run_gridsearch.py --stage validate --data-root data/validation/gridsearch --run-id local-v1
python -B codex/run_gridsearch.py --stage baseline-dev --run-id local-v1
python -B codex/run_gridsearch.py --stage coarse --run-id local-v1 --resume
python -B codex/run_gridsearch.py --stage fine --run-id local-v1 --resume
python -B codex/run_gridsearch.py --stage select --run-id local-v1
python -B codex/run_gridsearch.py --stage test --run-id local-v1
```

`setup_gridsearch.py`: download/resume ảnh và annotation, adapters, manifest/split validation; báo nguồn không truy cập được và exact local input cần có. Không tự chế ground truth identity. Việc gán nhãn event là bước riêng trong setup.

`run_gridsearch.py`: dùng lại encoder/store, wrapper thí nghiệm để lấy pre-NMS outputs, cache, score, select và report. Không cần framework hay service mới. Các stage sau đọc `run.json` để lấy đường dẫn/seed/config đã khóa.

Progress cần ghi ảnh đã xử lý, faces, cấu hình hoàn thành, ETA từ pilot, lỗi theo loại. Resume chỉ dùng cache có hash khớp. Lỗi download được ghi trước khi freeze manifest; lỗi inference sau freeze là kết quả failure và giữ trong coverage, không xóa mẫu.

## 10. Artifacts và điều kiện hoàn thành

Trong `storage/gridsearch/<run_id>/`:

- `run.json`: datasets/manifests/models/code hashes, seed, environment, effective input, grid, objective, trạng thái từng stage.
- `results/detector_grid.csv`, `retrieval_coarse.csv`, `retrieval_fine.csv`: counts, metrics, dataset, split, gallery scenario, đầy đủ parameters.
- `results/per_query.csv`: target, success/failure, TP/FP/FN, coverage, thời gian; dùng identity ẩn danh trong output chia sẻ.
- `results/dev_selection.json`: winner hoặc NO_FEASIBLE_CONFIG, lý do chọn, constraints, baseline comparison.
- `results/test_metrics.json`: baseline và winner, metrics/slices/counts/uncertainty.
- `results/failure_cases.csv`: đường dẫn ảnh và bbox mặt gây match để kiểm tra trực tiếp.
- `results/best_config.candidate.json`: cùng schema cấu hình app; thông tin mở rộng nằm trong `run.json`.
- `report.md`: dataset thực sự chạy, phần thiếu, split, đồ thị precision-recall, ảnh lỗi tiêu biểu, kết luận trong phạm vi đo.

Hoàn thành khi có retrieval test độc lập trên group photo có identity labels, baseline comparison, đầy đủ ca fail và candidate config truy nguyên được. Chỉ có WIDER/LFW pass thì đánh dấu PARTIAL, chưa gọi là tối ưu xong ứng dụng tìm ảnh event.

## 11. Nguồn setup đã kiểm tra

- WIDER: <https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/> — trang cũ chuyển sang tác giả; cần xác minh link archive thực tế lúc setup.
- PIPA metadata/splits/crawler và cảnh báo missing photos/resolution: <https://github.com/coallaoh/PIPA_dataset>.
- PIPA paper: <https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Zhang_Beyond_Frontal_Faces_2015_CVPR_paper.html>.
- TinyFace images/evaluation links: <https://qmul-tinyface.github.io/>.
- VGGFace2 và trạng thái không còn link tải official: <https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/>.

Các link là điểm bắt đầu setup; plan này chưa xác nhận đã tải được archive đầy đủ. Việc chuẩn bị nguồn và label là đầu việc cụ thể trong giai đoạn A.
