# Segmentation Lab · CS406

Ứng dụng local phục vụ đồ án Xử lý ảnh và Ứng dụng: khám phá ảnh và nhãn, phân đoạn nhị phân bằng LAB–Otsu, Logistic pixel, Compact U-Net hoặc TransUNet, rồi đối chiếu dự đoán với nhãn tham chiếu.

Repo này chứa mã backend, frontend, huấn luyện, đánh giá, EDA và kiểm thử. Checkpoint TransUNet được quản lý qua GitHub Releases; dataset, kết quả thực nghiệm, báo cáo và slide được quản lý riêng.

Sáu lượt huấn luyện Compact U-Net hoàn tất đủ 20 epoch. Trên 295 ảnh test cục bộ, cấu hình chính đạt Dice **0,6143** và LAB–Otsu đạt **0,4435**. Trên 100 ảnh ISIC 2018 validation, hai phương pháp lần lượt đạt **0,6722** và **0,6905**. Gói nộp chứa trọng số, điểm từng ảnh, cấu hình ablation, kết quả ba seed, báo cáo và slide. [Thiết kế và giao thức đánh giá](docs/experiments-completion.md).

## Quick start

Cần Python 3.11 trở lên. Chạy các lệnh sau tại thư mục repo trong PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Expected output: `Uvicorn running on http://127.0.0.1:8000`.

Mở **http://127.0.0.1:8000**. Có thể tải ảnh lên và chạy phương pháp LAB–Otsu mà không cần trọng số học sâu. Ảnh tải lên không có nhãn sẽ không hiện điểm đánh giá. Tài liệu API ở **http://127.0.0.1:8000/docs**.

Sau khi cài thư viện, cũng có thể khởi động bằng `./start.ps1`.

## Chuẩn bị dữ liệu và mô hình

Nguồn tham khảo gồm [DermaVQA](https://github.com/velvinnn/DermaVQA), [DermaVQA-DAS / MEDIQA-MAGIC 2025](https://www.imageclef.org/2025/medical/mediqa) và [tài nguyên, mã đánh giá năm 2025](https://github.com/wyim/ImageCLEF-MAGIC-2025). Xem [đối chiếu nguồn dữ liệu và checkpoint](docs/data-and-checkpoint.md) để phân biệt các phiên bản và kết quả cuộc thi. Lấy bản dữ liệu đã chuẩn bị từ người cung cấp tài nguyên đồ án, sau đó đặt vào cấu trúc sau:

```text
data/
  imagesTrain/       labelsTrain/
  imagesValid/       labelsValid/
  imagesTest/        labelsTest/
  TransUNet_R50_ViT_B_16_all_best_model.pth
```

Ảnh train/validation ghép với mask TIFF theo cùng tên gốc. Nhãn test có hậu tố `_mask_annN`; ứng dụng hỗ trợ từng annotation hoặc mặt nạ đồng thuận theo đa số nghiêm ngặt, hòa phiếu là nền. Pixel khác 0 được hiểu là foreground.

Để dùng TransUNet, cài thêm PyTorch:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

Checkpoint đã được phát hành, kích thước **421.301.094 byte**; thông tin kiểm chứng nằm trong [manifest mô hình](models/transunet-checkpoint.json). Tải từ [GitHub Releases](https://github.com/Whats-up-pro/cs406-segmentation-lab/releases/tag/checkpoint-v1) bằng GitHub CLI:

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
gh release download checkpoint-v1 --repo Whats-up-pro/cs406-segmentation-lab --pattern TransUNet_R50_ViT_B_16_all_best_model.pth --dir data
Get-FileHash -Algorithm SHA256 data/TransUNet_R50_ViT_B_16_all_best_model.pth
```

SHA-256 cần khớp `6df81c63fab9936d8d90e79bfea0264c11516e82d981c70ab62c89758793a508`. Có thể dùng biến môi trường `TRANSUNET_CHECKPOINT` để chỉ định đường dẫn khác. `TRANSUNET_DEVICE` nhận `auto`, `cpu` hoặc `cuda`; `TRANSUNET_CPU_THREADS` mặc định là 4.

Logistic pixel dùng model tại `results/pixel_classifier/model.json`. Tạo model bằng lệnh huấn luyện bên dưới, hoặc lấy model đã học từ người cung cấp tài nguyên. Khi chưa có trọng số, chọn LAB–Otsu trong demo.

Có thể dùng dữ liệu và kết quả ở thư mục khác mà không sao chép vào repo:

```powershell
$env:SEGMENTATION_DATA_DIR = 'D:\duong-dan\Project\data'
$env:SEGMENTATION_RESULTS_DIR = 'D:\duong-dan\Project\results'
$env:SEGMENTATION_DELIVERABLES_DIR = 'D:\duong-dan\Project\deliverables'
$env:PIXEL_MODEL_PATH = 'D:\duong-dan\Project\results\pixel_classifier\model.json'
$env:COMPACT_CHECKPOINT = 'D:\duong-dan\Project\results\completion\compact\bce_dice_aug_s406\best.pth'
```

Compact U-Net chỉ xuất hiện như một lựa chọn chạy được khi trọng số khớp SHA-256 trong `frozen-selection.json` nằm cùng thư mục. Ngưỡng mặc định lấy từ validation của lần huấn luyện đó.

## Huấn luyện và đánh giá

Chạy từ thư mục repo sau khi chuẩn bị dữ liệu:

```powershell
.venv\Scripts\python.exe -m scripts.eda --data-dir data --output results/eda
.venv\Scripts\python.exe -m scripts.train_pixel_model --data-dir data --output results/pixel_classifier --epochs 2 --seed 406
.venv\Scripts\python.exe -m scripts.evaluate --data-dir data --method both --split test --normalization unit --per-annotation --output results/comparison_test
New-Item -ItemType Directory -Force -Path docs | Out-Null
.venv\Scripts\python.exe -m scripts.analyze_results --output results/analysis
```

EDA và đánh giá xuất JSON, CSV và hình vào `results/`; giao diện đọc các kết quả khi chúng có sẵn. Huấn luyện Logistic chỉ dùng train để cập nhật hệ số và validation để chọn ngưỡng trước khi đánh giá test. Chọn thư mục output mới khi chạy lại huấn luyện vì script từ chối ghi đè run cũ.

CLI fine-tuning TransUNet tách riêng khỏi inference:

```powershell
.venv\Scripts\python.exe -m scripts.train --data-dir data --output results/transunet_finetune --epochs 10 --batch-size 2 --normalization unit --device auto --seed 406
```

Lệnh này khởi tạo một thí nghiệm mới từ checkpoint được cung cấp; không mô tả một lần fine-tuning đầy đủ đã hoàn tất. Huấn luyện toàn bộ mạng cần đủ bộ nhớ; xem `--help` cho chế độ `--trainable head` và các giới hạn số batch.

`scripts/audit_data.py` kiểm kê tên file, split và archive gốc; mặc định tìm `xulyanhungdung.zip` tại thư mục repo. Chỉ dùng bước đối chiếu archive khi có tệp này.

## Thí nghiệm Compact U-Net và Kaggle

Mô hình gọn được khởi tạo ngẫu nhiên, tách biệt với checkpoint TransUNet được cung cấp. [Giao thức cố định](config/experiments/compact_ablation.json) gồm bốn tổ hợp loss × augmentation ở seed 406 và tổ hợp BCE + Dice + augmentation ở seed 407, 408. Mỗi run chạy đủ 20 epoch; train cập nhật trọng số, validation chọn checkpoint và ngưỡng. Chuỗi RGB dùng chung là bilinear về 256 rồi về 128; nhãn dùng nearest neighbor. Test cục bộ đã được xem trước đợt mở rộng này, nên kết quả là benchmark hồi cứu có cấu hình đóng băng.

[Hướng dẫn Kaggle](docs/kaggle.md) cung cấp notebook GPU và cách tạo gói dữ liệu riêng tư. Notebook và mã huấn luyện đã có kiểm thử; lịch sử, trọng số và chỉ số của một run chỉ được coi là kết quả sau khi run thực sự hoàn tất.

```powershell
.venv\Scripts\python.exe -m scripts.audit_leakage --data-dir data --output results/completion/leakage
.venv\Scripts\python.exe -m scripts.run_compact_experiments --data-dir data --cache-dir results/completion/compact-cache --output results/completion/compact --device auto --epochs 20
.venv\Scripts\python.exe -m scripts.evaluate_completion --data-dir data --runs-dir results/completion/compact --cache-dir results/completion/compact-cache --pixel-model results/pixel_classifier/model.json --output results/completion/evaluation --device auto
.venv\Scripts\python.exe -m scripts.analyze_completion --evaluation-dir results/completion/evaluation --runs-dir results/completion/compact --output results/completion/analysis
.venv\Scripts\python.exe -m scripts.render_completion_cases --evaluation-dir results/completion/evaluation --data-dir data --output results/completion/cases
```

Đánh giá kiểm tra manifest, lịch sử và hash trước khi mở tập test. Phân tích xuất macro/micro từ evaluator, khoảng tin cậy bootstrap ghép cặp theo encounter, ablation, biến thiên giữa seed, đường học và các ca lỗi cần xem ảnh. Encounter trong tên file chưa được xác nhận là mã bệnh nhân. [Đối chiếu nguồn đã xác minh](docs/data-provenance-verified.md) giải thích sự khác nhau giữa dữ liệu cục bộ, DermaVQA 2024 và DermaVQA-DAS 2025.

## Kiểm thử

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-ml.txt
.venv\Scripts\python.exe -m pytest -q
```

Các test thông thường tạo dữ liệu giả trong thư mục tạm. Test suy luận bằng checkpoint thật được bật riêng khi đã có dữ liệu và trọng số:

```powershell
$env:RUN_MODEL_SMOKE = "1"
.venv\Scripts\python.exe -m pytest tests/test_inference.py -q
Remove-Item Env:RUN_MODEL_SMOKE
```

Nếu máy có Node.js, kiểm tra cú pháp frontend bằng `node --check frontend/app.js`.

[Danh sách phiên bản đã kiểm tra trên Windows CPU/Python 3.13.7](requirements-windows-cpu-tested.txt) được lấy từ môi trường mới có `include-system-site-packages=false`. Môi trường này đã chạy đầy đủ bộ kiểm thử và nạp checkpoint TransUNet thật trên ảnh validation. Có thể cài bằng `.venv\Scripts\python.exe -m pip install -r requirements-windows-cpu-tested.txt`; notebook Kaggle dùng môi trường GPU sẵn có của Kaggle.

## Cấu trúc mã

| Thư mục | Nội dung |
| --- | --- |
| [backend](backend/) | FastAPI, dữ liệu, bốn bộ dự đoán và các độ đo |
| [frontend](frontend/) | Giao diện HTML/CSS/JavaScript |
| [scripts](scripts/) | Kiểm kê, EDA, huấn luyện và đánh giá |
| [artifact_tools](artifact_tools/) | Dựng báo cáo, slide và hướng dẫn bảo vệ từ kết quả đã kiểm chứng |
| [tests](tests/) | Kiểm thử bằng dữ liệu giả và smoke test tùy chọn |
| [third_party/TransUNet](third_party/TransUNet/) | Mã nguồn tham khảo, phiên bản và giấy phép upstream |

Các route trình chiếu và tải báo cáo chỉ hoạt động khi bộ tài liệu riêng đã được đặt trong `deliverables/`.

## Nguồn và giới hạn

Kiến trúc TransUNet được điều chỉnh từ [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet), revision `02ef0010b36eb8328b5e689eadaf613602edf9b8`. Bản sao nguồn, giấy phép Apache-2.0 và mô tả thay đổi được giữ tại [NOTICE](third_party/TransUNet/NOTICE.md) và [LICENSE upstream](third_party/TransUNet/LICENSE). Giấy phép nguồn upstream không xác định quyền phân phối dataset hoặc checkpoint riêng.

Đội **Anastasia** đứng đầu bảng Segmentation Task Results của MEDIQA-MAGIC 2025 với Jaccard **0,645774234**, Dice **0,784766**, theo [bảng kết quả chính thức](https://www.imageclef.org/2025/medical/mediqa). Hai lần nộp của `khoingotuantu` bạn đọc có thể đối chiếu trong [ghi chú nguồn](docs/data-and-checkpoint.md). Đây là kết quả của đội dự thi; mối liên hệ giữa checkpoint cục bộ với các lần nộp đó còn cần xác nhận.

Đã xác minh train + test cục bộ chia lại toàn bộ 2.474 ID ảnh train của nguồn; validation giữ nguyên 157 ID. Không có ảnh thuộc 314 ID test chính thức. Cả 2.631 ảnh khớp kích thước và CRC32 của thành viên ZIP nguồn. Quy tắc hợp nhất mask train/valid, điều kiện mask segmentation và lịch sử checkpoint vẫn chưa được xác nhận đầy đủ. Chuẩn hóa TransUNet mặc định là RGB/255, chưa xác nhận là công thức huấn luyện gốc. Đồ án chưa được thẩm định sử dụng trong chẩn đoán y tế.
