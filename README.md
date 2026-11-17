# Segmentation Lab · CS406

Ứng dụng local phục vụ đồ án Xử lý ảnh và Ứng dụng: khám phá ảnh và nhãn, phân đoạn nhị phân bằng LAB–Otsu, Logistic pixel hoặc TransUNet, rồi đối chiếu dự đoán với nhãn tham chiếu.

Repo này chứa mã backend, frontend, huấn luyện, đánh giá, EDA và kiểm thử. Checkpoint TransUNet được quản lý qua GitHub Releases; dataset, kết quả thực nghiệm, báo cáo và slide được quản lý riêng.

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

Checkpoint có kích thước **421.301.094 byte**; thông tin kiểm chứng nằm trong [manifest mô hình](models/transunet-checkpoint.json). File lớn hơn giới hạn 100 MiB của Git thông thường nên được phát hành dưới dạng [release asset](https://github.com/Whats-up-pro/cs406-segmentation-lab/releases/tag/checkpoint-v1), thay vì commit trực tiếp file `.pth`. Khi asset đã được phát hành, tải bằng GitHub CLI:

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
gh release download checkpoint-v1 --repo Whats-up-pro/cs406-segmentation-lab --pattern TransUNet_R50_ViT_B_16_all_best_model.pth --dir data
Get-FileHash -Algorithm SHA256 data/TransUNet_R50_ViT_B_16_all_best_model.pth
```

SHA-256 cần khớp `6df81c63fab9936d8d90e79bfea0264c11516e82d981c70ab62c89758793a508`. Có thể dùng biến môi trường `TRANSUNET_CHECKPOINT` để chỉ định đường dẫn khác. `TRANSUNET_DEVICE` nhận `auto`, `cpu` hoặc `cuda`; `TRANSUNET_CPU_THREADS` mặc định là 4.

Logistic pixel dùng model tại `results/pixel_classifier/model.json`. Tạo model bằng lệnh huấn luyện bên dưới, hoặc lấy model đã học từ người cung cấp tài nguyên. Khi chưa có trọng số, chọn LAB–Otsu trong demo.

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

## Cấu trúc mã

| Thư mục | Nội dung |
| --- | --- |
| [backend](backend/) | FastAPI, dữ liệu, ba bộ dự đoán và các độ đo |
| [frontend](frontend/) | Giao diện HTML/CSS/JavaScript |
| [scripts](scripts/) | Kiểm kê, EDA, huấn luyện và đánh giá |
| [tests](tests/) | Kiểm thử bằng dữ liệu giả và smoke test tùy chọn |
| [third_party/TransUNet](third_party/TransUNet/) | Mã nguồn tham khảo, phiên bản và giấy phép upstream |

Các route trình chiếu và tải báo cáo chỉ hoạt động khi bộ tài liệu riêng đã được đặt trong `deliverables/`.

## Nguồn và giới hạn

Kiến trúc TransUNet được điều chỉnh từ [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet), revision `02ef0010b36eb8328b5e689eadaf613602edf9b8`. Bản sao nguồn, giấy phép Apache-2.0 và mô tả thay đổi được giữ tại [NOTICE](third_party/TransUNet/NOTICE.md) và [LICENSE upstream](third_party/TransUNet/LICENSE). Giấy phép nguồn upstream không xác định quyền phân phối dataset hoặc checkpoint riêng.

Đội **Anastasia** đứng đầu bảng Segmentation Task Results của MEDIQA-MAGIC 2025 với Jaccard **0,645774234**, Dice **0,784766**, theo [bảng kết quả chính thức](https://www.imageclef.org/2025/medical/mediqa). Hai lần nộp của `khoingotuantu` bạn đọc có thể đối chiếu trong [ghi chú nguồn](docs/data-and-checkpoint.md). Đây là kết quả của đội dự thi; mối liên hệ giữa checkpoint cục bộ với các lần nộp đó còn cần xác nhận.

Bản dữ liệu cục bộ có số lượng khác bản segmentation 2025 công bố; phiên bản, cách chọn ảnh/hợp nhất nhãn, quyền sử dụng và lịch sử huấn luyện checkpoint chưa được xác nhận đầy đủ. Chuẩn hóa TransUNet mặc định là RGB/255; đây là lựa chọn thực nghiệm, chưa được xác nhận là công thức huấn luyện gốc. Đồ án chưa được thẩm định để sử dụng trong chẩn đoán y tế.
