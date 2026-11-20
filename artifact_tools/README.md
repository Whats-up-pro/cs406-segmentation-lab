# Nguồn tạo báo cáo và slide

Các script trong thư mục này tạo báo cáo Word/PDF, PowerPoint có bảng/biểu đồ chỉnh sửa được, bản trình chiếu HTML ngoại tuyến và hướng dẫn bảo vệ. Chúng tách khỏi môi trường chạy ứng dụng; không cần cài công cụ tài liệu để dùng demo hoặc huấn luyện.

Đầu vào là thư mục `Project` đầy đủ trong gói đồ án riêng: bản báo cáo gốc, hình minh họa thật, dữ liệu và kết quả. Ảnh y tế, trọng số và kết quả lớn không nằm trong Git. Các script mặc định tìm `Project` cạnh repository; có thể đặt `CS406_PROJECT_DIR` và `CS406_SOURCE_DIR` để chọn vị trí khác.

```powershell
$env:CS406_PROJECT_DIR = 'D:\HK1_2026-2027\CS406\Project'
$env:CS406_SOURCE_DIR = (Get-Location).Path
```

`prepare_completion_report.py` chỉ tạo bản cuối khi đủ sáu run 20 epoch, chín phương pháp trên 295 ảnh local và 100 ảnh ISIC, phân tích có hash khớp và hình lỗi từ mask đã lưu. Trọng số/ngưỡng phải giống nhau giữa hai lần đánh giá. `--draft` chỉ dành cho kiểm tra bố cục riêng, không được xuất bản thành báo cáo cuối.

```powershell
python artifact_tools/prepare_completion_report.py --evaluation-dir ../Project/results/completion/evaluation-local --analysis-dir ../Project/results/completion/analysis-local --external-dir ../Project/results/completion/evaluation-isic --external-analysis-dir ../Project/results/completion/analysis-isic --case-dir ../Project/results/completion/cases-local
```

`build_completion_report.py` dùng `python-docx` để dựng Word. `package_completion_report.py` dùng bộ document runtime để render DOCX thật, lặp tới khi số trang mục lục ổn định, rồi thêm điều hướng PDF. `render_report.mjs` và `artifact_runtime_compat.mjs` là cầu nối tới runtime và font hệ thống, không sửa thư viện được quản lý.

`prepare_completion_presentation.py` nhận các thư mục đánh giá/phân tích/hình lỗi như lệnh trên, thêm `--runs-dir ../Project/results/completion/compact-final --output ../Project/output/completion/presentation-input.json`. Script kiểm tra lịch sử đóng băng và ảnh demo đúng model 20 epoch. Dùng JSON này cho cả hai lệnh:

```powershell
node artifact_tools/build_completion_presentation.mjs ../Project/output/completion/presentation-input.json ../Project/output/completion/defense-build
python artifact_tools/prepare_completion_defense.py --input ../Project/output/completion/presentation-input.json --output ../Project/docs/defense-completion.md
```

Phần xuất PDF/PPTX dùng `@oai/artifact-tool` của Codex cùng các tiện ích kiểm tra gói của skill Presentations. `CODEX_ARTIFACT_RUNTIME` chọn thư mục runtime nếu khác vị trí mặc định. Đây là phụ thuộc của công cụ soạn tài liệu; bản Word/PowerPoint xuất ra mở bằng ứng dụng thông thường. Không tuyên bố các công cụ soạn tài liệu đã được cài trong môi trường sạch của demo.

Thư mục build là bản riêng để kiểm tra. Phải render và xem toàn bộ trang/slide, kiểm tra số liệu, chữ, mục lục và các liên kết trước khi đưa vào bộ nộp. Kiểm tra cấu trúc thành công chưa chứng minh tài liệu đã được duyệt trực quan.
