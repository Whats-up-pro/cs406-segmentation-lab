# Phần triển khai và bằng chứng của đồ án

Bộ đồ án triển khai pipeline RGB → phân đoạn → mask → đánh giá và demo. Bảng dưới liệt kê đầu ra kiểm tra được và lý do thiết kế; không gán lịch sử công việc cho từng cá nhân khi chưa có thông tin thực tế.

| Phần triển khai | Vì sao thực hiện | Bằng chứng |
| --- | --- | --- |
| Kiểm kê và EDA | Xác định số mẫu, chất lượng pixel và bất đồng annotation trước khi diễn giải mô hình | `scripts/audit_data.py`, `scripts/eda.py`, dữ liệu JSON/CSV tại `Project/results` |
| Đối chiếu nguồn và leakage | Phân biệt test nội bộ với test cuộc thi; kiểm tra byte trùng, ENC và ứng viên perceptual | `docs/data-provenance-verified.md`, `scripts/audit_leakage.py`, bản kê trong `Project/results/completion/provenance` |
| Baseline xử lý ảnh | Tạo đối chứng có giả định màu sắc, ngưỡng và morphology rõ ràng | `backend/baseline.py`, `tests/test_baseline.py`, bảng test đã lưu |
| Logistic có giám sát | Kiểm tra biểu diễn màu và thống kê cục bộ khi chỉ train cập nhật hệ số | `backend/pixel_model.py`, `scripts/train_pixel_model.py`, manifest hai epoch train và chọn ngưỡng trên validation |
| Compact U-Net từ đầu | Bổ sung mạng có lịch sử độc lập với checkpoint được cung cấp | `backend/compact_model.py`, `scripts/train_compact.py`; recipe chính đã đóng băng đủ 20 epoch |
| Ablation và nhiều seed | Tách quan sát về loss/augmentation dưới cùng ngân sách; không chọn seed theo test | `config/experiments/compact_ablation.json`, `scripts/run_compact_experiments.py`; chỉ báo kết quả từng run sau khi hoàn tất |
| Đánh giá và phân tích | Giữ chung input/consensus, kiểm tra frozen manifest, phân biệt macro/micro và ước lượng bất định ghép cặp | `scripts/evaluate_completion.py`, `scripts/analyze_completion.py`, CSV/hash đầu ra |
| TransUNet tham khảo | Kiểm tra và dùng trọng số được cung cấp, ghi rõ nguồn và giới hạn lịch sử | `backend/transunet.py`, `backend/inference.py`, giấy phép và NOTICE upstream, manifest checkpoint |
| Demo và tái lập | Cho phép chạy predictor thật, đổi annotation, xem sai khác và xuất mask đúng kích thước | `backend/main.py`, `frontend`, `scripts/check_ui.py`, hướng dẫn chạy và kiểm thử |

Kết quả hiện có cho thấy Logistic hai epoch không cải thiện LAB–Otsu, còn checkpoint TransUNet khớp benchmark cục bộ hơn nhưng chưa biết exposure trong huấn luyện gốc. Những nhận xét về Compact/ablation/ISIC phải lấy từ các manifest hoàn tất tương ứng. Đầu ra và các quyết định này là nội dung nhóm cần giải thích trong bảo vệ; việc có tài liệu không xác nhận thay khả năng hiểu và trả lời của từng thành viên.
