# Hoàn thiện và nâng mức nghiên cứu đồ án CS406

## Mục tiêu và thẩm quyền

Người dùng yêu cầu tiếp tục tự chủ đến khi toàn bộ yêu cầu trong DOCX được xử lý và chất lượng cao hơn rõ rệt. Triển khai liên tục trong phiên, không dừng giữa các bước để xin duyệt thao tác có thể đảo ngược. Căn cứ gốc: `../Project/CS406_Yeu_cau_do_an_Xu_ly_anh_va_Ung_dung.docx` tính từ thư mục repo (đường dẫn tuyệt đối workspace được ghi trong ledger). Không hứa điểm chấm hoặc hiệu năng mô hình chưa đo.

## Hiện trạng được kiểm tra

- Repo nguồn `cs406-segmentation-lab`, gốc nhánh `b264cd6`; bản đầy đủ và dữ liệu ở thư mục anh em `Project`.
- Python 3.13.7, PyTorch 2.14.1+cpu, 16 CPU logic, không có CUDA, còn khoảng 17,5 GiB đĩa khi bắt đầu.
- 2.179/157/295 ảnh train/validation/test, 2.179/157/885 nhãn. LAB–Otsu, Logistic học mới và checkpoint TransUNet đã có phép đo.
- Checkpoint TransUNet đã được phát hành tại `checkpoint-v1`, kích thước 421.301.094 byte và SHA-256 khớp manifest.
- Báo cáo và slide có sẵn; bản PDF xuất lại còn mục lục lỗi và ghi chú biên tập. Lịch sử checkpoint và phép biến đổi từ nguồn công khai sang archive cục bộ chưa được xác nhận.

## Hướng thực hiện

Chốt nguồn/kiểm kê và rubric, sau đó xây dựng thực nghiệm mới có toàn bộ lịch sử do đồ án kiểm soát: Compact U-Net huấn luyện từ khởi tạo ngẫu nhiên, train/validation tách biệt, không dùng checkpoint cung cấp. Dùng loss BCE và BCE+Dice, augmentation bật/tắt để tạo bốn cấu hình ablation tại seed 406; lặp cấu hình BCE+Dice+augmentation tại seed 407 và 408. Các cấu hình được định trước trước khi đọc kết quả test. Có so sánh với các phương pháp hiện có, đánh giá độ bất định, ca sai và stress test ngoài nguồn.

Các lựa chọn đã xét: chỉ chỉnh tài liệu không xử lý khoảng trống thực nghiệm; fine-tune toàn bộ TransUNet 105M tham số trên CPU cần nguồn lực lớn và không giải quyết lịch sử checkpoint; mô hình gọn từ đầu tạo được thực nghiệm có kiểm soát trên phần cứng hiện có. Chọn phương án mô hình gọn, giữ TransUNet như kết quả tham khảo với giới hạn nguồn gốc rõ ràng. Nếu GPU được người dùng cung cấp, có thể bổ sung fine-tuning chứ không thay đổi thí nghiệm chính.

## Quy tắc thực nghiệm

- Không sửa archive/checkpoint gốc; không tạo nhãn y khoa hoặc thông tin tác giả/thành viên giả.
- Huấn luyện và lựa chọn checkpoint/ngưỡng chỉ đọc train/validation. Test được đọc sau khi checkpoint và cấu hình lựa chọn đã được ghi thành file.
- Test cục bộ đã từng được quan sát ở giai đoạn trước; không gọi là nghiên cứu mù hoặc chưa ai nhìn thấy. Logistic đã có bằng chứng train có kiểm soát; mô hình gọn bổ sung bằng chứng mạnh hơn, không xóa bỏ thí nghiệm đó.
- Giữ split được cung cấp. Kiểm tra byte trùng, mã encounter và near-duplicate ứng viên; báo điều kiện hạn chế, không tự coi encounter là bệnh nhân.
- Mô hình gọn dùng ảnh 128×128, mask nhị phân nearest; so sánh test tại 256×256 với cùng nhãn đa số nghiêm ngặt. Demo trả kích thước gốc và giải thích khác biệt giao thức.
- Kiến trúc Compact U-Net: ba lần giảm mẫu, kênh 12/24/48/96, hai Conv3×3+GroupNorm+SiLU mỗi block, decoder nối skip và nội suy bilinear, một logit đầu ra.
- Mỗi cấu hình train đầy đủ dữ liệu, tối đa 20 epoch, batch 16, AdamW lr 0,001, weight decay 0,0001, seed khai báo, số luồng CPU 4. Đo tốc độ thật trước khi chốt lịch chạy; mọi điều chỉnh do tài nguyên phải ghi trước khi tiếp tục so sánh.
- Lựa chọn checkpoint: macro Dice validation tại ngưỡng 0,5, lấy epoch sớm nhất khi hòa. Sau train chọn ngưỡng trên lưới 0,30..0,70 bước 0,05 bằng validation. Đóng băng manifest trước test.
- BCE+Dice dùng tổng BCEWithLogits và soft Dice loss. Augmentation đồng bộ ảnh/nhãn gồm lật ngang/dọc và xoay góc vuông; thay đổi sáng/tương phản chỉ cho ảnh. Validation không augmentation.
- Báo macro Dice/IoU/Precision/Recall/Pixel Accuracy, micro theo tổng TP/FP/FN/TN, kết quả từng ảnh/annotation, thời gian và bộ nhớ/param phù hợp. Bootstrap cặp theo nhóm filename 2.000 lần; ba seed báo trung bình/độ lệch chuẩn riêng với CI theo ảnh.
- Stress test dùng tập ngoài nguồn có ảnh và mask, quyền dùng/citation xác minh; công bố khác biệt ảnh/semantics và không coi đó là xác nhận lâm sàng.

## Phần mềm và tài liệu

- Giữ backend FastAPI và frontend HTML/CSS/JS; thêm mô hình mới, tình trạng tài nguyên, kết quả/đường dẫn download có thật và khả năng chạy lại trên máy mới.
- Đường dẫn dữ liệu/kết quả có thể cấu hình bằng biến môi trường để không phải nhân bản dataset trong repo.
- Báo cáo Word/PDF đủ chín chương, hình/bảng đúng số và nguồn, mục lục cập nhật, không còn ghi chú nội bộ; bìa dùng thông tin đã xác nhận.
- Slide và kịch bản bảo vệ tập trung bài toán → nguồn/nhãn → phương pháp → kiểm soát thí nghiệm → kết quả/ablation → ca sai → demo → giới hạn.
- Bộ nộp gồm source, hướng dẫn, requirements, manifest và liên kết dataset/checkpoint/artefact hợp lệ. Dữ liệu ảnh không tự đăng lại lên repo.
- Ma trận rubric phân biệt yêu cầu bắt buộc, mở rộng tự chọn và hạng mục người dùng/tác giả phải xác nhận. Tài liệu hướng dẫn hiểu mô hình hỗ trợ phần bảo vệ; không tự xác nhận thay năng lực trả lời của thành viên.

## Tiêu chí nghiệm thu

Mỗi điều khoản DOCX có bằng chứng và kết quả kiểm tra; thí nghiệm mới có đầy đủ train/valid/history/checkpoint/frozen test protocol và không dùng nhãn test để chọn cấu hình; ablation và chạy nhiều seed là kết quả thật; stress test ghi đúng phạm vi. Demo có kiểm thử API/giao diện; báo cáo/slide được render và rà hình; source package được thử từ bản sạch. Mục nào phụ thuộc xác nhận bên ngoài phải hiện rõ thay vì đánh dấu đạt giả.
