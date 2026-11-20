# Nguồn và cách sử dụng nhãn

Đồ án dùng ảnh RGB và các mask TIFF được cung cấp cùng archive tài nguyên. Nhóm không thực hiện một đợt vẽ nhãn mới trong thực nghiệm này. Nguồn ảnh và cách chia lại tập được xác minh tại [bằng chứng provenance](data-provenance-verified.md); nguồn ảnh đã xác minh không tự xác minh lịch sử mask.

## Quy trình được công bố và phạm vi đối chiếu

[Bài DermaVQA-DAS, mục 2 và bảng 3](https://arxiv.org/html/2512.24340v2) mô tả segmentation của vùng vấn đề có liên quan lâm sàng, với ba mask mỗi ảnh từ một nhóm bốn **medical annotators**. Hai bác sĩ da liễu thiết kế schema DAS; thông tin này không chứng minh mọi người vẽ mask là bác sĩ da liễu. Tổng số mask công bố có chênh lệch nhỏ so với đúng ba lần số ảnh, nên sử dụng số annotation thực có thay vì ép luôn bằng ba.

Tài nguyên cục bộ có một mask/ảnh ở train và validation; test có 3 ảnh với hai mask, 289 ảnh với ba mask và 3 ảnh với bốn mask. Chưa có bản kê segmentation gốc hoặc log chuyển đổi để xác định mask train/valid được lấy từ một annotator, hợp nhất bằng majority, union hay chỉnh sửa khác. Không gọi các mask này là consensus chính thức đã được xác minh. Đồ án giữ nguyên tệp nhãn đầu vào và ghi hash để kiểm tra các bước do chính pipeline thực hiện.

## Quy tắc triển khai hiện tại

| Thành phần | Quy tắc kiểm tra được |
| --- | --- |
| Lớp | Foreground là vùng quan tâm theo nhãn được cung cấp; background là phần còn lại, không phải nhãn chẩn đoán bệnh |
| TIFF gốc | Toàn bộ nhãn kiểm kê có giá trị 0/1; không có tệp hỏng, lệch kích thước, mask rỗng hoặc toàn foreground |
| Ghép cặp | Train/valid ghép cùng stem; test đọc mọi hậu tố `_mask_annN` thực có |
| Nhị phân hóa | Pixel khác 0 thành foreground trước khi resize |
| Resize mask | Nearest neighbor; không nội suy bilinear tạo lớp trung gian |
| Huấn luyện | Mask 128 × 128 của train/valid; chỉ train cập nhật trọng số |
| Chọn ngưỡng | Mask validation gốc nhị phân hóa rồi resize 256 × 256 |
| Consensus đánh giá | Resize từng annotation về 256 rồi lấy đa số nghiêm ngặt; hòa là background |
| Theo annotator | Lưu riêng điểm của mỗi annotation, không dùng nó để tạo dự đoán |
| Demo | Dự đoán từ ảnh; nhãn chỉ dùng chấm sau suy luận; ảnh upload không nhãn không có metric |

Mã chấm chính thức dùng phiếu hòa là foreground, khác quy tắc cục bộ khi số mask chẵn. Tập ảnh, độ phân giải và macro/micro cũng khác; không coi điểm benchmark cục bộ là điểm leaderboard.

Tiền xử lý được kiểm thử bằng mask nhân tạo, ảnh không vuông, mask rỗng và nhiều annotation trong `tests/test_api.py`, `tests/test_metrics.py`, `tests/test_completion_evaluation.py`, `tests/test_compact_training.py`. EDA trên nhãn thật nằm tại `Project/results/eda`; ví dụ ảnh/nhãn và giới hạn lịch sử được trình bày trong Chương 3–4 của báo cáo. Các kiểm thử này kiểm tra quy tắc sử dụng mask, không chứng minh quy trình vẽ nhãn gốc.
