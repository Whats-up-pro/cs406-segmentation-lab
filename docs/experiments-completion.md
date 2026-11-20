# Thực nghiệm mở rộng CS406

Giao thức này bổ sung một mô hình phân đoạn học từ đầu có lịch sử kiểm tra được, ablation loss và augmentation, nhiều seed và stress test ngoài nguồn. Sáu kết quả Compact U-Net chỉ được báo cáo sau khi có đủ lịch sử 20 epoch và manifest đóng băng. Mã chạy và kiểm thử không thay thế bằng chứng đã huấn luyện.

## Câu hỏi và đơn vị so sánh

Thí nghiệm chính kiểm tra khả năng học vùng foreground từ dữ liệu train hiện có bằng một mạng nhỏ. Hai yếu tố ablation là có thêm soft Dice trong loss và có tăng cường ảnh hay không. So sánh bốn tổ hợp ở cùng seed 406 cho biết tác động quan sát được của từng thay đổi dưới cùng ngân sách 20 epoch. Ba lần chạy tổ hợp đầy đủ ở seed 406, 407, 408 mô tả biến thiên do khởi tạo, thứ tự dữ liệu và augmentation.

Recipe chính `bce_dice_aug_s406` được khai báo trước khi tính điểm mới trên test. Không chọn seed tốt nhất theo test. Các kết quả TransUNet được cung cấp thuộc nhóm tham khảo vì lịch sử huấn luyện chưa được xác minh. Sự khác nhau giữa hai mạng không chứng minh tác dụng riêng của Transformer, kiến trúc hay pretraining.

Test cục bộ đã được xem trong giai đoạn phân tích trước đó. Việc chỉ dùng train/valid để tối ưu và chọn mô hình ngăn bước huấn luyện mới đọc test, nhưng không biến benchmark hồi cứu thành một nghiên cứu mù với holdout chưa từng được quan sát.

## Kiến trúc Compact U-Net

Mạng dùng ý tưởng encoder–decoder với skip connection của [U-Net](https://arxiv.org/abs/1505.04597). Đây là một triển khai gọn cho thí nghiệm có kiểm soát, không phải kiến trúc mới hoặc bản tái lập nguyên bản U-Net 2015.

| Mức | Kích thước đặc trưng với ảnh 128 × 128 | Số kênh |
| --- | --- | ---: |
| Encoder 1 | 128 × 128 | 12 |
| Encoder 2 | 64 × 64 | 24 |
| Encoder 3 | 32 × 32 | 48 |
| Bottleneck | 16 × 16 | 96 |
| Decoder 3 | 32 × 32 | 48 |
| Decoder 2 | 64 × 64 | 24 |
| Decoder 1 | 128 × 128 | 12 |
| Head 1 × 1 | 128 × 128 | 1 logit |

Mỗi block gồm hai convolution 3 × 3 không bias, mỗi convolution nối tiếp GroupNorm và SiLU. Max pooling 2 × 2 giảm kích thước; decoder nội suy bilinear về đúng kích thước skip, ghép kênh rồi áp dụng block convolution. Số nhóm chuẩn hóa là `gcd(4, channels)`. [Group Normalization](https://arxiv.org/abs/1803.08494) tính thống kê trong nhóm kênh và không phụ thuộc thống kê giữa các phần tử batch. Thí nghiệm này chưa tách riêng tác động của GroupNorm.

Với 12 kênh cơ sở, mạng có **274.849 tham số**. Trọng số được khởi tạo từ seed, không lấy tham số từ TransUNet. Kích thước checkpoint cụ thể đọc từ tệp được huấn luyện; không suy ra tốc độ chạy từ số tham số.

## Tiền xử lý và loss

Chuỗi RGB phiên bản `rgb-bilinear-256-then-model-v2` dùng PIL bilinear về 256 × 256 rồi về 128 × 128 và chia 255. Cùng chuỗi được dùng khi tạo cache train, valid và khi dự đoán. Với đầu vào đã ở 256 × 256, bước đầu không đổi pixel. Mask được nhị phân hóa theo giá trị khác 0 và nội suy nearest neighbor. Mask validation 256 × 256 được lưu riêng cho bước chọn ngưỡng cuối.

Loss là `BCEWithLogitsLoss + λ × (1 − soft Dice)`, với λ bằng 0 hoặc 1. Soft Dice tính riêng cho mỗi ảnh, với tử số `2Σ(p·y) + 10⁻⁶` và mẫu số `Σp + Σy + 10⁻⁶`, sau đó lấy trung bình batch. Đây là biến thể triển khai của mục tiêu chồng lấp khả vi; [V-Net](https://arxiv.org/abs/1606.04797) là tài liệu tham khảo về tối ưu mục tiêu dựa trên Dice, không phải nguồn điểm số so sánh cục bộ. Loss liên tục khác Dice nhị phân sau chọn ngưỡng.

## Ngân sách và đóng băng lựa chọn

| Tham số | Giá trị khai báo |
| --- | --- |
| Số epoch mỗi run | 20 |
| Batch size | 16; batch cuối được giữ |
| Optimizer | AdamW |
| Learning rate | 0,001 cố định |
| Weight decay | 0,0001 |
| Gradient clipping | L2 norm tối đa 1 |
| Cập nhật tham số | Chỉ train, đủ 2.179 ảnh mỗi epoch |
| Chọn checkpoint | Dice valid tại ngưỡng 0,5 cao nhất; hòa chọn epoch sớm nhất |
| Chọn ngưỡng | Valid 157 ảnh, tại 256 × 256; lưới 0,30–0,70 bước 0,05; hòa chọn ngưỡng nhỏ nhất |

Augmentation chỉ áp dụng train: lật hai chiều độc lập với xác suất 0,5, quay một trong bốn bội số 90 độ với xác suất đều nhau. Hình học được áp dụng đồng bộ cho ảnh và mask. Hệ số brightness và contrast độc lập, phân bố đều trong [0,85; 1,15], chỉ áp dụng ảnh. Không augmentation validation hoặc test. Bộ sinh ngẫu nhiên gắn với seed, epoch và index mẫu.

Cache chỉ khám phá thư mục train/valid; ghi SHA-256 của từng ảnh và mask, fingerprint của danh sách nguồn, kích thước và recipe tiền xử lý. Resume phải khớp cấu hình, nguồn và dữ liệu, đồng thời phục hồi optimizer và lịch sử; một kiểm thử nhỏ xác nhận trọng số CPU sau gián đoạn khớp run không gián đoạn. CPU bật deterministic algorithms nghiêm ngặt. CUDA ghi nhận chế độ cảnh báo do bilinear backward có thể không hỗ trợ kernel deterministic; không hứa kết quả GPU trùng từng bit.

`config.json`, `history.json`, `best.pth` và `frozen-selection.json` lưu số ảnh đã xử lý, môi trường, seed, epoch được chọn, lưới ngưỡng và hash. Evaluator kiểm tra chúng trước khi khám phá test. Cache v1 và checkpoint pilot cũ được giữ như artefact lịch sử, không trộn vào run cuối v2.

## Giao thức đánh giá

Mọi phương pháp dùng cùng ảnh RGB 256 × 256. Compact đổi tiếp về 128; TransUNet về 224. Bản đồ điểm được khôi phục về 256 để tính metric. Mỗi annotation được nhị phân hóa và resize nearest neighbor trước khi tạo consensus bằng đa số nghiêm ngặt; hòa là nền. Quy tắc này khác evaluator chính thức MEDIQA-MAGIC ở trường hợp số annotation chẵn, nên không đối chiếu điểm cục bộ như thể cùng bài thi chính thức.

Đầu ra gồm Dice, IoU, Precision, Recall, Pixel Accuracy và boundary F1. Boundary là mask trừ erosion 3 × 3, ngoài ảnh là nền; hai biên khớp khi khoảng cách Euclid không quá 2 pixel. Macro cân bằng ảnh, micro gộp TP/FP/FN/TN; bảng theo annotation được giữ riêng. Hai mask cùng rỗng cho điểm 1; chỉ một mask rỗng cho các điểm foreground bằng 0; Pixel Accuracy vẫn đo tỷ lệ khớp thực tế.

Phân tích ghép cặp bắt buộc cùng ID ảnh ở hai phương pháp. Khoảng 95% dùng 2.000 lần bootstrap theo encounter, giữ toàn bộ ảnh trong nhóm được lấy lại và tính trung bình có trọng số đều theo ảnh. Mẫu không có encounter là một nhóm riêng. Đây là khoảng mô tả dưới giả định lấy mẫu cụm; tên encounter chưa xác nhận định danh bệnh nhân. Nhiều so sánh mang tính thăm dò, không có hiệu chỉnh familywise và không tuyên bố kiểm định xác nhận.

Thời gian trong CSV là thời gian gọi predictor; lần đầu bao gồm lazy load, không có warm-up chuyên biệt. Median, p95 và thời gian các lần tiếp theo được mô tả với cùng giới hạn đó. Không so tốc độ GPU với CPU như một ablation thuật toán.

Một số epoch của run CPU kéo dài qua thời gian máy tạm ngưng, nên `seconds` trong lịch sử bao gồm quãng gián đoạn hệ thống. Không dùng tổng thời lượng các epoch này để kết luận tốc độ huấn luyện hoặc so sánh phần cứng. Việc gián đoạn và di chuyển thư mục resume được ghi riêng, không xóa hoặc sửa thời gian đã lưu.

## Stress test ngoài nguồn

Bộ ISIC 2018 Task 1 validation gồm 100 cặp dermoscopy/mask được tải từ nguồn chính thức và kiểm tra hash; manifest giữ vai trò nguồn là validation của ISIC, vai trò thí nghiệm là external stress test. Tham số và ngưỡng của mô hình được giữ nguyên. Ảnh dermoscopy, ranh giới tổn thương và ảnh chụp vùng quan tâm của bộ cục bộ khác nhau về modality, nội dung và quy trình chú thích. Điểm thấp hoặc cao ở đây chưa xác nhận chất lượng lâm sàng hay khả năng tổng quát trên mọi ảnh da.

## Lệnh và bằng chứng

Lệnh chạy CPU/CUDA, gói Kaggle riêng tư và cấu trúc đầu ra được mô tả tại [README](../README.md) và [Kaggle](kaggle.md). Dữ liệu, checkpoint và kết quả lớn được lưu ngoài Git. Chỉ dùng bảng/đồ thị xuất từ CSV có hash khớp summary, không điền số dự kiến vào báo cáo. Khi đủ kết quả, cập nhật [tiến độ](completion-status.md) và ma trận yêu cầu bằng bằng chứng cuối.
