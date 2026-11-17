# Nguồn dữ liệu, kết quả cuộc thi và checkpoint

Đối chiếu nguồn ngày 17/11/2026 sau khi người cung cấp đồ án bổ sung liên kết DermaVQA và hai lần nộp của đội Anastasia.

## DermaVQA và phiên bản segmentation năm 2025

[velvinnn/DermaVQA](https://github.com/velvinnn/DermaVQA) giới thiệu DermaVQA trong công trình MICCAI 2024, với ảnh và câu hỏi da liễu từ IIYI và Reddit. README liên kết hai cuộc thi năm 2024 và bộ dữ liệu trên OSF. Đây là nguồn tham khảo về dòng dữ liệu DermaVQA.

[Trang MEDIQA-MAGIC 2025](https://www.imageclef.org/2025/medical/mediqa) gọi dữ liệu cho các tác vụ segmentation và QA năm 2025 là **DermaVQA-DAS**. Tác vụ segmentation xác định vùng vấn đề da liễu từ ảnh và lịch sử lâm sàng. Dữ liệu và cách chấm được mô tả tại [kho chính thức năm 2025](https://github.com/wyim/ImageCLEF-MAGIC-2025); truy cập bộ tài nguyên cuộc thi qua [nền tảng được ban tổ chức liên kết](https://ai4media-bench.aimultimedialab.ro/competitions/62/).

| Tập | Ảnh công bố 2025 | Mask công bố 2025 | Ảnh trong archive đồ án | Mask trong archive đồ án |
| --- | ---: | ---: | ---: | ---: |
| Train | 2.474 | 7.448 | 2.179 | 2.179 |
| Validation | 157 | 472 | 157 | 157 |
| Test | 314 | 944 | 295 | 885 |

Cấu trúc tên ảnh `IMG_ENC...` và hậu tố annotation tương thích với tài nguyên 2025. Các số lượng khác nhau đòi hỏi bản kê chọn ảnh và quy tắc biến đổi nhãn để xác nhận chính xác phiên bản cục bộ. Thông tin bổ sung hiện chưa giải thích việc train/validation chỉ giữ một mask mỗi ảnh hoặc các mẫu bị loại. Thống kê archive do lần kiểm kê đồ án ghi nhận; repo code không phân phối dữ liệu ảnh.

## Bảng Segmentation Task Results

[Bảng kết quả của ban tổ chức](https://www.imageclef.org/2025/medical/mediqa) ghi các lần nộp đứng đầu cùng điểm số của đội **Anastasia**:

| Owner | Team | Experiment directory | Jaccard | Dice |
| --- | --- | --- | ---: | ---: |
| khoingotuantu | Anastasia | Test/1491_khoingotuantu_transunet 5 model | 0.645774234 | 0.784766 |
| khoingotuantu | Anastasia | Test/1490_khoingotuantu_transunet 6 new model | 0.645774234 | 0.784766 |
| zhentu | Anastasia | Test/1390_zhentu_Transunet | 0.645774234 | 0.784766 |

Theo trang cuộc thi, phép chấm dùng đa số theo pixel làm nhãn chuẩn cho micro Jaccard/Dice và báo thêm các thống kê theo annotation. Đồ án hiện lấy macro trung bình theo ảnh trên 295 ảnh đã resize về 256×256. Điểm cuộc thi và điểm đồ án cần được trình bày cùng giao thức tương ứng.

Tên các lần nộp là bằng chứng về đội và hệ thống được liệt kê. Để gắn checkpoint đơn lẻ đang có với một lần nộp cụ thể cần thêm hash trọng số, cấu hình ensemble và log/manifest từ tác giả. Chưa có bằng chứng đó trong bộ tài nguyên hiện tại.

## Checkpoint được phân phối cùng repo

| Thuộc tính | Giá trị |
| --- | --- |
| File | `TransUNet_R50_ViT_B_16_all_best_model.pth` |
| Kích thước | 421.301.094 byte |
| SHA-256 | `6df81c63fab9936d8d90e79bfea0264c11516e82d981c70ab62c89758793a508` |
| Kiến trúc suy luận | TransUNet R50-ViT-B/16, một kênh đầu ra |
| Cách phân phối | Release asset tại tag `checkpoint-v1` |
| Đường dẫn sau khi tải | `data/TransUNet_R50_ViT_B_16_all_best_model.pth` |

Xem [manifest máy đọc được](../models/transunet-checkpoint.json) và [hướng dẫn tải](../README.md#chuẩn-bị-dữ-liệu-và-mô-hình). Chủ repo phát hành file bằng script `Upload-CS406.ps1` trong workspace đồ án; link tải chỉ có hiệu lực sau khi asset được upload thành công. File có thể nạp khớp 409 tensor với kiến trúc đã triển khai. Nguồn huấn luyện, chuẩn hóa gốc và việc checkpoint đã từng thấy tập test nào vẫn cần tác giả xác nhận. Checkpoint được cung cấp sẵn; đồ án chưa fine-tune đầy đủ TransUNet.

Giấy phép Apache-2.0 trong `third_party/TransUNet` áp dụng cho mã kiến trúc được sử dụng lại. Điều kiện sử dụng dataset và trọng số là các thông tin riêng từ người cung cấp tài nguyên.
