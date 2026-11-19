# Nguồn dữ liệu, kết quả cuộc thi và checkpoint

Đối chiếu cập nhật ngày 19/11/2026 từ liên kết người dùng cung cấp và danh mục ảnh của ban tổ chức.

## DermaVQA và phiên bản segmentation năm 2025

[velvinnn/DermaVQA](https://github.com/velvinnn/DermaVQA) giới thiệu DermaVQA trong công trình MICCAI 2024, với ảnh và câu hỏi da liễu từ IIYI và Reddit. README liên kết hai cuộc thi năm 2024 và bộ dữ liệu trên OSF. Đây là nguồn tham khảo về dòng dữ liệu DermaVQA.

[Trang MEDIQA-MAGIC 2025](https://www.imageclef.org/2025/medical/mediqa) gọi dữ liệu cho các tác vụ segmentation và QA năm 2025 là **DermaVQA-DAS**. Tác vụ segmentation xác định vùng vấn đề da liễu từ ảnh và lịch sử lâm sàng. Dữ liệu và cách chấm được mô tả tại [kho chính thức năm 2025](https://github.com/wyim/ImageCLEF-MAGIC-2025); truy cập bộ tài nguyên cuộc thi qua [nền tảng được ban tổ chức liên kết](https://ai4media-bench.aimultimedialab.ro/competitions/62/).

| Tập | Ảnh công bố 2025 | Mask công bố 2025 | Ảnh trong archive đồ án | Mask trong archive đồ án |
| --- | ---: | ---: | ---: | ---: |
| Train | 2.474 | 7.448 | 2.179 | 2.179 |
| Validation | 157 | 472 | 157 | 157 |
| Test | 314 | 944 | 295 | 885 |

Đã đối chiếu danh mục ZIP ảnh gốc, metadata query và ánh xạ encounter ở revision cố định: **train + test cục bộ bằng đúng toàn bộ 2.474 ID ảnh train nguồn**, validation cục bộ bằng đúng 157 ID validation nguồn. Không có ID thuộc 314 ảnh test chính thức trong dữ liệu cục bộ. Cả 2.631 tệp ảnh khớp kích thước và CRC32 của thành viên ZIP nguồn; CRC32 không thay thế kiểm chứng SHA-256 mật mã cho từng ảnh. Vì vậy, 295 ảnh test là cách chia nội bộ từ train nguồn, không phải test của leaderboard. [Bằng chứng, URL và giới hạn](data-provenance-verified.md#confirmed-source-split-membership-of-the-local-test-set) mô tả phép ghép và bản kê lưu ngoài Git.

Chưa xác minh cách nhiều mask gốc được biến đổi thành một mask train/validation cục bộ. Giấy phép kho DermaVQA 2024 khai báo CC BY 4.0; điều đó chưa xác định mọi điều kiện của mask segmentation năm 2025 hoặc checkpoint. Repo code không phân phối dữ liệu ảnh.

## Bảng Segmentation Task Results

[Bảng kết quả của ban tổ chức](https://www.imageclef.org/2025/medical/mediqa) ghi các lần nộp đứng đầu cùng điểm số của đội **Anastasia**:

| Owner | Team | Experiment directory | Jaccard | Dice |
| --- | --- | --- | ---: | ---: |
| khoingotuantu | Anastasia | Test/1491_khoingotuantu_transunet 5 model | 0.645774234 | 0.784766 |
| khoingotuantu | Anastasia | Test/1490_khoingotuantu_transunet 6 new model | 0.645774234 | 0.784766 |
| zhentu | Anastasia | Test/1390_zhentu_Transunet | 0.645774234 | 0.784766 |

Theo mã chấm của ban tổ chức, đa số với số phiếu hòa nhận foreground; đồ án dùng đa số nghiêm ngặt, hòa nhận nền. Đồ án lấy macro theo ảnh trên 295 ảnh chia lại từ train nguồn và resize về 256×256. Tập ảnh, nhãn tham chiếu và đơn vị tổng hợp khác bài thi chính thức; không so hai điểm như một leaderboard chung.

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

Xem [manifest máy đọc được](../models/transunet-checkpoint.json) và [hướng dẫn tải](../README.md#chuẩn-bị-dữ-liệu-và-mô-hình). Asset đã phát hành tại [checkpoint-v1](https://github.com/Whats-up-pro/cs406-segmentation-lab/releases/tag/checkpoint-v1); kích thước và SHA-256 phía GitHub khớp tệp gốc. File có thể nạp khớp 409 tensor với kiến trúc đã triển khai. Nguồn huấn luyện, chuẩn hóa gốc và test exposure vẫn chưa được tác giả xác nhận. Việc test cục bộ lấy từ train nguồn làm giới hạn này đặc biệt quan trọng, nhưng không chứng minh checkpoint thực sự đã học các ảnh đó. Compact U-Net học từ đầu là thí nghiệm riêng có lịch sử kiểm tra được.

Giấy phép Apache-2.0 trong `third_party/TransUNet` áp dụng cho mã kiến trúc được sử dụng lại. Điều kiện sử dụng dataset và trọng số là các thông tin riêng từ người cung cấp tài nguyên.
