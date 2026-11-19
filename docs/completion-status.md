# Tiến độ hoàn thiện đồ án CS406

Cập nhật nền ngày 19/11/2026. Căn cứ là `Project/CS406_Yeu_cau_do_an_Xu_ly_anh_va_Ung_dung.docx` trong workspace `D:/HK1_2026-2027/CS406`; kho mã chính là `cs406-segmentation-lab`, bản dữ liệu và kết quả đầy đủ nằm ở thư mục anh em `Project`.

Ma trận [course-requirements.json](../config/course-requirements.json) ánh xạ từng điều khoản của cả 18 mục DOCX thành yêu cầu có bằng chứng và tiêu chí nghiệm thu. Trạng thái này không phải điểm số tự động, không bảo đảm giảng viên chấm đạt, và không thay thế kiểm tra nội dung, bố cục hay hiểu biết của thành viên. Rubric gợi ý trong DOCX dành 30% cho báo cáo, 20% cho phương pháp và mức độ hiểu; leakage, sao chép, không hiểu mô hình và kết quả không tái lập có thể được xử lý riêng.

## Bằng chứng đã có

- Bài toán RGB → mặt nạ nhị phân, phương pháp LAB–Otsu, Logistic pixel và checkpoint TransUNet được cung cấp; predictor và nhãn tham chiếu có vai trò riêng.
- Kiểm kê 2.179/157/295 ảnh và 2.179/157/885 mask; EDA đã giải mã đủ 2.631 ảnh và 3.221 mask. Mã encounter suy từ filename không được coi là bệnh nhân đã xác minh.
- Logistic **đã huấn luyện đầy đủ hai lượt trên train**, seed 406, 8.925.184 lượt pixel; validation chọn ngưỡng 0,30. Manifest, model và lựa chọn được lưu trước khi script khám phá test. Test Dice 0,341623 thấp hơn LAB 0,443523; đây là một thực nghiệm có kiểm soát hoàn tất, không phải kết quả giả hay fine-tuning TransUNet.
- Checkpoint TransUNet có 105.275.921 tham số, 409 tensor và manifest SHA-256. Test Dice 0,745545 là kết quả tham khảo trên dữ liệu cung cấp; chưa biết dữ liệu huấn luyện gốc hoặc test exposure của checkpoint.
- Có so sánh đủ test, Dice/IoU/Accuracy/Precision/Recall, CSV từng ảnh/annotation, phân tích lỗi, bootstrap theo nhóm filename và độ nhạy ảnh trùng chính xác. Các khoảng bootstrap mô tả dữ liệu hiện có; chúng không chứng minh patient independence hoặc nguồn gốc checkpoint.
- Có ứng dụng dự đoán thật, báo cáo chín phần, slide và hướng dẫn bảo vệ. Các tài liệu cũ vẫn cần cập nhật và kiểm tra với kết quả cuối cùng.

**Test ban đầu đã được xem để phân tích lỗi.** Thực nghiệm tiếp theo phải chọn recipe/checkpoint/ngưỡng bằng train/validation, đóng băng lựa chọn trước đánh giá test, đồng thời nói rõ việc nhóm đã quan sát test trước đó. Quy tắc này bảo vệ việc không dùng test để cập nhật tham số; nó không khôi phục một tập test chưa từng được người nghiên cứu quan sát.

## Việc có thể triển khai bằng mã và artifact

| Hạng mục | Tiêu chí nghiệm thu | Trạng thái nền |
| --- | --- | --- |
| Kiểm tra leakage | Hash, mã nhóm và ứng viên gần trùng xuyên tập, kèm giao thức/ngưỡng và rà soát ứng viên | `results/completion/leakage/summary.json`: giải mã 2.631 ảnh, so sánh 1.031.223 cặp xuyên tập, 0 nhóm byte trùng và 0 ứng viên theo ngưỡng đã khai báo; không chứng minh patient independence |
| Compact U-Net từ đầu | Train-only update, valid-only selection, đầy đủ số ảnh/lịch sử/checkpoint/hash và frozen selection | Đang triển khai; không đánh dấu complete khi chỉ có script hoặc smoke test |
| Ablation và nhiều seed | Loss × augmentation định trước, cùng ngân sách; chạy thật các seed đã khai báo | Planned; kết quả phải do manifest hoàn tất cung cấp |
| Stress test ngoài nguồn | Nguồn/quyền dùng/ảnh/mask kiểm chứng được, cấu hình đóng băng, giới hạn modality/semantics | Planned; chạy ảnh mới được không tự chứng minh generalization |
| Demo cuối | API/browser kiểm tra predictor mới, upload không nhãn, PNG đúng kích thước và lỗi tài nguyên | Cần bản ghi kiểm chứng của revision cuối |
| Báo cáo và slide cuối | Số liệu khớp CSV, render từng trang/slide, TOC/caption/reference đúng, không còn ghi chú biên tập | Cần tái xuất và rà bản cuối |
| Bộ nộp chạy lại | Source sạch, requirements, model manifest, tài nguyên lấy được và lệnh chạy đã thử | Cần kiểm tra package và liên kết thật |

DOCX khuyến khích ablation và đánh giá ngoài nguồn; chúng là phần mở rộng để nâng chất lượng, không phải điều kiện bắt buộc được tự thêm vào rubric. Không bắt buộc huấn luyện toàn bộ TransUNet hoặc đạt một Dice tối thiểu. Kết quả không cao nhưng được thiết kế và phân tích khoa học vẫn có giá trị.

## Thông tin không được điền bằng suy đoán

| Nội dung | Người có thể xác nhận | Tình trạng |
| --- | --- | --- |
| Thành viên và MSSV | Sinh viên/nhóm | Đủ bốn thành viên: Phạm Tấn Gia Quốc 23521308, Huỳnh Thái Bảo 23520105, Trần Minh Tiến 23521587, Lâm Quang Huy 23520609; không còn placeholder. Thông tin ở `config/team.json` và xác nhận theo hội thoại ở `Project/docs/confirmations/team.json` |
| Phân công và đóng góp thực tế | Sinh viên/nhóm | Chưa xác nhận; không chuyển vai trò gợi ý trong kịch bản thành lịch sử công việc đã diễn ra |
| Lớp, giảng viên hướng dẫn và các trường hành chính trên bìa | Sinh viên/đơn vị học phần | Chưa xác nhận; không phát minh để lấp chỗ trống |
| Phiên bản dataset, ảnh bị lọc và quan hệ với tài nguyên công khai | Người cung cấp dữ liệu | Nguồn DermaVQA-DAS/MEDIQA-MAGIC có dấu hiệu phù hợp nhưng số lượng khác; cần bản kê biến đổi |
| Ý nghĩa foreground, quy trình gán nhãn và cách hợp nhất nhãn train/valid | Chủ dữ liệu/tác giả annotation | Chưa được tài liệu hóa đầy đủ |
| Quyền lưu trữ, minh họa và phân phối ảnh/trọng số | Chủ quyền tài nguyên | License mã kiến trúc không thay thế license dữ liệu hoặc checkpoint |
| Dữ liệu, preprocessing và lịch sử checkpoint TransUNet | Tác giả checkpoint | Chưa xác minh; không cản trở mô tả nó là kết quả tham khảo khi thí nghiệm chính được kiểm soát độc lập |
| Mọi thành viên hiểu và trả lời được | Chính từng thành viên qua tập bảo vệ | Kịch bản/câu hỏi chỉ hỗ trợ chuẩn bị, không xác nhận thay năng lực |

Những xác nhận bắt buộc được lưu thành JSON có `confirmed: true`, `confirmed_by`, `confirmed_at` và `scope`, cộng các trường kiểm tra riêng trong ma trận. Đây là bản ghi xác nhận có người chịu trách nhiệm, không phải chữ ký được hệ thống tự xác thực. Chỉ ghi `confirmed` khi có phát biểu tương ứng của người dùng hoặc tác giả tài nguyên, giữ đúng phạm vi xác nhận. Thông tin lớp/GVHD là công việc hành chính; không phải điều khoản riêng được DOCX quy định. Metadata nhóm ở `config/team.json` đã có `team_size: 4`, `placeholder_count: 0`. Danh tính đầy đủ không xác nhận thay đóng góp thực tế hoặc kết quả tập bảo vệ.

## Chạy kiểm tra bằng chứng

Từ thư mục repo:

```powershell
..\Project\.venv\Scripts\python.exe -m scripts.verify_submission --repo-dir . --project-dir ../Project
..\Project\.venv\Scripts\python.exe -m scripts.verify_submission --repo-dir . --project-dir ../Project --json
..\Project\.venv\Scripts\python.exe -m scripts.verify_submission --repo-dir . --project-dir ../Project --claim-ready
```

Trên môi trường đã cài riêng, thay đường dẫn interpreter bằng `python`. Có thể thêm `--output <audit.json>` để lưu JSON; `--package <source.zip>` rà ZIP **mã nguồn**, loại raw dataset, môi trường riêng, thông tin riêng và binary trọng số. Dataset/model release được quản lý riêng, không kiểm bằng quy tắc source ZIP này.

Lệnh mặc định mô tả cả phần đạt, thiếu và pending, exit 0 để có thể tiếp tục công việc. `--claim-ready` trả exit 1 khi còn mandatory evidence thiếu/sai hoặc human confirmation pending; cấu hình không hợp lệ trả exit 2. Optional chưa xong vẫn được liệt kê nhưng không biến thành yêu cầu mandatory. `--json` xuất kết quả máy đọc được; thiếu file, JSON status sai, checkpoint size/hash sai và payload source bị cấm không được coi là hoàn tất. Các đường dẫn bằng chứng phải nằm trong root khai báo.

Tên `submission_ready` trong JSON có nghĩa là mọi kiểm tra bằng chứng bắt buộc đã qua theo ledger này. Nó không chứng nhận điểm số, tính đúng của mọi câu văn, nội dung nhãn, chất lượng hình đã render, link tải trên mạng hoặc hiểu biết con người. Các hạng mục đó cần bằng chứng rà soát/xác nhận riêng. Thay `state` từ pending/planned sang evidence_available/confirmed chỉ sau khi tiêu chí nghiệm thu thực sự được đáp ứng.

## Nhật ký quyết định và kiểm chứng

- Ruling: chỉ triển khai bốn file được phân công cho Task 1 — tránh ghi đè phần Compact U-Net, leakage audit và artifact đang được triển khai song song; root agent quản lý ledger chung, git index và commit.
- Ruling: không chạy toàn bộ suite trong lúc nguồn khác đang được chỉnh sửa song song — focused tests chứng minh verifier; full suite của revision tích hợp do root agent điều phối trước kết luận cuối.
- TDD ngày 19/11/2026: `tests/test_submission.py` được chạy trước implementation, 14 ca thất bại vì verifier chưa có; sau implementation, cùng lệnh `python -m pytest tests/test_submission.py -q -p no:cacheprovider` cho **14 passed**. Chưa dùng kết quả này để kết luận toàn bộ yêu cầu đồ án hoàn tất.
- Kiểm tra thực tế phát hiện stdout mặc định cp1252 của Python trên Windows không in được tiếng Việt; regression test tái hiện exit 2 sai trước khi sửa đầu ra UTF-8. Sau sửa, focused suite cho **15 passed**; audit thực tế kiểm đủ 58 mục nguyên tử/18 điều khoản, không tự chấm điểm, và `--claim-ready` trả exit 1 vì còn bằng chứng/xác nhận pending.
- Đã đọc báo cáo leakage thật: toàn bộ ảnh giải mã được, không có ứng viên xuyên tập ở cặp ngưỡng pHash≤6 và dHash≤8; mục audit kỹ thuật chuyển sang evidence_available. Ba mục kỹ thuật cuối (demo, render, package) và năm nhóm xác nhận con người vẫn pending; không diễn giải audit sạch thành không có mọi dạng near-duplicate hoặc lịch sử checkpoint đã được xác minh.
