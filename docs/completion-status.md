# Tiến độ hoàn thiện đồ án CS406

Cập nhật nền ngày 19/11/2026. Căn cứ là `Project/CS406_Yeu_cau_do_an_Xu_ly_anh_va_Ung_dung.docx` trong workspace `D:/HK1_2026-2027/CS406`; kho mã chính là `cs406-segmentation-lab`, bản dữ liệu và kết quả đầy đủ nằm ở thư mục anh em `Project`.

Cập nhật ngày 20/11/2026: code tích hợp đã qua 204 kiểm thử, 2 kiểm thử tùy chọn được bỏ qua; môi trường cài mới tách system packages đã được kiểm tra, cùng smoke test checkpoint thật trên validation. Demo model Compact 20 epoch qua 12 kiểm tra trình duyệt. Bốn trong sáu run đã hoàn tất và đóng băng, hai run cuối đang chạy CPU. Báo cáo nháp riêng có 33 trang, mục lục và 54 đích điều hướng đã đối chiếu; bộ báo cáo/slide cuối và đánh giá ISIC vẫn chưa hoàn tất. Không coi bản nháp là bộ nộp cuối.

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
| Compact U-Net từ đầu | Train-only update, valid-only selection, đầy đủ số ảnh/lịch sử/checkpoint/hash và frozen selection | Recipe chính CPU hoàn tất 20 epoch, đủ 2.179/157 ảnh mỗi epoch; chọn epoch 17, ngưỡng valid 0,40. Chưa đánh giá test của mô hình cuối |
| Ablation và nhiều seed | Loss × augmentation định trước, cùng ngân sách; chạy thật các seed đã khai báo | Planned; kết quả phải do manifest hoàn tất cung cấp |
| Stress test ngoài nguồn | Nguồn/quyền dùng/ảnh/mask kiểm chứng được, cấu hình đóng băng, giới hạn modality/semantics | Đã chuẩn bị đủ 100 cặp ISIC validation CC0 và kiểm tra hash; đánh giá còn chờ đủ sáu run đóng băng |
| Demo cuối | API/browser kiểm tra predictor mới, upload không nhãn, PNG đúng kích thước và lỗi tài nguyên | 12 browser checks đạt với checkpoint 20 epoch trên validation và upload; không chấm test của mô hình cuối. Sẽ rà lại với bảng kết quả hoàn tất |
| Báo cáo và slide cuối | Số liệu khớp CSV, render từng trang/slide, TOC/caption/reference đúng, không còn ghi chú biên tập | Nháp Word/PDF mới 33 trang có bìa đủ bốn thành viên và 54 đích TOC đã kiểm tra. Chờ bảng thực nghiệm để tái xuất bản cuối và slide |
| Bộ nộp chạy lại | Source sạch, requirements, model manifest, tài nguyên lấy được và lệnh chạy đã thử | Cần kiểm tra package và liên kết thật |

DOCX khuyến khích ablation và đánh giá ngoài nguồn; chúng là phần mở rộng để nâng chất lượng, không phải điều kiện bắt buộc được tự thêm vào rubric. Không bắt buộc huấn luyện toàn bộ TransUNet hoặc đạt một Dice tối thiểu. Kết quả không cao nhưng được thiết kế và phân tích khoa học vẫn có giá trị.

## Thông tin không được điền bằng suy đoán

| Nội dung | Người có thể xác nhận | Tình trạng |
| --- | --- | --- |
| Thành viên và MSSV | Sinh viên/nhóm | Đủ bốn thành viên: Phạm Tấn Gia Quốc 23521308, Huỳnh Thái Bảo 23520105, Trần Minh Tiến 23521587, Lâm Quang Huy 23520609; không còn placeholder. Thông tin ở `config/team.json` và xác nhận theo hội thoại ở `Project/docs/confirmations/team.json` |
| Phân công cá nhân | Sinh viên/nhóm | Chưa có lịch sử phân công cá nhân; không gán suy đoán. Phần triển khai, lý do thiết kế và bằng chứng ở `docs/project-contributions.md` đáp ứng phần mô tả công việc của §15 |
| Lớp, giảng viên hướng dẫn và các trường hành chính trên bìa | Sinh viên/đơn vị học phần | Chưa xác nhận; không phát minh để lấp chỗ trống |
| Phiên bản ảnh và quan hệ với tài nguyên công khai | Bản kê nguồn và kiểm tra khách quan | Đã xác minh ảnh, tập nguồn và cách chia lại bằng ID/size/CRC32; lịch sử biến đổi mask còn là giới hạn riêng |
| Quy trình gán nhãn và cách hợp nhất nhãn train/valid | Công bố nguồn và tài nguyên cục bộ | Quy trình medical annotators công bố, mask cục bộ và xử lý nhãn đã mô tả tại `docs/data-annotation.md`; chưa xác minh cách tạo một mask train/valid từ mask gốc |
| Quyền lưu trữ, minh họa và phân phối ảnh/trọng số | Chủ quyền tài nguyên | License mã kiến trúc không thay thế license dữ liệu hoặc checkpoint |
| Dữ liệu, preprocessing và lịch sử checkpoint TransUNet | Tác giả checkpoint | Chưa xác minh; không cản trở mô tả nó là kết quả tham khảo khi thí nghiệm chính được kiểm soát độc lập |
| Mọi thành viên hiểu và trả lời được | Chính từng thành viên qua tập bảo vệ | Kịch bản/câu hỏi chỉ hỗ trợ chuẩn bị, không xác nhận thay năng lực |

DOCX yêu cầu mô tả nguồn, cách gán nhãn và công việc đã thực hiện, không yêu cầu chủ dữ liệu ký xác nhận hoặc một ma trận đóng góp có chữ ký. Những mục này được kiểm tra bằng tài liệu và bằng chứng khách quan, đồng thời giữ rõ giới hạn chưa biết. Định dạng JSON có `confirmed`, người xác nhận, thời gian và phạm vi chỉ là cách lưu phát biểu đã có; không phải một hồ sơ ký bổ sung do học phần quy định. Metadata nhóm ở `config/team.json` đã có `team_size: 4`, `placeholder_count: 0`. Năng lực hiểu và trả lời của từng thành viên vẫn là yêu cầu con người chưa được chứng minh; có kịch bản hoặc JSON không xác nhận thay năng lực đó. Lớp/GVHD là thông tin hành chính chưa được cung cấp.

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
- Rà lại nguyên văn §5/§15 sau khi có bằng chứng nguồn: hai mục mô tả nguồn/annotation và mục mô tả công việc của nhóm là yêu cầu tài liệu, không phải xác nhận bắt buộc từ chủ tài nguyên hay từng cá nhân. Phân loại cũ đã đặt thêm điều kiện; ba mục được sửa đúng phạm vi. Lịch sử mask và năng lực bảo vệ không được suy diễn là đã xác minh.
