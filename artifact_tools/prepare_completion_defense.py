"""Write the defense guide from the same frozen evidence as the final slides."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source,output):
    data=json.loads(source.read_text(encoding='utf-8'))
    if data['format']!='cs406-defense-input-v1' or data['draft'] is not False:
        raise ValueError('Final presentation evidence is required')
    for filename,expected in {**data['evidence_sha256'],**data['asset_sha256']}.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
            raise ValueError('Evidence changed: '+filename)
    primary=data['primary'];local=data['local']['methods'];external=data['external']['methods']
    frozen=data['runs'][primary]['frozen'];ci=data['analysis']['comparisons'][primary+'-minus-baseline']['dice']
    number=lambda value:f'{value:.4f}'.replace('.',',')
    members=data['team']['members']
    roles=[('1–4','2 phút 20 giây','Bài toán, nguồn ảnh, split, nhãn đồng thuận'),
           ('5–8','3 phút 20 giây','Bốn phương pháp và thiết kế huấn luyện/ablation'),
           ('9–12','3 phút 20 giây','Đường học, kết quả local, ablation/seed, ISIC'),
           ('13–15','3 phút','Lỗi thật, demo và kết luận có giới hạn')]
    text=f'''# Hướng dẫn bảo vệ CS406

Tài liệu dùng cùng số liệu đã kiểm chứng với bộ 15 slide chính và 4 phụ lục. Mã nguồn công khai tại https://github.com/Whats-up-pro/cs406-segmentation-lab. Tài nguyên ảnh và bằng chứng đầy đủ nằm trong gói nộp riêng. Đây là tài liệu chuẩn bị; chưa xác nhận nhóm đã tập bảo vệ hoặc mọi thành viên đã hiểu toàn bộ phương pháp.

## Phân vai tập thử trong 12 phút

Phân vai dưới đây là gợi ý trình bày, không phải khai báo ai đã thực hiện phần kỹ thuật nào. Mọi thành viên cần giải thích được toàn bộ pipeline và đổi vai khi luyện tập.

| Thành viên | MSSV | Slide | Thời lượng | Nội dung |
| --- | --- | --- | --- | --- |
'''
    text+='\n'.join('| '+' | '.join([person['name'],person['student_id'],*role])+' |' for person,role in zip(members,roles))
    text+=f'''

## Những số liệu cần nói đúng

- Local: 2.179 ảnh train, 157 valid, 295 test; mask tương ứng 2.179/157/885.
- 2.631 ảnh khớp byte size và CRC32 của kho ảnh nguồn DermaVQA. Train + test local chia lại toàn bộ 2.474 ảnh train nguồn; không có 314 ảnh test chính thức.
- Compact U-Net có 274.849 tham số, khởi tạo ngẫu nhiên. Sáu run đủ 20 epoch; chỉ train cập nhật tham số.
- Recipe chính là BCE + Dice + augmentation, seed 406, đã khai báo trước khi đánh giá mới. Chọn epoch {frozen['best_epoch']}; ngưỡng valid {frozen['threshold']:.2f}. Không thay bằng seed có test cao nhất.
- Macro Dice local: LAB–Otsu {number(local['baseline']['macro']['mean']['dice'])}; Logistic {number(local['pixel_logistic']['macro']['mean']['dice'])}; Compact {number(local[primary]['macro']['mean']['dice'])}; TransUNet tham khảo {number(local['transunet']['macro']['mean']['dice'])}.
- Compact − LAB: ΔDice {number(ci['delta'])}; CI bootstrap 95% [{number(ci['ci95'][0])}; {number(ci['ci95'][1])}]. Đây là 2.000 bootstrap ghép cặp encounter, không phải kiểm định ưu thế kiến trúc.
- 100 ảnh ISIC validation dùng stress test ngoài nguồn, giữ nguyên trọng số/ngưỡng/chuẩn hóa. Compact Dice {number(external[primary]['macro']['mean']['dice'])}; không xem điểm này là kiểm định lâm sàng.
- TransUNet được cấp trọng số; chưa biết lịch sử huấn luyện. Điểm leaderboard của Anastasia không phải điểm đồ án và chưa xác nhận file được cấp là checkpoint đội đoạt giải.

## Kịch bản demo khoảng 90 giây

1. Trong PowerShell tại thư mục workspace, chạy `powershell -ExecutionPolicy Bypass -File .\\Start-CS406.ps1 -CheckOnly`, rồi chạy lại không có `-CheckOnly`. Mở `http://127.0.0.1:8001`.
2. Chọn một ảnh validation và Compact U-Net. Nêu rõ ngưỡng được lấy từ valid và model đã học đủ 20 epoch. Chạy phân đoạn thật.
3. Đổi giữa ảnh, mask và lớp sai khác. Chỉ đúng một vùng TP, FP hoặc FN bằng màu; đổi annotation nếu mẫu có nhiều nhãn.
4. Xuất PNG và kiểm tra kích thước bằng ảnh đầu vào. Nói rõ metric trên ảnh gốc của demo khác benchmark 256×256 trong bảng báo cáo.
5. Tải một ảnh mới lên, chạy lại. Chỉ có dự đoán và độ phủ, không có Dice/IoU khi không có nhãn.
6. Mở trang Thực nghiệm để đối chiếu bảng đã lưu. Nếu dùng ảnh hoặc slide dự phòng, nói rõ đó là bằng chứng của lần chạy đã ghi.

## Câu hỏi kỹ thuật và cách trả lời

**Nhóm tự làm gì?** Kiểm kê/EDA và đối chiếu nguồn; hiện thực LAB–Otsu và Logistic; huấn luyện Compact từ đầu; ablation loss/tăng cường và ba seed; đánh giá local/ISIC, bootstrap và lỗi từ mask thật; xây API/UI cùng tài liệu tái lập. TransUNet dùng kiến trúc và trọng số có dẫn nguồn, không phải trọng số nhóm tự học.

**Dữ liệu lấy từ đâu?** Ảnh được ghép đủ với kho DermaVQA do ban tổ chức MEDIQA-MAGIC liên kết, bằng ID, split, byte size và CRC32. CRC32 không thay per-file SHA mật mã của nguồn. Ảnh train/test local thuộc train nguồn. Chưa xác minh phép tạo một mask train/valid cục bộ từ annotation gốc; không khẳng định độc lập bệnh nhân từ filename encounter.

**Nhãn do ai tạo?** Công bố DermaVQA-DAS mô tả nhóm bốn medical annotators với ba mask danh nghĩa mỗi ảnh. Hai bác sĩ da liễu thiết kế schema DAS, không đồng nghĩa mọi người vẽ mask là bác sĩ da liễu. Dự án không tạo một đợt nhãn mới. Trong local, mọi pixel khác 0 là foreground.

**Consensus tính ra sao?** Nhị phân hóa từng annotation, resize nearest về 256×256, cộng phiếu. Foreground khi số phiếu lớn hơn n/2; hòa là nền. Đây là quy tắc local, khác mã chính thức khi n chẵn. Không chọn annotation tạo điểm tốt nhất.

**Vì sao Compact U-Net?** Cần một mô hình có lịch sử cập nhật rõ ràng, chạy được trên CPU. Mạng 12/24/48/96 kênh và skip connection có 274.849 tham số. Đây là biến thể gọn từ encoder–decoder U-Net, không tuyên bố kiến trúc mới hoặc thay thế TransUNet trên mọi dữ liệu.

**Loss tối ưu gì?** BCEWithLogits + λ(1−soft Dice), λ=0 hoặc 1. Soft Dice tính riêng mỗi ảnh rồi trung bình batch; smoothing 10⁻⁶. BCE là loss theo pixel, Dice bổ sung mục tiêu chồng lấp. Loss giảm không đảm bảo Dice nhị phân tăng ở mọi epoch.

**Tăng cường có làm sai nhãn?** Flip ngang/dọc và xoay k×90° dùng cùng biến đổi cho RGB/mask. Brightness/contrast chỉ tác động RGB. Chỉ train tăng cường; valid và test giữ nguyên.

**Test có tham gia chọn không?** Đánh giá gốc đã được xem để phân tích lỗi và được công bố là benchmark hồi cứu. Trong nghiên cứu Compact mở rộng, recipe được khai báo, checkpoint và ngưỡng chọn từ valid và băm trước lần chấm mới. Điều đó giảm lựa chọn sau test nhưng không biến test cũ thành một tập mù hoàn toàn.

**Ablation chứng minh điều gì?** Bốn tổ hợp BCE/Dice × không/có augmentation dùng cùng seed 406 và ngân sách. Báo cáo cả điểm tăng và giảm; mỗi tổ hợp chưa có ba seed riêng nên không quy kết mọi chênh lệch thành quan hệ nhân quả. Ba seed chỉ dành cho tổ hợp đầy đủ.

**Macro khác micro thế nào?** Macro lấy trung bình metric từng ảnh, mỗi ảnh có trọng số bằng nhau. Micro cộng TP/FP/FN/TN trước khi tính, chịu trọng số diện tích foreground và nền. Không đổi IoU trung bình sang Dice trung bình bằng công thức phi tuyến.

**Boundary F1 bổ sung gì?** So sự khớp ranh giới với dung sai Euclid 2 pixel tại 256×256, bổ sung overlap diện tích. Dung sai phụ thuộc độ phân giải; không suy ra ranh giới ảnh gốc đã tốt.

**CI có phải độ tin cậy y tế không?** Không. Bootstrap mô tả chênh lệch trên benchmark dưới giả định cụm encounter. Filename không xác minh bệnh nhân; nhiều so sánh là thăm dò, chưa hiệu chỉnh familywise. Khoảng này không loại bỏ bias nguồn/checkpoint hay test đã được xem.

**Vì sao ISIC có thể giảm điểm?** Khác modality dermoscopy, nội dung và quy trình nhãn. Chỉ báo cáo mức chuyển giao quan sát được, không tự gán nguyên nhân nếu chưa có đối chứng. Giữ nguyên threshold/weights; không tune ISIC. Lịch sử TransUNet chưa biết nên không xác nhận checkpoint chưa từng thấy ISIC.

**Có phát hiện leakage không?** Không có trùng byte hoặc encounter xuyên split local; không có ứng viên đồng thời pHash≤6 và dHash≤8 trên các cặp đã kiểm tra. ISIC cũng không có trùng/ứng viên tại ngưỡng đó. Kết quả không loại trừ mọi phép biến đổi hoặc quan hệ bệnh nhân. Trong mỗi split vẫn có ảnh trùng được giữ nguyên và mô tả.

**Vì sao không dùng điểm cuộc thi?** Tập ảnh/split, quy tắc consensus, độ phân giải và cách tổng hợp khác. Không có test chính thức trong archive local. Hạng nhất công bố của đội Anastasia là thông tin tham khảo, không xác minh danh tính checkpoint hoặc tái lập kết quả.

**Chạy lại được tới đâu?** Có source/config/seed, hash ảnh/mask và source, history 20 epoch, checkpoint, lựa chọn valid, CSV từng ảnh/annotation, PNG dự đoán. Resume phục hồi optimizer và lịch sử với hash khớp. CPU deterministic nghiêm ngặt; CUDA có cảnh báo kernel bilinear nên không hứa hash CPU/GPU giống nhau.

## Tập bảo vệ trước buổi nộp

Mỗi người tự vẽ pipeline RGB → predictor → xác suất → ngưỡng → mask; giải thích nhãn chỉ dùng ở bước học/chấm. Chọn một mẫu lỗi thật và tự tính Dice từ TP/FP/FN. Đổi vai trình bày phương pháp của người khác. Chạy demo một lần khi không có mạng và mở được tài liệu dự phòng. Ghi câu chưa trả lời được rồi kiểm tra lại mã nguồn/kết quả; không học thuộc lời khẳng định vượt bằng chứng.

Bằng chứng đầu vào của hướng dẫn: SHA-256 `{hashlib.sha256(source.read_bytes()).hexdigest()}`. Thông tin phân vai trình bày ở trên là đề xuất, không thay thế yêu cầu tất cả thành viên hiểu và trả lời được đồ án.
'''
    if output.exists():raise FileExistsError('Use a new guide path; preserve existing artifacts')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(text,encoding='utf-8')
    print(json.dumps({'guide':str(output.resolve())}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.input,args.output)
