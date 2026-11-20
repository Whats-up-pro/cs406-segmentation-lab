"""Prepare a revised nine-chapter source from verified evidence, preserving report.md."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(os.environ.get('CS406_PROJECT_DIR',Path(__file__).resolve().parents[1].parent/'Project')).resolve()
REPO = Path(os.environ.get('CS406_SOURCE_DIR',ROOT.parent/'cs406-segmentation-lab')).resolve()
sys.path.insert(0, str(REPO))
from scripts.analyze_completion import load_verified_evaluation

NAMES = {'baseline':'LAB–Otsu', 'pixel_logistic':'Logistic pixel', 'transunet':'TransUNet tham khảo',
         'bce_plain_s406':'BCE', 'bce_dice_s406':'BCE + Dice', 'bce_aug_s406':'BCE + tăng cường',
         'bce_dice_aug_s406':'BCE + Dice + tăng cường', 'bce_dice_aug_s407':'Tổ hợp đầy đủ seed 407',
         'bce_dice_aug_s408':'Tổ hợp đầy đủ seed 408'}
PRIMARY = 'bce_dice_aug_s406'
TABLE_TITLES = ['Phân bố ảnh và mặt nạ theo tập dữ liệu', 'Thống kê khám phá dữ liệu theo từng tập',
                'Công thức và ý nghĩa các độ đo', 'Kết quả tham khảo trên 295 ảnh kiểm tra cục bộ',
                'Kết quả Logistic với ngưỡng chọn từ tập xác thực',
                'Ảnh hưởng chuẩn hóa đầu vào trên 157 ảnh xác thực', 'Một số trường hợp dự đoán đúng và sai']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_section(text, start, end, content):
    first, last = text.index(start), text.index(end, text.index(start)+len(start))
    return text[:first]+content.rstrip()+'\n\n'+text[last:]


def table(title, headers, rows):
    return '\n\n<!-- table: '+title+' -->\n\n'+'\n'.join(
        ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n\n'


def dec(value, digits=4):
    return f'{value:.{digits}f}'.replace('.', ',')


def verified_inputs(evaluation, analysis, external, external_analysis):
    local, _ = load_verified_evaluation(evaluation)
    outside, _ = load_verified_evaluation(external)
    protocol = json.loads((REPO/'config/experiments/compact_ablation.json').read_text(encoding='utf-8'))
    required = {row['name'] for row in protocol['runs']}|{'baseline','pixel_logistic','transunet'}
    for name, summary, count in [('local',local,295),('external',outside,100)]:
        if set(summary['methods']) != required or summary['available_images'] != count:
            raise ValueError(f'Incomplete final {name} method/image coverage')
        for method in [row['name'] for row in protocol['runs']]:
            frozen = summary['methods'][method]['provenance']['frozen']
            if frozen['epochs'] != 20 or frozen['counts'] != {'train':2179,'valid':157}:
                raise ValueError('Final report requires full declared training, not pilot results')
    for method in required:
        left, right = local['methods'][method], outside['methods'][method]
        if left['threshold'] != right['threshold']:
            raise ValueError('External evaluation changed a frozen threshold')
        if method in {row['name'] for row in protocol['runs']}:
            same = left['provenance']['frozen'] == right['provenance']['frozen']
        else:
            same = left['provenance'] == right['provenance']
        if not same:
            raise ValueError('External evaluation changed model identity: '+method)
    analyses = []
    for directory, source in [(analysis,evaluation),(external_analysis,external)]:
        path = directory/'analysis.json'
        item = json.loads(path.read_text(encoding='utf-8'))
        if (item['primary_method'] != PRIMARY or item['model_selection_performed'] is not False
                or item['input_sha256']['summary.json'] != sha(source/'summary.json')
                or item['input_sha256']['per_image.csv'] != sha(source/'per_image.csv')):
            raise ValueError('Analysis is stale or differs from final evaluation')
        analyses.append(item)
    return local, outside, *analyses


def verified_cases(directory, evaluation):
    path = directory/'case-manifest.json'
    item = json.loads(path.read_text(encoding='utf-8'))
    if (item['primary_method'] != PRIMARY or item['prediction_performed'] is not False
            or item['parameter_selection_performed'] is not False
            or item['source_sha256']['evaluation/summary.json'] != sha(evaluation/'summary.json')
            or item['source_sha256']['evaluation/per_image.csv'] != sha(evaluation/'per_image.csv')):
        raise ValueError('Case figures are stale or use a different evaluation')
    for name, expected in item['output_sha256'].items():
        target = (directory/name).resolve()
        if not target.is_relative_to(directory.resolve()) or sha(target) != expected:
            raise ValueError('Case figure hash mismatch: '+name)
    return item


def prepare(args):
    original = ROOT/'docs/report.md'
    text = original.read_text(encoding='utf-8')
    text = text.replace('**So sánh phương pháp xử lý ảnh truyền thống, mô hình Logistic và TransUNet**',
                        '**So sánh LAB–Otsu, Logistic pixel, Compact U-Net và TransUNet**')
    text = text.replace('nguồn dữ liệu, ý nghĩa nhãn và lịch sử huấn luyện của tệp trọng số vẫn cần xác nhận trước khi kết luận về khả năng tổng quát hóa.',
                        'ảnh nguồn đã được đối chiếu với kho DermaVQA, còn biến đổi mask cục bộ và lịch sử huấn luyện của trọng số TransUNet chưa được xác minh đầy đủ.')
    text = text.replace('Logistic học trọng số từ đặc trưng cục bộ; TransUNet bổ sung một mô hình học sâu có sẵn trọng số.',
                        'Logistic học trọng số từ đặc trưng cục bộ; Compact U-Net học từ khởi tạo ngẫu nhiên; TransUNet bổ sung một mô hình học sâu có sẵn trọng số.')
    text = text.replace('Ba câu hỏi thực nghiệm là: quy tắc màu đạt mức chồng lấp nào; TransUNet khác LAB–Otsu bao nhiêu dưới cùng giao thức; và kết quả sai liên quan thế nào đến kích thước vùng, bất đồng nhãn hoặc giới hạn biểu diễn. Mô hình Logistic tạo thêm một đối chứng được huấn luyện với lịch sử rõ ràng.',
                        'Các câu hỏi thực nghiệm là: quy tắc màu đạt mức chồng lấp nào; mô hình học từ đầu khác đối chứng bao nhiêu; loss và tăng cường dữ liệu thay đổi kết quả thế nào dưới cùng ngân sách; kết quả biến thiên qua seed và trên dữ liệu ngoài nguồn ra sao. Logistic là một đối chứng học pixel; TransUNet là tham khảo có lịch sử trọng số chưa biết. Phân tích lỗi mô tả quan hệ với kích thước vùng và bất đồng nhãn.')
    text = text.replace('huấn luyện Logistic, phân tích kết quả và xây dựng ứng dụng minh họa.',
                        'huấn luyện Logistic và Compact U-Net, ablation loss/tăng cường, ba seed, bootstrap ghép cặp, stress test ngoài nguồn, phân tích kết quả và xây dựng ứng dụng minh họa.')
    text = text.replace('chạy một trong ba phương pháp và tải mặt nạ PNG. Logistic tự đặt ngưỡng đã chọn từ tập xác thực khi đổi phương pháp.',
                        'chạy một trong bốn phương pháp và tải mặt nạ PNG. Logistic và Compact tự đặt ngưỡng đã chọn từ tập xác thực khi đổi phương pháp.')
    text = text.replace('địa chỉ `http://127.0.0.1:8000`.',
                        'địa chỉ `http://127.0.0.1:8001` khi dùng `Start-CS406.ps1`; chạy trực tiếp Uvicorn có thể chọn cổng khác.')
    counter = 0
    def caption(match):
        nonlocal counter
        title = TABLE_TITLES[counter]; counter += 1
        return '<!-- table: '+title+' -->\n\n'+match.group()
    text = re.sub(r'(?m)^\|[^\n]+\n\|[ :|\-]+\n', caption, text)
    if counter != len(TABLE_TITLES):
        raise ValueError('Original report table inventory changed; review before rewriting')

    provenance_path = ROOT/'results/completion/provenance/comparison_summary.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if (provenance['set_equality']['local_train_plus_test_equals_official_archive_train'] is not True
            or provenance['image_integrity']['matching_zip_member_crc32'] != 2631):
        raise ValueError('Source image split provenance is not verified')
    old_start = text.index('Một nguồn ứng viên là MEDIQA-MAGIC')
    old_end = text.index('### 3.2.', old_start)
    text = text[:old_start]+'''Đối chiếu ngày 19/11/2026 xác định quan hệ với ảnh nguồn DermaVQA được ban tổ chức MEDIQA-MAGIC liên kết: hợp 2.179 ảnh train và 295 ảnh test cục bộ bằng đúng toàn bộ 2.474 ID ảnh train của kho nguồn; 157 ID validation cũng khớp toàn bộ validation nguồn. Không có ID nào trong 314 ảnh test chính thức xuất hiện ở dữ liệu cục bộ. Việc ghép được kiểm tra độc lập qua danh mục query, ánh xạ encounter và central directory của ZIP công khai [7](https://www.imageclef.org/2025/medical/mediqa), [8](https://github.com/wyim/ImageCLEF-MAGIC-2025), [9](https://osf.io/p8bfu).

Cả 2.631 ảnh cục bộ khớp kích thước byte và CRC32 của thành viên ZIP nguồn. CRC32 không thay thế SHA-256 mật mã cho từng ảnh; toàn bộ ZIP nguồn không được tải lại để băm. Bản kê có URL, revision, hash và ID đối chiếu nằm tại `results/completion/provenance`. Bởi vậy, tập test 295 ảnh là cách chia nội bộ từ ảnh train của nguồn, không phải tập test leaderboard. Điểm cuộc thi của Anastasia không được coi là kết quả của đồ án hoặc bằng chứng checkpoint đã được cung cấp là bản đoạt giải.

Giấy phép bản ảnh DermaVQA 2024 trên OSF khai báo CC BY 4.0. Quan hệ trên xác minh ảnh và split nguồn; chưa xác minh phép tạo một mask train/valid cục bộ từ các annotation gốc, điều kiện tài nguyên segmentation năm 2025 hoặc lịch sử checkpoint. Chú thích công bố mô tả vùng vấn đề da liễu; foreground trong triển khai được xác định bằng pixel nhãn khác 0. Phương pháp hiện tại chỉ dùng ảnh RGB, không đưa query, phản hồi hoặc thông tin lâm sàng bằng văn bản vào mô hình.

Quy trình công bố của DermaVQA-DAS dùng một nhóm bốn medical annotators để cung cấp ba mask cho mỗi ảnh, đánh dấu vùng vấn đề có liên quan lâm sàng [13](https://arxiv.org/html/2512.24340v2). Hai bác sĩ da liễu thiết kế schema DAS; không suy ra mọi người vẽ mask đều là bác sĩ da liễu. Số mask thực tế hơi khác đúng ba lần số ảnh. Nghiên cứu này dùng nguyên mask đã cung cấp và không thực hiện một đợt vẽ nhãn mới; quy tắc biến đổi mask trong archive trước đó vẫn chưa xác minh.

'''+text[old_end:]
    leak = '''Kiểm tra bổ sung duyệt đủ 1.031.223 cặp ảnh xuyên split, với điều kiện đồng thời pHash không quá 6 bit khác và dHash không quá 8 bit khác; không có ứng viên tại cặp ngưỡng này. Không có nhóm byte trùng hoặc encounter trùng xuyên tập. Các ngưỡng chỉ dùng tìm ứng viên gần trùng và không chứng minh mọi biến đổi ảnh đã được loại trừ. Encounter chưa phải bệnh nhân đã được xác minh. Dữ liệu gốc và 28 nhóm trùng nội bộ được giữ nguyên để tái lập.'''
    text = text.replace('### 3.3.', leak+'\n\n### 3.3.', 1)

    method_doc = (REPO/'docs/experiments-completion.md').read_text(encoding='utf-8')
    compact = method_doc[method_doc.index('## Kiến trúc Compact U-Net'):method_doc.index('## Giao thức đánh giá')]
    compact = compact.replace('## Kiến trúc Compact U-Net','### 4.6. Compact U-Net khởi tạo từ đầu')
    compact = compact.replace('## Tiền xử lý và loss','### 4.7. Tiền xử lý và hàm mất mát')
    compact = compact.replace('## Ngân sách và đóng băng lựa chọn','### 4.8. Ngân sách và đóng băng lựa chọn')
    # Give the two source tables explicit captions for native Word numbering.
    compact = compact.replace('| Mức |', '<!-- table: Kích thước đặc trưng của Compact U-Net -->\n\n| Mức |', 1)
    compact = compact.replace('| Tham số |', '<!-- table: Cấu hình huấn luyện và lựa chọn trên validation -->\n\n| Tham số |', 1)
    compact = compact.replace('[U-Net](https://arxiv.org/1505.04597)', '[2](https://arxiv.org/abs/1505.04597)')
    compact = compact.replace('[U-Net](https://arxiv.org/abs/1505.04597)', '[2](https://arxiv.org/abs/1505.04597)')
    compact = compact.replace('[Group Normalization](https://arxiv.org/abs/1803.08494)', 'Group Normalization [10](https://arxiv.org/abs/1803.08494)')
    compact = compact.replace('[V-Net](https://arxiv.org/abs/1606.04797)', 'V-Net [11](https://arxiv.org/abs/1606.04797)')
    text = text.replace('## 5. Thiết lập thực nghiệm', compact+'\n\n## 5. Thiết lập thực nghiệm', 1)
    text = text.replace('không thực hiện một thí nghiệm U-Net độc lập và không đưa điểm số từ bài báo vào bảng so sánh cục bộ.',
                        'bổ sung một biến thể Compact U-Net được khởi tạo ngẫu nhiên cho thí nghiệm có kiểm soát; không đưa điểm số từ bài báo vào bảng so sánh cục bộ.')

    analysis_methods = method_doc[method_doc.index('## Giao thức đánh giá'):method_doc.index('## Lệnh và bằng chứng')]
    analysis_methods = analysis_methods.replace('## Giao thức đánh giá','### 5.4. Giao thức so sánh mở rộng và khoảng tin cậy')
    analysis_methods = analysis_methods.replace('## Stress test ngoài nguồn','### 5.5. Stress test ISIC ngoài nguồn')
    text = text.replace('## 6. Kết quả thực nghiệm', analysis_methods+'\n\n## 6. Kết quả thực nghiệm', 1)

    receipts = {'source_report_sha256':sha(original), 'source_provenance_sha256':sha(provenance_path), 'draft':args.draft}
    if args.draft:
        extension = '''### 6.4. Trạng thái thí nghiệm mở rộng

Sáu cấu hình Compact U-Net đã được cố định trước bước đánh giá mới. Chưa có đủ sáu lịch sử 20 epoch và kết quả ngoài nguồn, nên phần này chưa báo cáo điểm số cho Compact hoặc ablation. Các số liệu ở mục 6.1–6.3 thuộc các thí nghiệm tham khảo đã hoàn tất trước đó.
'''
    else:
        paths = [args.evaluation_dir,args.analysis_dir,args.external_dir,args.external_analysis_dir]
        if any(path is None for path in paths):
            raise ValueError('Final source requires local/external evaluation and analysis directories')
        local, outside, analysis, external_analysis = verified_inputs(*paths)
        if args.case_dir is None:
            raise ValueError('Final report requires verified saved-mask error examples')
        cases = verified_cases(args.case_dir,args.evaluation_dir)
        spec = json.loads((REPO/'config/experiments/compact_ablation.json').read_text(encoding='utf-8'))
        extension = '### 6.4. Compact U-Net và ablation có kiểm soát\n\n'
        extension += 'Tất cả sáu lần chạy hoàn tất đủ 20 epoch, mỗi epoch dùng 2.179 ảnh train và 157 ảnh validation. Recipe chính là tổ hợp BCE + Dice + augmentation tại seed 406 đã được khai báo trước; không chọn seed tốt nhất bằng test.\n'
        extension += table('Kết quả mở rộng trên cùng 295 ảnh test cục bộ', ['Phương pháp','Dice','IoU','Precision','Recall','BF1'],
                           [[NAMES[key]]+[dec(local['methods'][key]['macro']['mean'][metric]) for metric in ('dice','iou','precision','recall','boundary_f1')]
                            for key in ('baseline','pixel_logistic',PRIMARY,'transunet')])
        extension += table('Ablation loss và augmentation ở seed 406', ['Cấu hình','Epoch chọn','Ngưỡng valid','Dice test','IoU test'],
                           [[NAMES[row['name']], local['methods'][row['name']]['provenance']['frozen']['best_epoch'],
                             dec(local['methods'][row['name']]['threshold'],2), dec(local['methods'][row['name']]['macro']['mean']['dice']),
                             dec(local['methods'][row['name']]['macro']['mean']['iou'])] for row in spec['runs'][:4]])
        variation = analysis['seed_variation']['scores']
        extension += f"Tổ hợp đầy đủ qua ba seed có Dice trung bình {dec(variation['dice']['mean'])}, độ lệch chuẩn mẫu {dec(variation['dice']['sample_std'])}; IoU tương ứng {dec(variation['iou']['mean'])} và {dec(variation['iou']['sample_std'])}. Đây là biến thiên giữa run trên cùng benchmark, không phải ba bộ dữ liệu độc lập.\n"
        extension += table('Chênh lệch Dice của recipe chính và khoảng bootstrap 95 phần trăm', ['Đối chứng','Chênh lệch','CI thấp','CI cao'],
                           [[NAMES[key],dec(analysis['comparisons'][PRIMARY+'-minus-'+key]['dice']['delta']),
                             *[dec(value) for value in analysis['comparisons'][PRIMARY+'-minus-'+key]['dice']['ci95']]]
                            for key in ('baseline','pixel_logistic','bce_plain_s406','bce_dice_s406','bce_aug_s406','transunet')])
        extension += 'Các khoảng ghép cặp lấy mẫu lại encounter và cân bằng theo ảnh; so sánh mang tính thăm dò, không khẳng định quan hệ nhân quả hoặc ưu thế kiến trúc. Điểm TransUNet thuộc checkpoint có lịch sử chưa biết, nên không mang cùng bảo đảm về exposure như các lần Compact học từ đầu.\n\n'
        figures = ROOT/'docs/figures/completion'; figures.mkdir(parents=True,exist_ok=True)
        for name in ('method_comparison.png','training_curves.png'):
            source = args.analysis_dir/name
            if not source.is_file(): raise ValueError('Missing final analysis figure: '+name)
            shutil.copyfile(source, figures/name)
        extension += '![So sánh các cấu hình đã đóng băng trên cùng benchmark.](figures/completion/method_comparison.png)\n\n'
        extension += '![Đường loss huấn luyện và Dice validation từng epoch.](figures/completion/training_curves.png)\n\n'
        extension += '### 6.5. Stress test ngoài nguồn ISIC\n\n'
        extension += table('Đánh giá không tinh chỉnh trên 100 ảnh ISIC validation', ['Phương pháp','Dice','IoU','Precision','Recall','BF1'],
                           [[NAMES[key]]+[dec(outside['methods'][key]['macro']['mean'][metric]) for metric in ('dice','iou','precision','recall','boundary_f1')]
                            for key in ('baseline','pixel_logistic',PRIMARY,'transunet')])
        extension += 'Ngưỡng, trọng số và cách chuẩn hóa giữ nguyên từ nguồn cục bộ. Khác biệt ảnh dermoscopy và ranh giới tổn thương giới hạn cách diễn giải; không coi điểm ISIC là kiểm định lâm sàng. Với TransUNet, dữ liệu huấn luyện gốc chưa biết nên không xác nhận ISIC là dữ liệu chưa từng được checkpoint quan sát.\n\n'
        reasons = {'best_foreground_dice':'Dice cao nhất khi nhãn không rỗng', 'worst_dice':'Dice thấp nhất',
                   'largest_primary_minus_baseline':'Chênh lệch lớn nhất so với LAB–Otsu',
                   'smallest_primary_minus_baseline':'Chênh lệch nhỏ nhất so với LAB–Otsu'}
        error_section = '### 7.4. Trường hợp Compact U-Net từ mặt nạ đã lưu\n\n'
        error_section += 'Ví dụ được chọn sau khi đánh giá theo bốn quy tắc cố định: Dice cao nhất với nhãn không rỗng, Dice thấp nhất, chênh lệch Dice lớn nhất và nhỏ nhất so với LAB–Otsu; mẫu trùng được gộp. Đây là mô tả lỗi sau đánh giá, không dùng chọn lại tham số. Mỗi hình được dựng từ mask PNG đã lưu, đối chiếu ma trận TP/FP/FN/TN và toàn bộ độ đo với CSV; không chạy suy luận lại.\n'
        error_section += table('Các ví dụ Compact U-Net theo quy tắc xếp hạng đã khai báo',
                               ['Mẫu','Quy tắc chọn','Dice Compact','Dice LAB'],
                               [[row['id'], '; '.join(reasons[key] for key in row['selection_reasons']),
                                 dec(row['primary_dice']),dec(row['baseline_dice'])] for row in cases['cases']])
        for name in cases['plates']:
            shutil.copyfile(args.case_dir/name,figures/name)
            error_section += f'![Ảnh RGB, nhãn, dự đoán Compact và sai khác; TP xanh lục, FP đỏ, FN xanh lam.](figures/completion/{name})\n\n'
        error_section += 'Vùng đỏ là dự đoán foreground ngoài nhãn; vùng lam là foreground trong nhãn bị bỏ sót; nền đen là TN. Các hình cho phép xác định vị trí và mức dự đoán thừa hoặc thiếu ở từng mẫu. Chỉ số xếp hạng không tự xác định nguyên nhân do màu da, ánh sáng hay kiến trúc; diễn giải nhân quả cần thí nghiệm bổ sung.\n\n'
        text = text.replace('## 8.',error_section+'## 8.',1)
        receipts['case_manifest_sha256'] = sha(args.case_dir/'case-manifest.json')
        receipts['final_evidence_sha256'] = {str(path/'summary.json' if i in (0,2) else path/'analysis.json'):sha(path/('summary.json' if i in (0,2) else 'analysis.json')) for i,path in enumerate(paths)}
        main_dice = dec(local['methods'][PRIMARY]['macro']['mean']['dice'])
        external_dice = dec(outside['methods'][PRIMARY]['macro']['mean']['dice'])
        abstract_extension = f'''Thực nghiệm mở rộng bổ sung Compact U-Net có 274.849 tham số, được khởi tạo ngẫu nhiên và chỉ cập nhật trên train. Sáu cấu hình hoàn tất đủ 20 epoch, dùng validation để chọn checkpoint và ngưỡng trước khi tính điểm mới trên benchmark. Recipe chính đạt Dice {main_dice} ở 295 ảnh test nội bộ và {external_dice} ở 100 ảnh ISIC validation không tinh chỉnh. Ablation loss/augmentation, ba seed và bootstrap ghép cặp mô tả mức biến thiên quan sát được. Đối chiếu nguồn xác định train và test nội bộ cùng chia lại từ ảnh train chính thức; kết quả không tái lập leaderboard và không xác nhận lịch sử checkpoint TransUNet.

'''
        first_chapter = re.search(r'(?m)^## 1\..*$',text)
        text = text[:first_chapter.start()]+abstract_extension+text[first_chapter.start():]
        conclusion = f'''## 9. Kết luận và hướng phát triển

Đồ án xây dựng quy trình kiểm tra dữ liệu, huấn luyện có kiểm soát và đánh giá phân đoạn RGB, cùng ứng dụng dự đoán thật bằng bốn phương pháp. Compact U-Net khởi tạo ngẫu nhiên đạt Dice {main_dice} với recipe chính trên 295 ảnh test cục bộ; các tổ hợp loss và augmentation cùng ba seed được đánh giá dưới ngân sách 20 epoch. Báo cáo giữ kết quả tốt lẫn kết quả không cải thiện, phân biệt tham khảo TransUNet với mô hình tự huấn luyện và bổ sung stress test 100 ảnh ISIC không tinh chỉnh.

Đối chiếu ID và CRC32 xác nhận ảnh cục bộ lấy từ kho ảnh nguồn, trong đó train và test nội bộ cùng chia từ 2.474 ảnh train của nguồn; 314 ảnh test chính thức không có trong bộ cục bộ. Kiểm tra byte, encounter và ứng viên perceptual không phát hiện giao nhau xuyên split tại giao thức đã khai báo. Điều này chưa xác nhận độc lập theo bệnh nhân, biến đổi mask gốc hoặc lịch sử checkpoint được cung cấp. Test cũng đã được xem ở giai đoạn trước, nên kết quả là benchmark hồi cứu có lựa chọn mô hình đóng băng.

Hướng phát triển là xác minh quy trình tạo mask, đánh giá trên dữ liệu độc lập theo bệnh nhân và cơ sở, rồi thử bảo toàn tỷ lệ hoặc patch độ phân giải cao bằng một giao thức mới chọn trên validation. Những thay đổi tương lai cần ngân sách, đối chứng và phân tích lỗi tương ứng trước khi kết luận về hiệu quả. Hệ thống hiện phục vụ nghiên cứu và minh họa, chưa được thẩm định sử dụng lâm sàng.
'''
        text = replace_section(text, '## 9.', '## Tài liệu tham khảo', conclusion)
    text = text.replace('## 7. Phân tích các trường hợp sai', extension+'\n\n## 7. Phân tích các trường hợp sai', 1)
    if '### 5.4.' not in text:
        # Preserve the original chapter title if its wording differs.
        marker = re.search(r'(?m)^## 6\..*$', text)
        text = text[:marker.start()]+analysis_methods+'\n\n'+text[marker.start():]
    text = re.sub(r'8\. W\.-w\. Yim.*?(?=\n\n|$)',
                  '8. W.-w. Yim và ban tổ chức. *ImageCLEF-MAGIC-2025 task resources and evaluation code*. [Kho mã nguồn chính thức](https://github.com/wyim/ImageCLEF-MAGIC-2025). Ảnh nguồn được đối chiếu bằng bản kê chính thức; các giới hạn về mask và checkpoint được nêu ở Chương 3.', text, flags=re.S)
    tail = text.find('\nNguồn ứng viên đã được đối chiếu')
    if tail != -1:text = text[:tail]
    text += '''
9. Ban tổ chức DermaVQA. *Kho ảnh nguồn images_final.zip*. [OSF](https://osf.io/p8bfu). Bản kê ZIP và metadata query được đối chiếu ngày 19/11/2026; hash và revision lưu trong bằng chứng provenance.
10. Y. Wu và K. He (2018). *Group Normalization*. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494).
11. F. Milletari, N. Navab và S.-A. Ahmadi (2016). *V-Net Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. [arXiv:1606.04797](https://arxiv.org/abs/1606.04797).
12. ISIC. *ISIC 2018 Challenge datasets*. [Trang dữ liệu chính thức](https://challenge.isic-archive.com/data/#2018). Dữ liệu Task 1 validation dùng như stress test ngoài nguồn, ảnh và nhãn có manifest hash.
13. W.-w. Yim và cộng sự (2025, cập nhật 2026). *DermaVQA-DAS: Dermatology Assessment Schema (DAS) & Datasets for Closed-Ended Question Answering & Segmentation in Patient-Generated Dermatology Images*. [arXiv:2512.24340v2](https://arxiv.org/html/2512.24340v2), mục 2 và bảng 3; đối chiếu ngày 19/11/2026.
'''
    assert all(re.search(rf'(?m)^## {number}\. ',text) for number in range(1,10))
    path = ROOT/'docs/report-completion.md'
    path.write_text(text, encoding='utf-8')
    receipts['generated_source_sha256'] = sha(path)
    (ROOT/'output/completion/report-source-manifest.json').write_text(json.dumps(receipts,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'source':str(path),'draft':args.draft,'characters':len(text)},ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',action='store_true',help='Private layout draft; retains only completed original result tables')
    parser.add_argument('--evaluation-dir',type=Path)
    parser.add_argument('--analysis-dir',type=Path)
    parser.add_argument('--external-dir',type=Path)
    parser.add_argument('--external-analysis-dir',type=Path)
    parser.add_argument('--case-dir',type=Path)
    prepare(parser.parse_args())
