"""Build the revised course report without overwriting the earlier deliverables."""
import argparse
import json
import os
import re
from pathlib import Path
from PIL import Image
from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ROOT = Path(os.environ.get('CS406_PROJECT_DIR',Path(__file__).resolve().parents[1].parent/'Project')).resolve()
REPO = Path(os.environ.get('CS406_SOURCE_DIR',ROOT.parent/'cs406-segmentation-lab')).resolve()
FONT = 'Times New Roman'
WIDTH_CM = 15.5
TABLE_TITLES = [
    'Phân bố ảnh và mặt nạ theo tập dữ liệu',
    'Thống kê khám phá dữ liệu theo từng tập',
    'Công thức và ý nghĩa các độ đo',
    'Kết quả so sánh trên 295 ảnh kiểm tra',
    'Kết quả Logistic với ngưỡng chọn từ tập xác thực',
    'Ảnh hưởng chuẩn hóa đầu vào trên 157 ảnh xác thực',
    'Một số trường hợp dự đoán đúng và sai',
]
ABBREVIATIONS = [
    ('RGB', 'Không gian màu gồm ba kênh đỏ, lục và lam.'),
    ('CIELAB', 'Không gian màu với độ sáng L và hai thành phần màu a, b.'),
    ('EDA', 'Phân tích khám phá dữ liệu.'),
    ('CNN', 'Mạng nơ-ron tích chập.'),
    ('ViT', 'Vision Transformer, kiến trúc Transformer dành cho ảnh.'),
    ('SGD', 'Bộ tối ưu hạ gradient ngẫu nhiên.'),
    ('BCE', 'Hàm mất mát entropy chéo nhị phân.'),
    ('Dice, IoU', 'Các độ đo mức chồng lấp giữa dự đoán và nhãn.'),
    ('TP, FP, FN, TN', 'Lần lượt: đúng vùng quan tâm, dự đoán thừa, bỏ sót và đúng nền.'),
    ('JSON, CSV', 'Các định dạng lưu cấu hình và bảng kết quả thực nghiệm.'),
    ('TIFF, PNG', 'Các định dạng ảnh dùng cho nhãn và mặt nạ.'),
    ('CPU, CUDA', 'Bộ xử lý trung tâm và nền tảng tính toán GPU của NVIDIA.'),
    ('SHA-256', 'Hàm băm dùng để kiểm tra tệp trùng byte và truy vết phiên bản.'),
]


def clean(text):
    return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text).replace('**', '').replace('`', '').replace('*', '')


def parse_source():
    lines = (ROOT / 'docs/report-completion.md').read_text(encoding='utf-8').splitlines()
    blocks = []
    next_table_title = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith('<!-- table:') and line.endswith('-->'):
            next_table_title = line[len('<!-- table:'):-3].strip()
        elif line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            blocks.append({'kind': 'heading', 'level': level, 'text': clean(line[level:].strip())})
        elif line.startswith('```'):
            values = []; i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                values.append(lines[i]); i += 1
            blocks.append({'kind': 'code', 'text': '\n'.join(values)})
        elif line.startswith('|'):
            values = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [clean(v.strip()) for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', v.replace(' ', '')) for v in row): values.append(row)
                i += 1
            if not next_table_title:
                raise ValueError('Every report table requires an explicit caption')
            blocks.append({'kind': 'table', 'rows': values, 'title': next_table_title})
            next_table_title = None
            continue
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            blocks.append({'kind': 'figure', 'text': match[1], 'path': match[2]})
        else:
            blocks.append({'kind': 'paragraph', 'text': line})
        i += 1
    chapter = 0; fc = {}; tc = {}; table_index = 0
    for b in blocks:
        if b['kind'] == 'heading':
            if b['level'] == 2 and re.match(r'^\d+\.', b['text']):
                number, title = b['text'].split('.', 1); chapter = int(number)
                b.update(key=f'chapter_{chapter}', display=f'CHƯƠNG {chapter}. {title.strip().upper()}', outline=1)
            elif b['level'] >= 3:
                b.update(key='heading_'+re.match(r'[\d.]+', b['text'])[0].strip('.').replace('.', '_'), display=b['text'], outline=b['level']-1)
            elif b['text'] == 'Tóm tắt':
                b.update(key='summary', display='TÓM TẮT', outline=1)
            elif b['text'] == 'Tài liệu tham khảo':
                b.update(key='references', display='TÀI LIỆU THAM KHẢO', outline=1)
        elif b['kind'] == 'figure':
            fc[chapter] = fc.get(chapter, 0) + 1
            number = f'{chapter}.{fc[chapter]}'
            b.update(key='figure_'+number.replace('.', '_'), display=f'Hình {number}. {b["text"]}')
            b['nav_label']=b['display']
        elif b['kind'] == 'table':
            tc[chapter] = tc.get(chapter, 0) + 1
            number = f'{chapter}.{tc[chapter]}'
            b.update(key='table_'+number.replace('.', '_'), display=f'Bảng {number}. {b["title"]}')
            table_index += 1
    return blocks


def font(run, size=None, bold=None, italic=None, name=FONT):
    run.font.name = name
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), name)
    run._element.rPr.rFonts.set(qn('w:cs'), name)
    run.font.color.rgb = RGBColor(0,0,0)
    if size is not None: run.font.size = Pt(size)
    if bold is not None: run.bold = bold
    if italic is not None: run.italic = italic


def link(p, label, target=None, anchor=None, size=13, bold=False):
    node = OxmlElement('w:hyperlink')
    if anchor: node.set(qn('w:anchor'), anchor)
    else: node.set(qn('r:id'), p.part.relate_to(target, RT.HYPERLINK, is_external=True))
    r = OxmlElement('w:r'); pr = OxmlElement('w:rPr')
    fonts = OxmlElement('w:rFonts')
    for a in ['ascii','hAnsi','eastAsia','cs']: fonts.set(qn('w:'+a), FONT)
    pr.append(fonts)
    size_el = OxmlElement('w:sz'); size_el.set(qn('w:val'), str(int(size*2))); pr.append(size_el)
    color = OxmlElement('w:color'); color.set(qn('w:val'), '000000'); pr.append(color)
    underline=OxmlElement('w:u');underline.set(qn('w:val'),'none');pr.append(underline)
    if bold: pr.append(OxmlElement('w:b'))
    r.append(pr); text = OxmlElement('w:t'); text.text=label; text.set(qn('xml:space'),'preserve'); r.append(text)
    node.append(r); p._p.append(node)


def inline(p, value):
    regex = re.compile(r'(\*\*.+?\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))')
    for token in regex.split(value):
        if not token: continue
        if token.startswith('**'): font(p.add_run(token[2:-2]), bold=True)
        elif token.startswith('*'): font(p.add_run(token[1:-1]), italic=True)
        elif token.startswith('`'): font(p.add_run(token[1:-1]), size=10, name='Consolas')
        elif re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)', token):
            label,target = re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)', token).groups()
            if label.isdigit(): link(p, '['+label+']', anchor='ref_'+label)
            elif target.startswith(('https://','http://')): link(p,label,target=target)
            else: font(p.add_run(label))
        else: font(p.add_run(token))


def bookmark(p, key, counter):
    first = OxmlElement('w:bookmarkStart'); first.set(qn('w:id'), str(counter)); first.set(qn('w:name'), key)
    last = OxmlElement('w:bookmarkEnd'); last.set(qn('w:id'), str(counter))
    p._p.insert(0, first); p._p.append(last)


def field_run(p, kind, instruction=None, hidden=False):
    r = p.add_run()
    if hidden: r.font.hidden=True
    if kind == 'instruction':
        n=OxmlElement('w:instrText');n.set(qn('xml:space'),'preserve');n.text=instruction
    else:
        n=OxmlElement('w:fldChar');n.set(qn('w:fldCharType'),kind)
    r._r.append(n)


def caption_text(p, value):
    """Use visible SEQ captions, the source Word expects for a table of figures."""
    label, chapter, sequence, title = re.fullmatch(r'(Hình|Bảng) (\d+)\.(\d+)\. (.+)', value).groups()
    font(p.add_run(f'{label} {chapter}.'),size=12)
    field_run(p,'begin')
    reset='\\r 1 ' if sequence=='1' else ''
    field_run(p,'instruction',f' SEQ {label} {reset}\\* ARABIC ')
    field_run(p,'separate')
    font(p.add_run(sequence),size=12)
    field_run(p,'end')
    font(p.add_run('. '+title),size=12)


def set_section(section, numbering=None):
    section.page_width=Cm(21);section.page_height=Cm(29.7)
    section.left_margin=Cm(3.5);section.right_margin=Cm(2)
    section.top_margin=Cm(2.5);section.bottom_margin=Cm(2.5)
    section.header_distance=Cm(1.25);section.footer_distance=Cm(1.25)
    for n in list(section._sectPr.findall(qn('w:pgNumType'))):section._sectPr.remove(n)
    section.header.is_linked_to_previous=False;section.footer.is_linked_to_previous=False
    if numbering:
        n=OxmlElement('w:pgNumType');n.set(qn('w:fmt'),numbering);n.set(qn('w:start'),'1');section._sectPr.append(n)
        p=section.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.line_spacing=1
        f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');p._p.append(f)


def style_document(d):
    for s in d.styles:
        if hasattr(s,'font'):
            s.font.name=FONT;s.font.color.rgb=RGBColor(0,0,0)
            pr=s.element.get_or_add_rPr();rf=pr.get_or_add_rFonts()
            for a in ['ascii','hAnsi','eastAsia','cs']:rf.set(qn('w:'+a), FONT)
        for n in list(s.element.iter(qn('w:pBdr'))):n.getparent().remove(n)
    normal=d.styles['Normal'];normal.font.size=Pt(13)
    pf=normal.paragraph_format;pf.line_spacing=1.5;pf.space_after=Pt(0)
    pf.first_line_indent=Cm(1);pf.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY;pf.widow_control=True
    for name in ['FrontTitle','FrontSubTitle','NavigationTitle','TOC 1','TOC 2','Table of Figures','CodeBlock','ReferenceEntry']:
        if name not in d.styles:d.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
        d.styles[name].base_style=normal;d.styles[name].font.name=FONT
    for name,size in [('Heading 1',16),('Heading 2',14),('Heading 3',13),('FrontTitle',16),('NavigationTitle',16)]:
        s=d.styles[name];s.font.size=Pt(size);s.font.bold=True
        p=s.paragraph_format;p.first_line_indent=Cm(0);p.space_before=Pt(8);p.space_after=Pt(6)
        p.keep_with_next=True;p.keep_together=True;p.line_spacing=1.2
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER if name in ['Heading 1','FrontTitle','NavigationTitle'] else WD_ALIGN_PARAGRAPH.LEFT
    for name in ['FrontTitle']:
        p=d.styles[name].element.get_or_add_pPr();n=OxmlElement('w:outlineLvl');n.set(qn('w:val'),'0');p.append(n)
    s=d.styles['FrontSubTitle'];s.font.size=Pt(14);s.font.bold=True
    p=s.paragraph_format;p.first_line_indent=Cm(0);p.alignment=WD_ALIGN_PARAGRAPH.LEFT
    p.line_spacing=1.15;p.space_before=Pt(8);p.space_after=Pt(8);p.keep_with_next=True
    for name,indent,bold in [('TOC 1',0,True),('TOC 2',.65,False)]:
        s=d.styles[name];s.font.size=Pt(12);s.font.bold=bold
        p=s.paragraph_format;p.first_line_indent=Cm(0);p.left_indent=Cm(indent);p.space_before=Pt(0);p.space_after=Pt(4)
        p.line_spacing=1.15;p.alignment=WD_ALIGN_PARAGRAPH.LEFT;p.keep_together=True
        p.tab_stops.add_tab_stop(Cm(WIDTH_CM),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
    s=d.styles['Table of Figures'];s.font.size=Pt(12);s.font.bold=False;s.font.italic=False
    p=s.paragraph_format;p.first_line_indent=Cm(0);p.space_after=Pt(5);p.line_spacing=1.15
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT;p.keep_together=True
    p.tab_stops.add_tab_stop(Cm(WIDTH_CM),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
    s=d.styles['Caption'];s.font.size=Pt(12);s.font.italic=True;s.font.bold=False
    p=s.paragraph_format;p.first_line_indent=Cm(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.space_before=Pt(5);p.space_after=Pt(8);p.line_spacing=1.15;p.keep_together=True;p.keep_with_next=False
    s=d.styles['CodeBlock'];s.font.name='Consolas';s.font.size=Pt(10)
    p=s.paragraph_format;p.first_line_indent=Cm(0);p.line_spacing=1.1;p.alignment=WD_ALIGN_PARAGRAPH.LEFT;p.space_after=Pt(8)
    s=d.styles['ReferenceEntry'];s.font.size=Pt(13)
    p=s.paragraph_format;p.left_indent=Cm(.8);p.first_line_indent=Cm(-.8);p.line_spacing=1.15;p.space_after=Pt(8);p.keep_together=True
    for name in ['Header','Footer']:
        s=d.styles[name];s.font.size=Pt(11 if name=='Header' else 12)
        s.paragraph_format.first_line_indent=Cm(0);s.paragraph_format.line_spacing=1
    d.settings.element.append(OxmlElement('w:updateFields'))
    d.settings.element[-1].set(qn('w:val'),'false')
    lang=OxmlElement('w:lang');lang.set(qn('w:val'),'vi-VN')
    d.styles['Normal'].element.get_or_add_rPr().append(lang)


def contents(d, entries, pages, instruction):
    first=None;last=None
    for e in entries:
        level=0 if e['key'].startswith(('figure_','table_')) else e.get('outline',1)
        p=d.add_paragraph(style='TOC 1' if level==1 else 'TOC 2' if level>1 else 'Table of Figures')
        if first is None:
            first=p;field_run(p,'begin');field_run(p,'instruction',instruction);field_run(p,'separate')
        # Cache both navigation and page text; Word can rebuild the native field.
        label=e.get('nav_label',e['display'])
        font(p.add_run(label),size=12,bold=level==1)
        # Word renders the dotted tab leader from the paragraph style. Literal
        # dots duplicate that leader and can push the page number onto a new line.
        p.add_run('\t');font(p.add_run(pages.get(e['key'],'0')),size=12,bold=level==1)
        last=p
    if last is not None:field_run(last,'end')


def make_table(d, rows, kind):
    columns=len(rows[0]);t=d.add_table(rows=len(rows),cols=columns)
    t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.LEFT
    if kind=='table_4_1':widths=[4.1,8.4,3.0]
    elif kind=='table_4_2':widths=[4.4,11.1]
    elif kind=='table_7_2':widths=[4.4,6.1,2.5,2.5]
    elif columns==6:widths=[4.4]+[2.22]*5
    elif kind.startswith('table_7_'):widths=[4.9,2.55,2.55,5.5]
    elif columns==4:widths=[5.5,3.3,3.3,3.4]
    elif columns==5:widths=[5.5,2.5,2.5,2.5,2.5]
    elif columns==3:widths=[3.2,5.3,7.0]
    elif columns==2:widths=[3.1,12.4]
    else:widths=[WIDTH_CM/columns]*columns
    for col,w in zip(t.columns,widths):col.width=Cm(w)
    borders=OxmlElement('w:tblBorders')
    for name in ['top','bottom','left','right','insideH','insideV']:
        n=OxmlElement('w:'+name);n.set(qn('w:val'),'single');n.set(qn('w:sz'),'4');n.set(qn('w:color'),'D9D9D9');borders.append(n)
    t._tbl.tblPr.append(borders)
    for r,values in enumerate(rows):
        tr=t.rows[r]._tr.get_or_add_trPr();tr.append(OxmlElement('w:cantSplit'))
        if r==0:tr.append(OxmlElement('w:tblHeader'))
        for c,value in enumerate(values):
            cell=t.cell(r,c);cell.width=Cm(widths[c]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
            for side in ['top','bottom','left','right']:
                n=OxmlElement('w:'+side);n.set(qn('w:w'),'80');n.set(qn('w:type'),'dxa');m.append(n)
            pr.append(m)
            if r==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E7E6E6');pr.append(shade)
            p=cell.paragraphs[0];p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.line_spacing=1.1
            p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2)
            p.paragraph_format.keep_with_next=r==0
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if c==0 or (kind=='table_7_2' and c==1) or (columns<=4 and c==columns-1 and kind.startswith('table_7_')) else WD_ALIGN_PARAGRAPH.CENTER
            if r==0 and kind.startswith('table_7_'):value=value.replace('Dice ','Dice\n')
            font(p.add_run(value),size=12,bold=r==0)
    d.add_paragraph().paragraph_format.space_after=Pt(2)


def build(output, page_map):
    blocks=parse_source();pages=json.loads(page_map.read_text(encoding='utf-8')) if page_map.is_file() else {}
    if 'pages' in pages:pages=pages['pages']
    d=Document();style_document(d);set_section(d.sections[0])
    team=json.loads((REPO/'config/team.json').read_text(encoding='utf-8'))
    assert team['team_size']==4 and team['placeholder_count']==0
    members='\n'.join(member['name']+'   '+member['student_id'] for member in team['members'])
    for text,size,before in [('BÁO CÁO ĐỒ ÁN MÔN HỌC',16,25),('CS406   XỬ LÝ ẢNH VÀ ỨNG DỤNG',16,8),
                             ('PHÂN ĐOẠN NHỊ PHÂN ẢNH RGB\nTRONG LĨNH VỰC Y TẾ',20,68),
                             ('Đối chứng màu sắc và Logistic\nCompact U-Net học từ đầu và TransUNet tham khảo',14,20),
                             ('Thành viên thực hiện',14,40),(members,14,12),
                             ('Tháng 11 năm 2026',14,40)]:
        p=d.add_paragraph(style='Title' if size==20 else 'Normal');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.space_before=Pt(before);p.paragraph_format.line_spacing=1.2
        font(p.add_run(text),size=size,bold=size>=16)
    set_section(d.add_section(WD_SECTION_START.NEW_PAGE),'lowerRoman')
    p=d.add_paragraph('MỤC LỤC',style='NavigationTitle');p.paragraph_format.page_break_before=False
    index=[];counter=1
    front=[{'key':'front_lists','display':'DANH MỤC HÌNH VÀ BẢNG','outline':1},
           {'key':'front_abbrev','display':'DANH MỤC TỪ VIẾT TẮT','outline':1}]
    headings=front+[b for b in blocks if b['kind']=='heading' and 'key' in b]
    contents(d,headings,pages,' TOC \\o "1-3" \\h \\z \\u ')
    for f in front:
        p=d.add_paragraph(f['display'],style='FrontTitle');p.paragraph_format.page_break_before=True
        bookmark(p,f['key'],counter);counter+=1;index.append(f)
        if f['key']=='front_lists':
            p=d.add_paragraph('Danh mục hình',style='FrontSubTitle');p.paragraph_format.keep_with_next=True
            contents(d,[b for b in blocks if b['kind']=='figure'],pages,' TOC \\h \\z \\c "Hình" ')
            d.add_paragraph('Danh mục bảng',style='FrontSubTitle')
            contents(d,[b for b in blocks if b['kind']=='table'],pages,' TOC \\h \\z \\c "Bảng" ')
        else:
            make_table(d,[['Từ viết tắt','Diễn giải']]+ABBREVIATIONS,'abbreviations')
    started=False;body_started=False;references=False
    for b in blocks:
        if b.get('key')=='summary':started=True
        if not started:continue
        if b['kind']=='heading':
            if b.get('key')=='chapter_1':
                set_section(d.add_section(WD_SECTION_START.NEW_PAGE),'decimal');body_started=True
            if 'key' not in b:continue
            style='FrontTitle' if b['key'] in ['summary','references'] else 'Heading '+str(b['outline'])
            p=d.add_paragraph(b['display'],style=style)
            p.paragraph_format.page_break_before=(b['outline']==1 and b['key']!='chapter_1') or b['key'] in ['heading_7_2','heading_8_2']
            bookmark(p,b['key'],counter);counter+=1;index.append(b)
            if b['key']=='references':references=True
        elif b['kind']=='paragraph':
            if references and re.match(r'^\d+\. ',b['text']):
                number,text=b['text'].split('. ',1);p=d.add_paragraph(style='ReferenceEntry')
                bookmark(p,'ref_'+number,counter);counter+=1;inline(p,'['+number+'] '+text)
            else:
                p=d.add_paragraph();inline(p,b['text'])
        elif b['kind']=='code':
            p=d.add_paragraph(style='CodeBlock');font(p.add_run(b['text']),size=10,name='Consolas')
        elif b['kind']=='table':
            p=d.add_paragraph(style='Caption');caption_text(p,b['display']);p.paragraph_format.keep_with_next=True
            if b['key']=='table_3_2':p.paragraph_format.page_break_before=True
            for run in p.runs:font(run,size=12,bold=True,italic=False)
            bookmark(p,b['key'],counter);counter+=1;index.append(b)
            make_table(d,b['rows'],b['key'])
        elif b['kind']=='figure':
            source=ROOT/'docs'/b['path'];assert source.is_file(),source
            p=d.add_paragraph();p.paragraph_format.first_line_indent=Cm(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing=1;p.paragraph_format.keep_together=True
            p.paragraph_format.keep_with_next=True;p.paragraph_format.space_after=Pt(0)
            with Image.open(source) as image:ratio=image.height/image.width
            preferred={'figure_3_1':15,'figure_7_3':14.8}.get(b['key'],WIDTH_CM)
            width=min(preferred,18/ratio)
            p.add_run().add_picture(str(source),width=Cm(width))
            p=d.add_paragraph(style='Caption');caption_text(p,b['display']);bookmark(p,b['key'],counter);counter+=1
            index.append(b)
    output.parent.mkdir(parents=True,exist_ok=True)
    d.core_properties.title='Phân đoạn nhị phân ảnh RGB trong lĩnh vực y tế'
    d.core_properties.subject='Báo cáo đồ án CS406 – Xử lý ảnh và Ứng dụng'
    d.core_properties.author='; '.join(member['name'] for member in team['members'])
    d.save(output)
    info={'format':{'page':'A4','body_font':FONT,'body_pt':13,'line_spacing':1.5,'margins_cm':{'left':3.5,'right':2,'top':2.5,'bottom':2.5}},
          'entries':[{k:b[k] for k in ['key','display','outline','nav_label'] if k in b} for b in index],
          'toc_order':[b['key'] for b in headings], 'figures':[b['key'] for b in blocks if b['kind']=='figure'],
          'tables':[b['key'] for b in blocks if b['kind']=='table']}
    (output.parent/'report-index.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'docx':str(output),'navigation_entries':len(index),'figures':len(info['figures']),'tables':len(info['tables'])}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/completion/report-build/candidate.docx')
    parser.add_argument('--pages',type=Path,default=ROOT/'output/completion/report-build/page-map.json')
    args=parser.parse_args();build(args.output.resolve(),args.pages.resolve())


if __name__=='__main__':main()
