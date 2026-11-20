"""Render, paginate, link and package the Vietnamese Word/PDF report.

Uses the bundled document runtime. --publish is used only after visual review.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unicodedata
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link

SCRIPT_DIR=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('CS406_PROJECT_DIR',SCRIPT_DIR.parent.parent/'Project')).resolve()
RUNTIME=Path(os.environ.get('CODEX_ARTIFACT_RUNTIME','C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime'))
NODE=RUNTIME/'dependencies/node/bin/node.exe'
PYTHON=RUNTIME/'dependencies/python/python.exe'


def normalize(value):
    return re.sub(r'\s+','',unicodedata.normalize('NFC',value))


def roman(n):
    result=''
    for value,letter in [(1000,'m'),(900,'cm'),(500,'d'),(400,'cd'),(100,'c'),(90,'xc'),(50,'l'),(40,'xl'),(10,'x'),(9,'ix'),(5,'v'),(4,'iv'),(1,'i')]:
        while n>=value:result+=letter;n-=value
    return result


def locate(work):
    index=json.loads((work/'report-index.json').read_text(encoding='utf-8'))
    layout=json.loads((work/'report-layout.json').read_text(encoding='utf-8'))
    locations={}
    for entry in index['entries']:
        match=[]
        for page in layout['pages']:
            for element in page.get('bodyElements',[]):
                painted=''.join(v.get('text','') for v in element.get('textLines',[]))
                if normalize(painted)==normalize(entry['display']):
                    match.append({'physical':page['index'],'frame':element['frame']})
        if len(match)!=1:raise ValueError(f"Expected one exact destination for {entry['key']}: {len(match)}")
        locations[entry['key']]=match[0]
    first_body=locations['chapter_1']['physical']
    labels={i:roman(i-1) if i<first_body else str(i-first_body+1) for i in range(2,len(layout['pages'])+1)}
    pages={key:labels[value['physical']] for key,value in locations.items()}
    reader=PdfReader(work/'candidate.pdf')
    for physical,label in labels.items():
        actual=reader.pages[physical-1].extract_text().strip().splitlines()[-1].strip()
        if actual!=label:raise ValueError(f'Footer {physical}: expected {label}, found {actual}')
    return {'pages':pages,'locations':locations,'labels':labels,'first_body':first_body,'physical_pages':len(layout['pages'])}


def add_pdf_navigation(work,mapping):
    index=json.loads((work/'report-index.json').read_text(encoding='utf-8'))
    layout=json.loads((work/'report-layout.json').read_text(encoding='utf-8'))
    reader=PdfReader(work/'candidate.pdf');writer=PdfWriter();writer.append_pages_from_reader(reader)
    writer.add_metadata({'/Title':'Phân đoạn nhị phân ảnh RGB trong lĩnh vực y tế','/Subject':'Báo cáo đồ án CS406','/Lang':'vi-VN'})
    last_chapter=None
    for entry in index['entries']:
        if 'outline' not in entry:continue
        parent=last_chapter if entry['outline']>1 else None
        item=writer.add_outline_item(entry['display'],mapping['locations'][entry['key']]['physical']-1,parent=parent)
        if entry['outline']==1:last_chapter=item
    linked=0
    for entry in index['entries']:
        dest=mapping['locations'][entry['key']]
        for page in layout['pages']:
            for element in page.get('bodyElements',[]):
                value=element.get('text','')
                label=entry.get('nav_label',entry['display'])
                if normalize(label) not in normalize(value) or normalize(value)==normalize(entry['display']):continue
                # Navigation entries include the title, dotted leader and page label.
                paragraphs=element.get('paragraphs',[])
                if not any(p.get('styleId') in ['TOC1','TOC2','TOC 1','TOC 2','ListEntry','TableofFigures','Table of Figures'] for p in paragraphs):
                    continue
                f=element['frame'];height=float(reader.pages[page['index']-1].mediabox.height)
                rect=(f['left']*.75,height-(f['top']+f['height'])*.75,(f['left']+f['width'])*.75,height-f['top']*.75)
                writer.add_annotation(page['index']-1,Link(rect=rect,target_page_index=dest['physical']-1));linked+=1
    # PDF viewers display the same Roman/Arabic page labels as the printed footers.
    writer.set_page_label(0,0,prefix='Bìa')
    writer.set_page_label(1,mapping['first_body']-2,style='/r',start=1)
    writer.set_page_label(mapping['first_body']-1,len(reader.pages)-1,style='/D',start=1)
    with (work/'final.pdf').open('wb') as handle:writer.write(handle)
    return linked


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir',type=Path,default=ROOT/'output/completion/report-build')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();work=args.workdir.resolve();work.mkdir(parents=True,exist_ok=True)
    page_file=work/'page-map.json'
    for iteration in range(1,6):
        old=json.loads(page_file.read_text(encoding='utf-8')).get('pages',{}) if page_file.exists() else {}
        subprocess.run([str(PYTHON),str(SCRIPT_DIR/'build_completion_report.py'),'--output',str(work/'candidate.docx'),'--pages',str(page_file)],cwd=ROOT,check=True)
        code="process.argv=['node','render',"+json.dumps(str(work/'candidate.docx'))+','+json.dumps(str(work/'candidate.pdf'))+"];await import("+json.dumps((SCRIPT_DIR/'render_report.mjs').as_uri())+").catch(e=>{console.error(e.stack);process.exitCode=1})"
        subprocess.run([str(NODE),'--input-type=module','-e',code],cwd=ROOT,check=True)
        current=locate(work);page_file.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f"Pagination pass {iteration}: {current['physical_pages']} physical pages, content starts at page {current['first_body']}",flush=True)
        if current['pages']==old:break
    else:raise RuntimeError('Table of contents pagination did not stabilize')
    linked=add_pdf_navigation(work,current)
    receipt={'format':'A4, Times New Roman 13, 1.5 lines, 3.5/2/2.5/2.5 cm margins',
             'physical_pages':current['physical_pages'],'first_body':current['first_body'],
             'navigation_targets':len(current['pages']),'pdf_navigation_links':linked,
             'toc_pages_verified':True,'footer_numbers_verified':True,'iterations':iteration}
    (work/'validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.publish:
        source_manifest=json.loads((ROOT/'output/completion/report-source-manifest.json').read_text(encoding='utf-8'))
        if source_manifest.get('draft') is not False:
            raise ValueError('A private layout draft cannot be published as the final course report')
        destination=ROOT/'deliverables/completion-20261119'
        destination.mkdir(parents=True,exist_ok=True)
        for name in ['CS406_Report.docx','CS406_Report.pdf']:
            if (destination/name).exists():raise FileExistsError('Use a new version directory; preserve published report')
        shutil.copyfile(work/'candidate.docx',destination/'CS406_Report.docx')
        shutil.copyfile(work/'final.pdf',destination/'CS406_Report.pdf')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
