/** Fifteen main slides and four appendices, from verified final measurements. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import './artifact_runtime_compat.mjs';

const ROOT=path.resolve(process.env.CS406_PROJECT_DIR||path.join(path.dirname(fileURLToPath(import.meta.url)),'../../Project'));
const inputPath=process.argv[2],buildPath=process.argv[3];
if(!inputPath||!buildPath)throw new Error('Usage: build_completion_presentation.mjs evidence.json new-build-directory');
const BUILD=path.resolve(buildPath);
if(!BUILD.startsWith(path.join(ROOT,'output')+path.sep))throw new Error('Build must stay in Project/output');
try{await fs.access(BUILD);throw new Error('Build directory already exists');}catch(error){if(error.code!=='ENOENT')throw error;}
const data=JSON.parse(await fs.readFile(inputPath,'utf8'));
if(data.format!=='cs406-defense-input-v1'||data.draft!==false)throw new Error('Final verified evidence is required');
for(const [filename,expected] of Object.entries({...data.evidence_sha256,...data.asset_sha256})){
  if(createHash('sha256').update(await fs.readFile(filename)).digest('hex')!==expected)throw new Error('Evidence changed: '+filename);
}
const RUNTIME=process.env.CODEX_ARTIFACT_RUNTIME||'C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime';
const MODULES=path.join(RUNTIME,'dependencies/node/node_modules');
process.env.RUNTIME_NODE_MODULES=MODULES;
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(MODULES,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const SKILL='C:/Users/Admin/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
await fs.mkdir(BUILD,{recursive:true});
const deck=Presentation.create({slideSize:{width:1280,height:720}}),tables=[],charts=[],outline=[];
const C={bg:'#FCFCFD',ink:'#182B3A',muted:'#617180',line:'#DCE2E8',baseline:'#7D8B9B',pixel:'#B5822C',compact:'#305E9C',trans:'#0D7F79'};
const PRIMARY=data.primary,full=data.local.methods[PRIMARY],frozen=data.runs[PRIMARY].frozen;
const NAMES={baseline:'LAB–Otsu',pixel_logistic:'Logistic pixel',[PRIMARY]:'Compact U-Net',transunet:'TransUNet tham khảo'};
const order=['baseline','pixel_logistic',PRIMARY,'transunet'];
const ablations=['bce_plain_s406','bce_dice_s406','bce_aug_s406',PRIMARY];
const short={bce_plain_s406:'BCE',bce_dice_s406:'BCE + Dice',bce_aug_s406:'BCE + aug.',[PRIMARY]:'BCE + Dice + aug.',bce_dice_aug_s407:'Đầy đủ · seed 407',bce_dice_aug_s408:'Đầy đủ · seed 408'};
const dec=(n,d=4)=>n.toFixed(d).replace('.',','),pct=(n,d=1)=>dec(n*100,d)+'%';
const localScore=(name,key='dice')=>data.local.methods[name].macro.mean[key];
const externalScore=(name,key='dice')=>data.external.methods[name].macro.mean[key];
const sources='results/completion: frozen-selection.json; evaluation/per_image.csv; analysis.json; case-manifest.json. Hash tệp nguồn nằm trong presentation-input.json.';
function txt(s,value,x,y,w,h,size=27,style={}){
  const obj=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  obj.text=value;obj.text.insets=0;obj.text.style={fontFamily:'Arial',fontSize:size,color:C.ink,...style};return obj;
}
function slide(section,title,talk,source=sources){
  const s=deck.slides.add();s.background.fill=C.bg;
  txt(s,section.toUpperCase(),64,31,1152,25,16,{color:C.muted,bold:true});
  txt(s,title,64,77,1152,100,42,{bold:true});
  s.shapes.add({geometry:'rect',position:{left:64,top:662,width:1152,height:1},fill:C.line,line:{fill:'none',width:0}});
  txt(s,'CS406   Xử lý ảnh và Ứng dụng',64,678,970,23,15,{color:C.muted});
  const n=deck.slides.count;txt(s,n<=15?`${String(n).padStart(2,'0')} / 15`:`Phụ lục ${n-15} / 4`,1065,678,151,23,15,{color:C.muted,alignment:'right'});
  const notes=talk+'\n\nNguồn: '+source;s.speakerNotes.text=notes;outline.push({section,title,notes});return s;
}
async function img(s,filename,x,y,w,h,alt){
  const location=path.isAbsolute(filename)?filename:path.join(ROOT,filename),bytes=await fs.readFile(location);
  s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',position:{left:x,top:y,width:w,height:h},fit:'contain',alt});
}
function table(s,values,y=190,height=320,widths=[380,772],size=23){
  const obj=s.tables.add({rows:values.length,columns:values[0].length,left:64,top:y,width:1152,height,columnWidths:widths,values});
  obj.borders.assign({fill:C.line,width:1,style:'solid'});
  obj.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({textStyle:{fontFamily:'Arial',fontSize:size,color:C.ink},margins:{top:8,bottom:8,left:14,right:14}});
  values.forEach((row,r)=>row.forEach((_,c)=>{const cell=obj.getCell(r,c);cell.fill=r===0?C.ink:'#FFFFFF';cell.text.style={fontFamily:'Arial',fontSize:size,color:r===0?'#FFFFFF':C.ink,bold:r===0};}));
  tables.push(deck.slides.count);return obj;
}
function bar(s,score,y=190,h=350){
  const names=order.map(n=>NAMES[n]),colors=[C.baseline,C.pixel,C.compact,C.trans];
  const obj=s.charts.add('bar',{position:{left:64,top:y,width:1152,height:h},categories:['Dice','IoU'],
    series:order.map((name,i)=>({name:names[i],values:[score(name,'dice'),score(name,'iou')],fill:colors[i]})),
    hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{typeface:'Arial',fontSize:22,fill:C.ink}},
    titlePlacement:'none',chartFill:C.bg,plotAreaFill:C.bg,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},
    barOptions:{direction:'column',grouping:'clustered',gapWidth:100},
    xAxis:{textStyle:{typeface:'Arial',fontSize:23,fill:C.ink},line:{fill:C.line,width:1}},
    yAxis:{min:0,max:1,majorUnit:.2,numberFormatCode:'0%',textStyle:{typeface:'Arial',fontSize:21,fill:C.muted},majorGridlines:{fill:C.line,width:1}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0%',textStyle:{typeface:'Arial',fontSize:20,fill:C.ink}}});
  order.forEach((name,i)=>['dice','iou'].forEach((key,j)=>{obj.series.getItemAt(i).dataLabelOverrides.add(j).text=pct(score(name,key));}));
  applyPresentationChartFont(obj,{fontFamily:'Arial'});charts.push(deck.slides.count);
}
function line(s,metric,x,w,max,format){
  const history=data.runs[PRIMARY].history;
  const obj=s.charts.add('line',{position:{left:x,top:230,width:w,height:280},categories:history.map(row=>String(row.epoch)),
    series:[{name:metric==='train_loss'?'Train loss':'Valid Dice',values:history.map(row=>row[metric]),line:{style:'solid',fill:C.compact,width:3},marker:{symbol:'none'}}],
    hasLegend:false,chartFill:C.bg,plotAreaFill:C.bg,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},
    lineOptions:{grouping:'standard',smooth:false},xAxis:{textStyle:{typeface:'Arial',fontSize:17,fill:C.muted},line:{fill:C.line,width:1}},
    yAxis:{min:0,max,numberFormatCode:format,textStyle:{typeface:'Arial',fontSize:20,fill:C.muted},majorGridlines:{fill:C.line,width:1}}});
  applyPresentationChartFont(obj,{fontFamily:'Arial'});charts.push(deck.slides.count);
}
function note(s,value,y=575,size=24){txt(s,value,64,y,1152,76,size,{color:C.muted});}

{
  const s=slide('Đồ án môn học','', 'Giới thiệu bài toán phân đoạn foreground, không chẩn đoán bệnh. Bốn tên trên bìa là thành viên đã được xác nhận. Khoảng 12 phút cho 15 slide chính; phụ lục dùng khi hỏi đáp.');
  txt(s,'Phân đoạn nhị phân\ntrên ảnh RGB',64,158,635,155,55,{bold:true});
  txt(s,'Quy tắc màu · học pixel\nCompact U-Net · TransUNet',68,334,580,90,28,{color:C.muted});
  txt(s,data.team.members.map(row=>`${row.name}   ${row.student_id}`).join('\n'),68,458,600,150,22);
  await img(s,'output/presentation-evidence/worked_source.png',730,220,220,220,'Ảnh RGB thật');
  await img(s,'output/presentation-evidence/worked_reference.png',978,220,220,220,'Mask tham chiếu của cùng ảnh');
  txt(s,'RGB',730,465,220,35,23);txt(s,'Foreground',978,465,220,35,23);outline[0].title='Phân đoạn nhị phân trên ảnh RGB';
}
{
  const s=slide('01   Bài toán','Đầu vào, đầu ra và mục tiêu', 'Mô hình chỉ nhận ảnh RGB. Mask chỉ phục vụ học hoặc đánh giá. Ví dụ không coi toàn bộ bàn tay là foreground; phải bám vùng được gán nhãn. Không dùng query hoặc nội dung lâm sàng bằng chữ.');
  await img(s,'output/presentation-evidence/worked_source.png',64,200,300,300,'Ảnh bàn tay');
  await img(s,'output/presentation-evidence/worked_reference.png',398,200,300,300,'Vùng foreground được gán nhãn');
  txt(s,'RGB',64,518,300,40,25);txt(s,'Mask nhị phân',398,518,300,40,25);
  txt(s,'Mỗi pixel: vùng quan tâm hoặc nền',760,210,450,100,33,{bold:true});
  txt(s,'So sánh độ chồng lấp, ranh giới và các vùng dự đoán sai',760,376,450,155,30);
  note(s,'Dice và IoU đo sự phù hợp với nhãn, không đo độ chính xác chẩn đoán.');
}
{
  const s=slide('01   Dữ liệu','Nguồn ảnh và cách chia cục bộ', 'Đối chiếu đầy đủ ID, kích thước byte và CRC32 với ZIP ảnh DermaVQA. Train cộng test cục bộ chính xác bằng tập train nguồn. Không có 314 ảnh test chính thức. Vì vậy đây là benchmark nội bộ và không tái lập leaderboard. Mask train/valid và lịch sử checkpoint chưa xác minh đầy đủ.', 'results/completion/provenance/comparison_summary.json; https://osf.io/p8bfu; https://github.com/wyim/ImageCLEF-MAGIC-2025');
  table(s,[['Tập local','Ảnh / mask','Quan hệ ảnh nguồn','Vai trò'],['Train','2.179 / 2.179','Một phần train nguồn','Cập nhật mô hình'],['Validation','157 / 157','Toàn bộ valid nguồn','Chọn epoch, ngưỡng'],['Test','295 / 885','Phần còn lại train nguồn','Đánh giá đã đóng băng']],195,300,[180,220,420,332]);
  note(s,'2.631 ảnh khớp CRC32; 0 trùng byte hoặc encounter xuyên split.\nEncounter chưa phải bệnh nhân đã được xác minh.',540);
}
{
  const s=slide('01   Nhãn','Đồng thuận đa số nghiêm ngặt', 'Hiển thị ba annotation thật. Một pixel là foreground khi hơn một nửa mask bỏ phiếu; hòa phiếu là nền. Sáu ảnh có hai hoặc bốn annotation, nên quy tắc hòa khác mã chấm chính thức. Không coi mọi annotator là bác sĩ da liễu.');
  for(let i=0;i<4;i++){
    const x=64+i*292;txt(s,i<3?`Annotation ${i+1}`:'Đồng thuận',x,195,270,42,25,{bold:true});
    await img(s,i<3?`output/presentation-evidence/annotation-${i+1}.png`:'output/presentation-evidence/worked_reference.png',x,255,270,270,'Mask thật, cùng một ảnh');
  }
  note(s,'Nhãn → nearest resize 256×256 → bỏ phiếu; ảnh test không đi vào predictor.',575);
}
{
  const s=slide('02   Phương pháp','Đối chứng LAB–Otsu', 'Đối chứng dùng tương phản màu CIELAB và ngưỡng Otsu, sau đó hình thái học. Các tham số thực tế theo BASELINE_PROTOCOL; không chọn ngưỡng từ nhãn test. Đây là đối chứng truyền thống cần đánh giá cả khi mô hình học không cải thiện.','backend/baseline.py; N. Otsu (1979), doi:10.1109/TSMC.1979.4310076');
  await img(s,'output/presentation-evidence/worked_source.png',64,205,270,270,'Ảnh RGB');
  await img(s,'output/presentation-evidence/baseline_before_morphology.png',366,205,270,270,'Mask trước hình thái');
  txt(s,'RGB',64,495,270,45,24);txt(s,'Sau ngưỡng Otsu',366,495,270,60,24);
  txt(s,'RGB → CIELAB\nTương phản màu\nOtsu → hình thái\nMask nhị phân',725,213,490,255,31);
  note(s,'Không học trọng số; giữ nguyên quy trình cho test local và ISIC.');
}
{
  const s=slide('02   Phương pháp','Hai phương pháp học tham khảo', 'Logistic được nhóm huấn luyện từ train cục bộ và chọn ngưỡng bằng valid. TransUNet dùng trọng số cung cấp, chuẩn hóa unit đã chọn từ valid; không biết lịch sử huấn luyện. Không gán checkpoint này cho đội vô địch hoặc so điểm như cùng giao thức.');
  table(s,[['Phương pháp','Tham số / lựa chọn','Phạm vi diễn giải'],['Logistic pixel','Huấn luyện 2 epoch, seed 406\nNgưỡng valid 0,30','Mô hình học mới trên train'],['TransUNet R50–ViT-B/16','Trọng số đã cung cấp\nChuẩn hóa [0; 1], ngưỡng 0,50','Kết quả mô tả; lịch sử exposure chưa biết']],195,310,[300,450,402]);
  note(s,'Logistic học quan hệ đặc trưng pixel; TransUNet kết hợp encoder CNN/Transformer.');
}
{
  const s=slide('02   Phương pháp','Compact U-Net khởi tạo từ đầu', 'Mô hình mới có 274849 tham số, không tải pretrain. Ba mức xuống mẫu và skip concatenation giữ thông tin vị trí. GroupNorm hỗ trợ batch nhỏ. Đây là một biến thể gọn, không đồng nhất mô hình U-Net gốc.','backend/compact_model.py; U-Net arXiv:1505.04597; Group Normalization arXiv:1803.08494');
  table(s,[['Thành phần','Kích thước / phép toán'],['Encoder → bottleneck','12 → 24 → 48 → 96 kênh; 3 mức xuống mẫu'],['Mỗi block','2 × (Conv 3×3 + GroupNorm + SiLU)'],['Decoder','Bilinear upsample + nối skip; 48 → 24 → 12'],['Đầu ra','Conv 1×1 → sigmoid; 274.849 tham số']],190,335,[330,822]);
  note(s,'RGB → 256×256 → 128×128 khi học; xác suất → 256×256 khi chấm.');
}
{
  const s=slide('03   Thực nghiệm','Ngân sách cố định và ablation', 'Bốn cấu hình tại seed 406 tách loss và augmentation, thêm seed 407 và 408 cho tổ hợp đầy đủ. Mọi run dùng đủ train/valid trong 20 epoch, AdamW cùng learning rate. Checkpoint chọn theo valid Dice tại 128×128, sau đó tìm ngưỡng trên valid 256×256; test không chọn mô hình.');
  table(s,[['Seed 406','Không tăng cường','Có tăng cường'],['BCE','BCE','BCE + augmentation'],['BCE + Dice','BCE + Dice','Recipe chính; thêm seed 407, 408']],195,232,[260,400,492]);
  txt(s,'20 epoch · batch 16 · AdamW lr 0,001',64,460,1152,45,30,{bold:true});
  note(s,'Train: flip, xoay 90°, brightness/contrast. Valid: chọn epoch và ngưỡng.\nRecipe chính được khai báo trước; không chọn seed tốt nhất bằng test.',536);
}
{
  const s=slide('03   Huấn luyện','Đường học của recipe chính', `Đường cong từng epoch trích từ history được băm trong frozen manifest. Chọn epoch ${frozen.best_epoch} bằng valid128, sau đó chọn ngưỡng ${frozen.threshold} trên valid256. Hai bước có độ phân giải khác nhau nên không đọc đường valid128 như điểm threshold-search256.`);
  txt(s,'Loss trên train',64,190,550,40,27,{bold:true});txt(s,'Dice validation · 128×128',670,190,546,40,27,{bold:true});
  line(s,'train_loss',64,550,Math.ceil(Math.max(...data.runs[PRIMARY].history.map(r=>r.train_loss))*10)/10,'0.0');
  line(s,'valid_dice',670,546,1,'0%');
  note(s,`Đủ 20 epoch · checkpoint epoch ${frozen.best_epoch} · ngưỡng valid ${dec(frozen.threshold,2)}\nDice valid tại 256×256 sau tìm ngưỡng: ${dec(frozen.valid_dice)}.`,548);
}
{
  const s=slide('03   Kết quả','295 ảnh test cục bộ', 'Mỗi ảnh có trọng số bằng nhau. Tất cả phương pháp chấm cùng nhãn strict-majority256. Compact dùng recipe406 đã khai báo; không thay bằng seed cótestcao hơn. TransUNet chỉ là tham khảo do lịch sử chưa biết.');
  bar(s,localScore);const ci=data.analysis.comparisons[PRIMARY+'-minus-baseline'].dice;
  note(s,`Compact − LAB: ΔDice ${dec(ci.delta)}; CI bootstrap 95% [${dec(ci.ci95[0])}; ${dec(ci.ci95[1])}].\nLấy mẫu lại encounter; so sánh thăm dò, không kết luận nhân quả.`,550,23);
}
{
  const s=slide('03   Kết quả','Ablation và biến thiên giữa ba seed', 'Bảng công bố cả bốn tổ hợp cùng ngân sách. Không mặc định mọi thành phần đều cải thiện. Mean và độ lệch chuẩn mẫu của seed 406, 407, 408 mô tả biến thiên run trên cùng benchmark, không phải ba bộ dữ liệu độc lập.');
  table(s,[['Cấu hình seed 406','Epoch chọn','Ngưỡng','Dice test','IoU test'],...ablations.map(n=>[short[n],String(data.runs[n].frozen.best_epoch),dec(data.runs[n].frozen.threshold,2),dec(localScore(n)),dec(localScore(n,'iou'))])],190,315,[400,190,170,196,196]);
  const scores=data.analysis.seed_variation.scores;
  note(s,`Ba seed, recipe đầy đủ: Dice ${dec(scores.dice.mean)} ± ${dec(scores.dice.sample_std)} (SD mẫu).\nĐây là độ biến thiên trên cùng 295 ảnh, không phải CI tổng thể.`,550);
}
{
  const s=slide('03   Ngoài nguồn','100 ảnh ISIC · không tinh chỉnh', 'ISIC 2018 Task 1 validation được dùng như bộ stress test ngoài nguồn. Mọi trọng số, ngưỡng và chuẩn hóa giữ nguyên. Dermoscopy và nhãn tổn thương khác ảnh patient-generated nên phải giới hạn diễn giải. Với trọng số TransUNet lịch sử không biết, chưa xác nhận đây là ảnh chưa từng thấy.','https://challenge.isic-archive.com/data/#2018; results/completion/external-overlap/summary.json; '+sources);
  bar(s,externalScore);
  note(s,'0 trùng byte hoặc ứng viên pHash/dHash với toàn bộ 2.631 ảnh local.\nKhác miền ảnh và nhãn; điểm ISIC không phải kiểm định lâm sàng.',550,23);
}
{
  const s=slide('04   Phân tích lỗi','Mặt nạ thật và vị trí dự đoán sai', 'Hai ví dụ được chọn bằng quy tắc xếp hạng sau chấm: Dice thấp nhất và chênh lệch lớn nhất so với LAB. Nếu cùng mẫu, dùng ví dụ khác trong danh sách đã chọn. Mask đã lưu được kiểm tra counts và sáu metrics với CSV. Xanh lục là TP, đỏ là FP, xanh lam là FN; không tự gán nguyên nhân vì màu da hay ánh sáng.');
  const selected=[data.cases.cases.find(c=>c.selection_reasons.includes('worst_dice'))];
  selected.push(data.cases.cases.find(c=>c.selection_reasons.includes('largest_primary_minus_baseline')&&c.id!==selected[0].id)||data.cases.cases.find(c=>c.id!==selected[0].id));
  if(selected.some(c=>!c))throw new Error('Two distinct final error cases are required');
  ['RGB','Nhãn','Compact U-Net','TP / FP / FN'].forEach((v,i)=>txt(s,v,64+i*292,184,270,35,23,{bold:true}));
  for(const [i,row] of selected.entries()){
    const y=222+i*208;txt(s,`${row.id} · Dice ${dec(row.primary_dice)} · LAB ${dec(row.baseline_dice)}`,64,y,1152,32,21);
    for(const [j,role] of ['source','reference','primary','errors'].entries())await img(s,path.join(data.case_dir,row.assets[role]),64+j*292,y+35,270,150,`${row.id}: ${role}`);
  }
  txt(s,'TP xanh lục · FP đỏ · FN xanh lam. Ví dụ chọn sau đánh giá.',64,633,1152,25,18,{color:C.muted});
}
{
  const s=slide('04   Ứng dụng','Demo dự đoán thật bằng bốn phương pháp', 'Mở ứng dụng và chạy Compact trên ảnh validation hoặc ảnh upload. Dùng ngưỡng đã chọn từ valid, hiển thị mask gốc, overlay và TP/FP/FN khi có nhãn. Ảnh upload không có nhãn thì không hiện metrics. Thiếu model báo lỗi rõ ràng. Ảnh màn hình là model 20 epoch trên valid; phép chấm tại kích thước gốc khác benchmark 256×256.');
  await img(s,data.demo_screenshot,64,183,1152,405,'Ứng dụng thực tế với Compact U-Net đã huấn luyện20epoch');
  note(s,'12 kiểm tra trình duyệt · upload · PNG gốc · chọn annotation · sáu kích thước màn hình.',604,22);
}
{
  const s=slide('05   Kết luận','Kết quả, giới hạn và khả năng tái lập', 'Chốt bằng số thật của recipe chính và giới hạn. Không hứa điểm chấm hoặc hiệu quả chẩn đoán. Mở source/config/history/checksum và bằng chứng số liệu khi được hỏi. Test cục bộ là cách chia nội bộ, nguồn mask và lịch sử checkpoint chưa đủ xác minh, đánh giá ngoài nguồn còn khác miền.');
  txt(s,`Compact U-Net: Dice local ${dec(localScore(PRIMARY))}`,64,195,1152,55,35,{bold:true});
  txt(s,`ISIC không tinh chỉnh: Dice ${dec(externalScore(PRIMARY))}`,64,270,1152,50,30);
  txt(s,'Đã bổ sung',64,368,530,40,27,{bold:true});
  txt(s,'Học từ đầu · ablation · ba seed\nCI ghép cặp · stress test\nDemo thật · cài đặt sạch',64,425,530,155,27);
  txt(s,'Giới hạn còn lại',695,368,520,40,27,{bold:true});
  txt(s,'Test đã được xem ở giai đoạn trước\nMask gốc và lịch sử TransUNet\nKhác miền ảnh, chưa kiểm định lâm sàng',695,425,520,155,26);
}
{
  const s=slide('Phụ lục A','Giao thức chấm và khoảng tin cậy', 'Macro và micro trả lời câu hỏi khác nhau. BF1 dùng dung sai Euclid 2 pixel. Hòa phiếu là nền. Nếu cả hai mask rỗng, overlap bằng 1; tham chiếu rỗng nhưng dự đoán không rỗng thì overlap bằng 0. Bootstrap ghép cặp encounter, 2.000 lượt seed 406, không chứng minh độc lập bệnh nhân.');
  table(s,[['Mục','Quy tắc'],['Tham chiếu','256×256, nearest resize; foreground nếu > n/2 phiếu'],['Macro / micro','Trung bình từng ảnh / cộng TP, FP, FN trước tính'],['Boundary F1','Dung sai Euclid 2 pixel tại lưới 256×256'],['Bootstrap','2.000 lượt ghép cặp theo encounter, seed 406'],['Phạm vi','So sánh thăm dò; test hồi cứu; không hiệu chỉnh đa kiểm định']],190,385,[300,852]);
}
{
  const s=slide('Phụ lục B','Cấu hình học và lựa chọn', 'AdamW cùng ngân sách, clipping L2 norm 1, GroupNorm dùng số nhóm bằng gcd(4, số kênh). Source snapshot và hash được lưu theo run. CPU dùng deterministic algorithms nghiêm ngặt; CUDA dùng chế độ cảnh báo cho kernel bilinear đã công bố, không hứa CPU và GPU có cùng hash.');
  table(s,[['Tham số','Giá trị'],['Tối ưu','AdamW lr 0,001; weight decay 0,0001; clip norm 1'],['Ngân sách','20 epoch × 2.179 train; batch 16; 157 valid mỗi epoch'],['Augmentation','H/V flip 0,5; xoay k×90°; brightness/contrast [0,85; 1,15]'],['Checkpoint','Valid Dice tại 128×128, ngưỡng 0,50; hòa chọn epoch sớm'],['Ngưỡng','Valid 256×256: 0,30..0,70 bước 0,05; hòa chọn nhỏ nhất']],190,385,[285,867],22);
}
{
  const s=slide('Phụ lục C','Sáu lần chạy và lựa chọn đã đóng băng', 'Công bố cả sáu run, epoch và ngưỡng chọn valid. Bốn run còn lại được chia qua hai nhóm chạy đồng thời, tổng 8 luồng CPU. Một run được resume sau 1 epoch với cùng cấu hình, optimizer và hash nguồn/dữ liệu. Không trộn trọng số GPU hoặc thay recipe chính bằng seed có test cao hơn.');
  table(s,[['Run','Epoch','Ngưỡng','Dice valid','Dice test'],...Object.keys(data.runs).map(n=>[short[n],String(data.runs[n].frozen.best_epoch),dec(data.runs[n].frozen.threshold,2),dec(data.runs[n].frozen.valid_dice),dec(localScore(n))])],190,390,[400,150,180,211,211],22);
}
{
  const s=slide('Phụ lục D','Nguồn tham khảo và bằng chứng', 'Mỗi slide có speaker notes và nguồn bằng chứng. Ảnh khớp kho DermaVQA; chưa suy ra biến đổi mask hoặc checkpoint nhất định thuộc đội vô địch. Mã nguồn công khai; tài nguyên ảnh và checkpoint theo điều kiện tương ứng.');
  const refs=[['Thuật toán','Otsu (1979), doi:10.1109/TSMC.1979.4310076'],['Kiến trúc','U-Net 1505.04597 · GroupNorm 1803.08494\nTransUNet 2102.04306 · V-Net 1606.04797'],['Dữ liệu','DermaVQA: github.com/velvinnn/DermaVQA; osf.io/p8bfu\nMEDIQA-MAGIC 2025; DermaVQA-DAS 2512.24340v2'],['Ngoài nguồn','ISIC 2018 Task 1 validation · challenge.isic-archive.com/data'],['Mã và kết quả','github.com/Whats-up-pro/cs406-segmentation-lab\nConfig, history, frozen manifest, CSV từng ảnh và mask PNG']];
  refs.forEach(([label,value],i)=>{txt(s,label,64,190+i*86,240,75,23,{bold:true});txt(s,value,324,190+i*86,892,75,23);});
}
if(deck.slides.count!==19)throw new Error('Expected 15 main slides and 4 appendices');
await fs.writeFile(path.join(BUILD,'slide-outline.json'),JSON.stringify(outline,null,2));
await fs.writeFile(path.join(BUILD,'input-sha256.txt'),createHash('sha256').update(await fs.readFile(inputPath)).digest('hex')+'\n');
const candidate=path.join(BUILD,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidate);
await fs.mkdir(path.join(BUILD,'published'));
await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:path.join(BUILD,'published/CS406_Defense.pptx'),
  pythonExecutable:path.join(RUNTIME,'dependencies/python/python.exe'),
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:19,requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:[...new Set(charts)],
  nativeChartTargetApplication:'portable',materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')});
await fs.writeFile(path.join(BUILD,'CS406_Defense.pdf'),new Uint8Array(await(await deck.export({format:'pdf'})).arrayBuffer()));
const pages=[];
for(const [i,s] of deck.slides.items.entries()){
  const bytes=Buffer.from(await(await deck.export({slide:s,format:'png',scale:1.25})).arrayBuffer());
  await fs.writeFile(path.join(BUILD,`slide-${String(i+1).padStart(2,'0')}.png`),bytes);
  pages.push({...outline[i],image:bytes.toString('base64')});
}
const escape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const html=`<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CS406 Phân đoạn nhị phân</title><style>*{box-sizing:border-box}body{margin:0;background:#172b38;color:white;font:16px Arial,sans-serif}main{height:calc(100dvh - 64px);display:grid;place-items:center}.slide{display:none;width:min(100vw,calc((100dvh - 64px)*16/9));max-width:100%;max-height:100%;aspect-ratio:16/9}.slide.active{display:block}.slide img{width:100%;height:100%;object-fit:contain}nav{height:64px;display:flex;align-items:center;justify-content:center;gap:14px;padding:0 10px}button,select{font:inherit;background:#294657;color:white;border:1px solid #637d8c;border-radius:4px;padding:8px 12px;cursor:pointer}button:focus-visible,select:focus-visible{outline:3px solid #76c9c2}select{max-width:35vw}#notes{position:fixed;bottom:76px;left:6%;width:88%;max-height:46dvh;overflow:auto;background:white;color:#182b3a;padding:24px;white-space:pre-wrap;line-height:1.6}#notes[hidden]{display:none}.sr{position:absolute;left:-10000px}@media(max-width:700px){nav{gap:6px}button,select{font-size:13px;padding:8px 7px}#fullscreen{display:none}}@media print{main{display:block;height:auto}.slide{display:block;width:100%;height:100vh;page-break-after:always}nav,#notes{display:none}}</style><main>${pages.map((v,i)=>`<section class="slide${i===0?' active':''}" aria-label="${i+1}. ${escape(v.title)}"><h1 class="sr">${escape(v.title)}</h1><img src="data:image/png;base64,${v.image}" alt="${escape(v.title)}"></section>`).join('')}</main><nav aria-label="Điều khiển trình chiếu"><button id="prev" aria-label="Trang trước">←</button><span id="counter" role="status"></span><button id="next" aria-label="Trang sau">→</button><select id="slide-picker" aria-label="Chọn nội dung">${pages.map((v,i)=>`<option value="${i}">${i+1}. ${escape(v.title)}</option>`).join('')}</select><button id="note-toggle">Ghi chú N</button><button id="fullscreen">Toàn màn hình</button></nav><aside id="notes" hidden></aside><script>const slides=[...document.querySelectorAll('.slide')],notes=${JSON.stringify(pages.map(v=>v.notes)).replace(/</g,'\\u003c')};let index=0;function show(n){index=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('active',i===index));document.querySelector('#counter').textContent=(index+1)+' / '+slides.length;document.querySelector('#slide-picker').value=String(index);document.querySelector('#notes').textContent=notes[index];location.hash=String(index+1)}document.querySelector('#prev').onclick=()=>show(index-1);document.querySelector('#next').onclick=()=>show(index+1);document.querySelector('#slide-picker').onchange=e=>show(Number(e.target.value));document.querySelector('#note-toggle').onclick=()=>document.querySelector('#notes').hidden=!document.querySelector('#notes').hidden;document.querySelector('#fullscreen').onclick=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(index+1)}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1)}else if(e.key==='Home')show(0);else if(e.key==='End')show(slides.length-1);else if(e.key.toLowerCase()==='n')document.querySelector('#note-toggle').click()});show(Number(location.hash.slice(1))-1||0);</script></html>`;
await fs.writeFile(path.join(BUILD,'CS406_Presentation.html'),html);
console.log(JSON.stringify({build:BUILD,slides:19,main:15,appendix:4,tables,charts:[...new Set(charts)],published:false}));
