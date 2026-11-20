/* Buildless, local-only workspace. All model results come from the Python API. */
"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const icon = (name, cls = "") => `<svg class="icon ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const number = (value) => new Intl.NumberFormat("vi-VN").format(value);
const percent = (value) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const splitName = {train: "Training", valid: "Validation", test: "Test"};
const methodName = {baseline: "LAB + Otsu", transunet: "TransUNet", pixel_logistic: "Logistic pixel", compact_unet: "Compact U-Net",
  bce_plain_s406: "Compact · BCE · seed 406", bce_dice_s406: "Compact · BCE + Dice · seed 406",
  bce_aug_s406: "Compact · BCE + tăng cường · seed 406", bce_dice_aug_s406: "Compact · BCE + Dice + tăng cường · seed 406",
  bce_dice_aug_s407: "Compact · tổ hợp đầy đủ · seed 407", bce_dice_aug_s408: "Compact · tổ hợp đầy đủ · seed 408"};
const state = {split: "test", page: 1, limit: 6, query: "", total: 0, summary: null, source: null,
  detail: null, annotation: 0, mode: "overlay", layer: "reference", prediction: null,
  busy: false, loading: false, listSequence: 0, selectionSequence: 0, predictionSequence: 0, objectURL: null, view: "workspace"};

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Không thể hoàn tất yêu cầu (HTTP ${response.status}).`;
    try { const body = await response.json(); if (typeof body.detail === "string") message = body.detail; } catch { /* Keep HTTP message. */ }
    throw new Error(message);
  }
  return response.json();
}

let toastTimer;
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 5000);
}
function error(message) { $("#app-error").textContent = message; $("#app-error").hidden = !message; }

function baseURL(source = state.source) {
  return `/api/samples/${source.split}/${encodeURIComponent(source.id)}`;
}
function annotationInfo() { return state.annotation === "consensus" ? state.detail?.consensus : state.detail?.annotations?.[state.annotation]; }
function referenceURL(mode = "overlay", download = false) {
  return `${baseURL()}/mask?annotation=${state.annotation}&mode=${mode}&max_size=${download ? 0 : 1400}${download ? "&download=true" : ""}`;
}
function updateControls() {
  $("#predict-button").disabled = !state.source || state.loading || state.busy;
  $("#predict-button span").textContent = state.busy ? "Đang phân đoạn…" : "Chạy phân đoạn";
  $(".inspector").classList.toggle("is-busy", state.busy);
  $("#download-button").disabled = state.loading || (state.layer === "prediction" ? !state.prediction : !annotationInfo() || Boolean(annotationInfo().error));
  $("#annotation-select").disabled = !state.detail?.annotations?.length || state.loading;
  $("#layer-source option[value=prediction]").disabled = !state.prediction;
  $("#opacity").disabled = !["overlay", "errors"].includes(state.mode);
  $('[data-mode=errors]').disabled = !state.prediction?.errors;
  $("#threshold").disabled = $("#method-select").value === "baseline";
  $("#image-stage").setAttribute("aria-busy", String(state.loading));
}

function resetPrediction() {
  state.prediction = null;
  if (state.mode === "errors") state.mode = "overlay";
  state.layer = "reference";
  $("#layer-source").value = "reference";
  $("#prediction-result").hidden = true;
  $("#result-placeholder").hidden = false;
  $("#result-status").textContent = "CHƯA CHẠY";
  updateControls();
}

function resizeFrame() {
  const detail = state.detail;
  if (!detail?.width || !detail?.height) return;
  const stage = $("#image-stage");
  const scale = Math.min((stage.clientWidth - 38) / detail.width, (stage.clientHeight - 36) / detail.height);
  $("#image-frame").style.width = `${Math.max(1, detail.width * scale)}px`;
  $("#image-frame").style.height = `${Math.max(1, detail.height * scale)}px`;
}
new ResizeObserver(resizeFrame).observe($("#image-stage"));

function renderLayer() {
  const mask = $("#mask-image");
  const original = $("#original-image");
  const annotation = annotationInfo();
  const hasMask = state.layer === "prediction" ? Boolean(state.prediction) : Boolean(annotation && !annotation.error);
  const binary = state.mode === "mask";
  const errors = state.mode === "errors" && state.prediction?.errors;
  original.style.visibility = binary && hasMask ? "hidden" : "visible";
  mask.hidden = !hasMask || state.mode === "original" || state.loading;
  if (hasMask) {
    mask.src = errors ? state.prediction.errors : state.layer === "prediction" ? state.prediction[binary ? "mask" : "overlay"] : referenceURL(binary ? "binary" : "overlay");
    mask.style.opacity = binary ? "1" : String(Number($("#opacity").value) / 100);
  } else { mask.removeAttribute("src"); }
  const label = errors ? "TP xanh · FP đỏ · FN lam" : state.layer === "prediction" ? `Dự đoán · ${methodName[state.prediction?.method]}` : `Nhãn tham chiếu${annotation ? ` · ${annotation.name}` : ""}`;
  $("#viewer-legend").textContent = hasMask ? label : "Ảnh đầu vào · chưa có mask";
  $("#layer-description").textContent = label;
  $("#stage-label").hidden = !hasMask || state.mode === "original" || state.loading;
  const coverage = state.layer === "prediction" ? state.prediction?.coverage : annotation?.coverage;
  $("#coverage-value").textContent = coverage == null ? "—" : `${coverage.toFixed(2)}%`;
  $$('[data-mode]').forEach(button => {
    const selected = button.dataset.mode === state.mode;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  updateControls();
}

function showPlaceholder(title, description) {
  $("#image-frame").hidden = true;
  $("#stage-label").hidden = true;
  $("#stage-placeholder").hidden = false;
  $("#stage-placeholder strong").textContent = title;
  $("#stage-placeholder span").textContent = description;
}

async function selectSample(split, id) {
  const sequence = ++state.selectionSequence;
  ++state.predictionSequence;
  state.loading = true;
  state.source = {kind: "dataset", split, id};
  state.annotation = 0;
  state.detail = null;
  resetPrediction();
  error("");
  showPlaceholder("Đang mở ảnh", "Chuẩn bị ảnh và nhãn tham chiếu…");
  $("#image-name").textContent = id;
  $("#image-meta").textContent = "Đang đọc dữ liệu…";
  $("#annotation-select").innerHTML = '<option value="0">Đang tải…</option>';
  $$(".sample-card").forEach(card => {
    card.classList.toggle("active", card.dataset.id === id);
    card.setAttribute("aria-pressed", String(card.dataset.id === id));
    card.querySelector(".sample-check").hidden = card.dataset.id !== id;
  });
  try {
    const detail = await api(`${baseURL()}`);
    const preview = new Image();
    preview.src = `${baseURL()}/image`;
    await preview.decode();
    if (sequence !== state.selectionSequence) return;
    if (state.objectURL) { URL.revokeObjectURL(state.objectURL); state.objectURL = null; }
    state.detail = detail;
    $("#original-image").src = preview.src;
    $("#original-image").alt = `Ảnh đầu vào ${detail.filename}`;
    $("#image-name").textContent = detail.filename;
    $("#image-meta").textContent = `${number(detail.width)} × ${number(detail.height)} px · ${detail.format} · ${detail.annotations.length} bản gán nhãn`;
    $("#source-badge").textContent = splitName[split].toUpperCase();
    $("#viewer-size").textContent = `${number(detail.width)} × ${number(detail.height)} px`;
    const consensusOption = detail.consensus ? `<option value="consensus">Đồng thuận · ${detail.consensus.coverage.toFixed(2)}% vùng ảnh</option>` : "";
    $("#annotation-select").innerHTML = detail.annotations.length ? consensusOption + detail.annotations.map(a => `<option value="${a.index}">${escapeHTML(a.name)} · ${a.coverage == null ? "Lỗi mask" : `${a.coverage.toFixed(2)}% vùng ảnh`}</option>`).join("") : '<option value="0">Chưa có nhãn tham chiếu</option>';
    state.annotation = detail.consensus ? "consensus" : 0;
    state.loading = false;
    $("#image-frame").hidden = false;
    $("#stage-placeholder").hidden = true;
    resizeFrame(); renderLayer();
    if (annotationInfo()?.error) error(annotationInfo().error);
  } catch (exc) {
    if (sequence !== state.selectionSequence) return;
    state.loading = false; state.source = null;
    error(exc.message); showPlaceholder("Không mở được ảnh", "Thử chọn ảnh khác hoặc kiểm tra dữ liệu."); updateControls();
  }
}

async function loadSamples(selectFirst = false) {
  const sequence = ++state.listSequence;
  const split = state.split;
  $("#sample-grid").setAttribute("aria-busy", "true");
  try {
    const data = await api(`/api/samples?${new URLSearchParams({split, q: state.query, page: state.page, limit: state.limit})}`);
    if (sequence !== state.listSequence) return;
    state.total = data.total;
    $("#sample-total").textContent = number(data.total);
    $("#sample-page").textContent = `${data.page} / ${Math.max(1, Math.ceil(data.total / state.limit))}`;
    $("#prev-page").disabled = state.page <= 1;
    $("#next-page").disabled = state.page * state.limit >= data.total;
    $("#sample-grid").innerHTML = data.items.length ? data.items.map(item => {
      const selected = state.source?.kind === "dataset" && state.source.split === split && state.source.id === item.id;
      return `<button class="sample-card${selected ? " active" : ""}" data-id="${escapeHTML(item.id)}" aria-label="Mở ảnh ${escapeHTML(item.id)}" aria-pressed="${selected}"><img src="/api/samples/${split}/${encodeURIComponent(item.id)}/image?max_size=240" alt="" loading="lazy"><span class="sample-name">${escapeHTML(item.id)}</span><span class="sample-check"${selected ? "" : " hidden"}>${icon("check")}</span></button>`;
    }).join("") : '<div class="empty-state">Không tìm thấy ảnh phù hợp. Hãy thử mã ảnh khác.</div>';
    if (selectFirst && data.items.length) await selectSample(split, data.items[0].id);
    else if (selectFirst && !data.items.length) {
      ++state.selectionSequence; state.source = null; state.detail = null; state.loading = false;
      resetPrediction(); showPlaceholder("Chưa có dữ liệu", "Giải nén ảnh vào thư mục data để bắt đầu."); renderLayer();
    }
  } catch (exc) { if (sequence === state.listSequence) error(exc.message); }
  finally { if (sequence === state.listSequence) $("#sample-grid").setAttribute("aria-busy", "false"); }
}

async function upload(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) { toast("Chọn ảnh JPG, PNG hoặc WebP."); return; }
  if (file.size > 20 * 1024 * 1024) { toast("Ảnh vượt quá giới hạn 20 MB."); return; }
  const sequence = state.selectionSequence;
  const url = URL.createObjectURL(file);
  try {
    const image = new Image(); image.src = url; await image.decode();
    if (sequence !== state.selectionSequence) { URL.revokeObjectURL(url); return; }
    if (image.width * image.height > 25000000) { URL.revokeObjectURL(url); toast("Ảnh vượt quá giới hạn 25 megapixel."); return; }
    ++state.selectionSequence;
    ++state.predictionSequence;
    if (state.objectURL) URL.revokeObjectURL(state.objectURL);
    state.objectURL = url; state.source = {kind: "upload", file}; state.loading = false;
    state.detail = {width: image.width, height: image.height, annotations: [], filename: file.name};
    state.annotation = 0; resetPrediction(); error("");
    $("#original-image").src = url; $("#original-image").alt = `Ảnh tải lên ${file.name}`;
    $("#image-name").textContent = file.name;
    $("#image-meta").textContent = `${number(image.width)} × ${number(image.height)} px · Ảnh tải lên`;
    $("#source-badge").textContent = "ẢNH CỦA BẠN";
    $("#viewer-size").textContent = `${number(image.width)} × ${number(image.height)} px`;
    $("#annotation-select").innerHTML = '<option>Ảnh tải lên chưa có nhãn</option>';
    $$(".sample-card").forEach(card => { card.classList.remove("active"); card.setAttribute("aria-pressed", "false"); card.querySelector(".sample-check").hidden = true; });
    $("#image-frame").hidden = false; $("#stage-placeholder").hidden = true;
    resizeFrame(); renderLayer(); toast("Đã mở ảnh. Chọn phương pháp để chạy phân đoạn.");
  } catch { URL.revokeObjectURL(url); toast("Không đọc được ảnh tải lên."); }
  finally { $("#upload-input").value = ""; }
}

async function predict() {
  if (!state.source || state.loading || state.busy) return;
  const sequence = state.predictionSequence;
  const source = state.source;
  const params = new URLSearchParams({method: $("#method-select").value, threshold: Number($("#threshold").value) / 100,
    normalization: $("#normalization-select").value, annotation: state.annotation});
  if (source.kind === "dataset") { params.set("sample_id", source.id); params.set("split", source.split); }
  state.busy = true; updateControls(); error("");
  try {
    const result = await api(`/api/predict?${params}`, {method: "POST", ...(source.kind === "upload" ? {body: source.file, headers: {"Content-Type": source.file.type}} : {})});
    if (sequence !== state.predictionSequence) { toast("Ảnh hoặc thiết lập đã đổi. Hãy chạy lại với lựa chọn hiện tại."); return; }
    state.prediction = result; state.layer = "prediction";
    $("#layer-source").value = "prediction";
    $("#prediction-result").hidden = false; $("#result-placeholder").hidden = true;
    $("#result-status").textContent = "ĐÃ XỬ LÝ";
    $("#dice-value").textContent = percent(result.metrics?.dice);
    $("#iou-value").textContent = percent(result.metrics?.iou);
    $("#precision-value").textContent = percent(result.metrics?.precision);
    $("#recall-value").textContent = percent(result.metrics?.recall);
    $("#prediction-note").textContent = `${(result.elapsed_ms / 1000).toFixed(2)} giây · Ngưỡng ${result.threshold.toFixed(2)}${result.metrics ? ` · So với ${result.reference.replace(/.*_(ann\d+)\.tiff?$/i, "$1")}, độ phân giải gốc.` : " · Chưa có nhãn để tính độ đo."}`;
    renderLayer();
  } catch (exc) { if (sequence === state.predictionSequence) error(exc.message); }
  finally { state.busy = false; updateControls(); }
}

function downloadMask() {
  if ($("#download-button").disabled) return;
  const anchor = document.createElement("a");
  const name = state.detail?.filename?.replace(/\.[^.]+$/, "") || "segmentation";
  anchor.href = state.layer === "prediction" ? state.prediction.mask : referenceURL("binary", true);
  anchor.download = `${name}_${state.layer === "prediction" ? state.prediction.method : annotationInfo().name}_mask.png`;
  document.body.append(anchor); anchor.click(); anchor.remove();
}

function renderInventory() {
  const summary = state.summary;
  $("#nav-total").textContent = number(summary.total_images);
  $("#total-images").textContent = number(summary.total_images);
  for (const split of ["train", "valid", "test"]) $(`#${split}-count`).textContent = number(summary.splits[split].images);
  $('#method-select option[value=pixel_logistic]').disabled = !summary.pixel_model?.inference_ready;
  $('#method-select option[value=compact_unet]').disabled = !summary.compact_model?.inference_ready;
  $("#dataset-content").innerHTML = `<div class="content-panel"><h2>Dữ liệu đã giải nén</h2><p>${number(summary.total_images)} ảnh đầu vào và ${number(summary.total_masks)} file mask. Ảnh và nhãn được ghép theo mã ảnh; các bản gán nhãn trong tập test được nhận diện theo tên file thực tế.</p><table><thead><tr><th>Tập dữ liệu</th><th>Ảnh đầu vào</th><th>Mask</th><th>Chưa có nhãn</th><th>Thư mục</th></tr></thead><tbody>${Object.entries(summary.splits).map(([key, s]) => `<tr><td>${splitName[key]}</td><td>${number(s.images)}</td><td>${number(s.masks)}</td><td>${s.unlabeled}</td><td class="mono">data/images${{train:"Train",valid:"Valid",test:"Test"}[key]}</td></tr>`).join("")}</tbody></table></div><div id="eda-content"></div><div class="content-panel"><h2>Ảnh và nhãn tham chiếu</h2><p>Ảnh JPG/PNG và mask TIFF nhị phân. Một ảnh test có thể có nhiều bản gán nhãn. Không gian phân đoạn cho phép chọn từng bản để đối chiếu; thực nghiệm sử dụng mask đồng thuận theo đa số.</p><div class="detail-grid"><div><small>Đầu vào</small><strong>Ảnh RGB, nhiều độ phân giải</strong></div><div><small>Đầu ra</small><strong>Mask nhị phân cùng kích thước ảnh</strong></div><div><small>Nguồn dữ liệu</small><strong>Tham khảo DermaVQA-DAS 2025; cần đối chiếu bản cục bộ</strong></div><div><small>Kiểm tra dữ liệu</small><strong>Xem kết quả kiểm kê trong mục Thực nghiệm</strong></div></div></div>`;
  const model = summary.model;
  $("#model-content").innerHTML = `<div class="content-panel"><h2>TransUNet · R50 + ViT-B/16</h2><p>Kiến trúc kết hợp đặc trưng ResNet và Vision Transformer để tạo mask nhị phân. Checkpoint hiện có được dùng cho suy luận; dữ liệu huấn luyện và chuẩn hóa đầu vào gốc cần được xác nhận để diễn giải kết quả.</p><div class="detail-grid"><div><small>Checkpoint</small><strong>${escapeHTML(model.checkpoint_name || "Chưa có checkpoint")}</strong></div><div><small>Dung lượng</small><strong>${(model.checkpoint_bytes / 1024 ** 2).toFixed(1)} MiB</strong></div><div><small>Đầu vào mô hình</small><strong>224 × 224 · RGB</strong></div><div><small>Đầu ra</small><strong>Sigmoid → ngưỡng → mask nhị phân</strong></div></div><div class="pipeline"><span>Ảnh RGB</span>${icon("arrow")}<span>Resize & chuẩn hóa</span>${icon("arrow")}<span>TransUNet</span>${icon("arrow")}<span>Mask kích thước gốc</span></div></div><div class="content-panel"><h2>Baseline · LAB + Otsu</h2><p>Phương pháp xử lý ảnh truyền thống dùng thành phần màu trong không gian LAB, ngưỡng Otsu và phép toán hình thái. Baseline tạo mốc so sánh cho mô hình học sâu; không sử dụng mask tham chiếu để sinh dự đoán.</p></div><div class="content-panel"><h2>Đánh giá và khả năng tái lập</h2><p>Độ đo: Dice, IoU, Pixel Accuracy, Precision và Recall. Demo tính theo bản gán nhãn được chọn ở độ phân giải gốc. Thực nghiệm tổng hợp tính theo mask đồng thuận ở độ phân giải ghi trong từng kết quả. Chưa có lịch sử huấn luyện của checkpoint; không thể kết luận tập test chưa từng được dùng để huấn luyện.</p></div>`;
  if (summary.pixel_model?.inference_ready) {
    $("#model-content").insertAdjacentHTML("afterbegin", `<div class="content-panel"><h2>Logistic pixel · Huấn luyện local</h2><p>Mô hình tuyến tính học từ 10 đặc trưng màu và thống kê cục bộ. Hai lượt huấn luyện dùng tập train; ngưỡng ${summary.pixel_model.threshold.toFixed(2)} được chọn trên validation trước khi đánh giá test. Tham số được lưu trong JSON để kiểm tra và tái chạy.</p></div>`);
  }
  const compact = summary.compact_model;
  $("#model-content").insertAdjacentHTML("afterbegin", `<div class="content-panel"><h2>Compact U-Net · Huấn luyện từ đầu</h2><p>Encoder–decoder với kết nối skip. Trọng số khởi tạo ngẫu nhiên, chỉ tập train cập nhật tham số; validation chọn epoch và ngưỡng trước khi đánh giá.</p>${compact?.inference_ready ? `<div class="detail-grid"><div><small>Train / validation</small><strong>${number(compact.counts.train)} / ${number(compact.counts.valid)} ảnh</strong></div><div><small>Lịch huấn luyện</small><strong>${compact.epochs} epoch · seed ${compact.seed}</strong></div><div><small>Epoch được chọn bằng validation</small><strong>${compact.best_epoch}</strong></div><div><small>Ngưỡng đã chốt</small><strong>${compact.threshold.toFixed(2)}</strong></div></div>` : '<p>Chưa có trọng số và cấu hình đã chốt hợp lệ. Phương pháp sẽ được bật khi thí nghiệm hoàn tất.</p>'}<p class="field-note">Ablation so sánh BCE/BCE+Dice và augmentation; nhiều seed mô tả biến thiên khi khởi tạo. Kết quả test cục bộ đã được quan sát ở giai đoạn trước.</p></div>`);
  api("/api/experiments").then(({eda}) => renderEDA(eda)).catch(() => {
    $("#eda-content").innerHTML = '<div class="content-panel"><p>Chưa đọc được kết quả EDA. Hãy thử tải lại trang.</p></div>';
  });
}

function renderEDA(eda) {
  const target = $("#eda-content");
  if (!target || !eda) return;
  const total = eda.totals;
  const test = eda.splits.test;
  const agreement = test?.statistics.pairwise_dice_thumbnail_mean.mean;
  target.innerHTML = `<div class="content-panel"><h2>Khám phá dữ liệu · EDA</h2><p>Đã đọc toàn bộ pixel của ${number(total.images_decoded)} ảnh và ${number(total.masks_decoded)} mask. Phát hiện ${total.corrupt_images + total.corrupt_masks} file không đọc được, ${total.nonbinary_masks} mask không nhị phân và ${total.empty_masks} mask rỗng.</p><table><thead><tr><th>Tập</th><th>Foreground trung bình / mask</th><th>Foreground trung vị / mask</th><th>Độ sáng trung bình / ảnh</th></tr></thead><tbody>${Object.entries(eda.splits).map(([split, s]) => `<tr><td>${splitName[split]}</td><td>${percent(s.foreground_fraction_per_annotation.mean)}</td><td>${percent(s.foreground_fraction_per_annotation.median)}</td><td>${s.statistics.brightness_mean.mean.toFixed(1)} / 255</td></tr>`).join("")}</tbody></table><div class="detail-grid"><div><small>Đồng thuận nhãn test · Dice giữa các cặp</small><strong>${agreement?.toFixed(4) ?? "—"} · ${test.images_with_pairwise_dice_below_0_5} ảnh dưới 0,5</strong></div><div><small>Cần chú ý khi diễn giải</small><strong>Ảnh trùng trong nội bộ tập và nhãn bất đồng</strong></div></div><a href="/api/results/eda/overview.png" target="_blank" rel="noopener"><img class="result-figure" src="/api/results/eda/overview.png" alt="Biểu đồ EDA về phân chia dữ liệu, độ phân giải, foreground, chất lượng ảnh và đồng thuận nhãn" loading="lazy"></a><div class="artifact-links"><a class="button button-white" href="/api/results/eda/per_image.csv?download=true">CSV theo ảnh ${icon("download")}</a><a class="button button-white" href="/api/results/eda/summary.json?download=true">Thống kê JSON ${icon("download")}</a><a class="button button-white" href="/api/results/eda/outliers.png" target="_blank" rel="noopener">Xem ảnh bất thường ${icon("arrow")}</a></div><p class="field-note">Foreground là pixel có giá trị nhãn khác 0; ý nghĩa chính xác cần đối chiếu nguồn dataset. Thống kê mô tả trên các tập không được dùng để tối ưu tham số theo test.</p></div>`;
}

async function loadExperiments() {
  const target = $("#experiments-content");
  target.innerHTML = '<div class="content-panel"><p>Đang đọc kết quả thực nghiệm…</p></div>';
  try {
    const {reports, audit, analysis} = await api("/api/experiments");
    const panels = reports.map(report => {
      const data = report.data;
      const rows = Object.entries(data.methods || {}).map(([method, result]) => {
        const metrics = result.macro?.mean || result.metrics || {};
        return `<tr><td>${escapeHTML(methodName[method] || method)}</td><td>${percent(metrics.dice)}</td><td>${percent(metrics.iou)}</td><td>${percent(metrics.pixel_accuracy)}</td><td>${percent(metrics.precision)}</td><td>${percent(metrics.recall)}</td></tr>`;
      }).join("");
      const run = report.file.split("/").slice(-2, -1)[0];
      const runPath = report.file.replace(/^results\//, '').split('/').slice(0,-1).map(encodeURIComponent).join('/');
      const friendlyName = data.format === "cs406-completion-evaluation-v1" ? (data.dataset?.role === "external_stress_test" ? "Stress ngoài nguồn · ISIC 2018 validation" : "Thực nghiệm có kiểm soát · Compact U-Net và đối chứng") : {comparison_test: "So sánh trên toàn bộ tập test", pixel_classifier: "Logistic pixel · học từ tập train", baseline_test: "Baseline · toàn bộ tập test", comparison_valid_unit: "Validation · RGB / 255", transunet_valid_imagenet: "Validation · ImageNet", transunet_test_smoke: "Kiểm tra nhanh · 10 ảnh test"}[run] || run;
      return `<div class="content-panel"><h2>${escapeHTML(friendlyName)}</h2><p class="mono">${escapeHTML(report.file)}</p><p>${number(data.requested_images ?? data.available_images ?? data.sample_count ?? 0)} ảnh · ${escapeHTML(splitName[data.split] || data.split || "")} · ${data.failures?.length || 0} lỗi · ${data.limited_run ? "Chạy một phần" : "Toàn bộ tập ảnh"}</p>${rows ? `<table><thead><tr><th>Phương pháp</th><th>Dice</th><th>IoU</th><th>Pixel Acc.</th><th>Precision</th><th>Recall</th></tr></thead><tbody>${rows}</tbody></table>` : '<p>Đang chờ bảng độ đo.</p>'}${run === "comparison_test" ? `<a href="/api/results/comparison_test/examples.png" target="_blank" rel="noopener"><img class="result-figure" src="/api/results/comparison_test/examples.png" alt="Các trường hợp phân đoạn tốt và thất bại, so sánh nhãn tham chiếu với dự đoán" loading="lazy"></a>` : ""}<div class="artifact-links"><a class="button button-white" href="/api/results/${runPath}/per_image.csv?download=true">CSV từng ảnh ${icon("download")}</a><a class="button button-white" href="/api/results/${runPath}/summary.json?download=true">Kết quả JSON ${icon("download")}</a></div><details><summary>Giao thức & thông tin chạy</summary><pre>${escapeHTML(JSON.stringify(data.protocol || data.config || data.metadata || {}, null, 2))}</pre></details></div>`;
    });
    const analysisPanel = analysis ? `<div class="content-panel"><h2>Phân tích ghép cặp và ảnh trùng</h2><p>TransUNet có Dice cao hơn baseline trên ${analysis.paired_dice.transunet_wins}/295 ảnh; baseline tốt hơn trên ${analysis.paired_dice.baseline_wins} ảnh. Kết quả phụ thuộc kích thước foreground, mức đồng thuận nhãn và đơn vị tổng hợp.</p><a href="/api/results/analysis/foreground_strata.png" target="_blank" rel="noopener"><img class="result-figure" src="/api/results/analysis/foreground_strata.png" alt="Dice theo bốn nhóm tỷ lệ foreground" loading="lazy"></a><div class="artifact-links"><a class="button button-white" href="/api/results/analysis/summary.json?download=true">Phân tích JSON ${icon("download")}</a><a class="button button-white" href="/api/results/analysis/macro_comparison.png" target="_blank" rel="noopener">Khoảng bất định ${icon("arrow")}</a></div></div>` : "";
    target.innerHTML = analysisPanel + (panels.length ? panels.join("") : '<div class="content-panel"><h2>Chưa có kết quả tổng hợp</h2><p>Chạy lệnh đánh giá để tạo bảng so sánh từ dữ liệu thật.</p><pre>python -m scripts.evaluate --method both --split test --output results/comparison_test</pre></div>') + `<div class="content-panel"><h2>Ghi chú thực nghiệm</h2><p>Không suy luận chất lượng từ một ảnh đơn lẻ. Bảng so sánh cần cùng tập ảnh, cùng độ phân giải, cùng mask đồng thuận và cùng cách tính độ đo. Ảnh nguồn và cách chia cục bộ đã được đối chiếu với DermaVQA; phép tạo mask train/valid, lịch sử huấn luyện và chuẩn hóa gốc của checkpoint TransUNet chưa được xác minh đầy đủ. Test cục bộ không phải test leaderboard.</p>${audit ? `<details><summary>Kiểm kê và kiểm tra trùng lặp dữ liệu</summary><pre>${escapeHTML(JSON.stringify(audit, null, 2))}</pre></details>` : '<p>Chưa có kết quả kiểm kê dữ liệu.</p>'}</div>`;
  } catch (exc) { target.innerHTML = `<div class="alert error">${escapeHTML(exc.message)}</div>`; }
}

function navigate(view) {
  if (!["workspace", "dataset", "experiments", "model"].includes(view)) view = "workspace";
  state.view = view;
  $$(".page-view").forEach(section => { section.hidden = section.id !== `view-${view}`; });
  $$("[data-view]").forEach(link => { const active = link.dataset.view === view; link.classList.toggle("active", active); if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current"); });
  $("#breadcrumb-page").textContent = {workspace: "Phân đoạn ảnh", dataset: "Bộ dữ liệu", experiments: "Thực nghiệm", model: "Mô hình"}[view];
  if (view === "experiments") loadExperiments();
  if (view === "workspace") requestAnimationFrame(resizeFrame);
}

$("#sample-grid").addEventListener("click", event => { const card = event.target.closest("[data-id]"); if (card) selectSample(state.split, card.dataset.id); });
$("#split-select").addEventListener("change", () => {
  ++state.selectionSequence; ++state.predictionSequence; state.source = null; state.detail = null; state.loading = true;
  resetPrediction(); showPlaceholder("Đang đổi tập dữ liệu", "Đang tìm ảnh đầu tiên…");
  state.split = $("#split-select").value; state.page = 1; state.query = ""; $("#sample-search").value = ""; loadSamples(true);
});
let searchTimer;
$("#sample-search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.query = $("#sample-search").value; state.page = 1; loadSamples(); }, 220); });
$("#prev-page").addEventListener("click", () => { if (state.page > 1) { state.page--; loadSamples(); } });
$("#next-page").addEventListener("click", () => { if (state.page * state.limit < state.total) { state.page++; loadSamples(); } });
$$('[data-mode]').forEach(button => button.addEventListener("click", () => { state.mode = button.dataset.mode; if (state.mode === "errors") { state.layer = "prediction"; $("#layer-source").value = "prediction"; } renderLayer(); }));
$("#opacity").addEventListener("input", () => { $("#opacity-value").value = `${$("#opacity").value}%`; renderLayer(); });
$("#annotation-select").addEventListener("change", () => { ++state.predictionSequence; const value = $("#annotation-select").value; state.annotation = value === "consensus" ? value : Number(value); resetPrediction(); renderLayer(); });
$("#layer-source").addEventListener("change", () => {
  state.layer = $("#layer-source").value;
  if (state.layer === "reference" && state.mode === "errors") state.mode = "overlay";
  renderLayer();
});
$("#threshold").addEventListener("input", () => { ++state.predictionSequence; $("#threshold-value").value = (Number($("#threshold").value) / 100).toFixed(2); resetPrediction(); renderLayer(); });
$("#method-select").addEventListener("change", () => {
  ++state.predictionSequence;
  const method = $("#method-select").value;
  $("#normalization-field").hidden = method !== "transunet";
  $("#method-note").textContent = {transunet: "ResNet + Vision Transformer · checkpoint có sẵn.", baseline: "Phân đoạn bằng màu sắc và ngưỡng Otsu.", pixel_logistic: "10 đặc trưng màu và thống kê cục bộ · học từ train.", compact_unet: "U-Net gọn · khởi tạo ngẫu nhiên, chỉ train cập nhật trọng số."}[method];
  const threshold = method === "pixel_logistic" ? state.summary?.pixel_model?.threshold ?? .5 : method === "compact_unet" ? state.summary?.compact_model?.threshold ?? .5 : .5;
  $("#threshold").value = Math.round(threshold * 100);
  $("#threshold-value").value = threshold.toFixed(2);
  resetPrediction(); renderLayer();
});
$("#normalization-select").addEventListener("change", () => { ++state.predictionSequence; resetPrediction(); renderLayer(); });
$("#predict-button").addEventListener("click", predict);
$("#download-button").addEventListener("click", downloadMask);
$("#upload-button").addEventListener("click", () => $("#upload-input").click());
$("#upload-input").addEventListener("change", () => upload($("#upload-input").files[0]));
$("#fullscreen-button").addEventListener("click", async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await $("#image-stage").requestFullscreen(); } catch { toast("Trình duyệt không hỗ trợ chế độ toàn màn hình."); } });
$("#mask-image").addEventListener("error", () => {
  $("#mask-image").hidden = true;
  $("#original-image").style.visibility = "visible";
  $("#stage-label").hidden = true;
  $("#viewer-legend").textContent = "Không tải được mask · đang hiển thị ảnh gốc";
  toast("Không tải được lớp mask. Hãy chọn lại bản gán nhãn.");
});
$("#refresh-experiments").addEventListener("click", loadExperiments);
$$('[data-go]').forEach(button => button.addEventListener("click", () => { location.hash = button.dataset.go; }));
window.addEventListener("hashchange", () => navigate(location.hash.slice(1)));

(async function init() {
  navigate(location.hash.slice(1));
  try { state.summary = await api("/api/summary"); renderInventory(); await loadSamples(true); }
  catch (exc) { error(`${exc.message} Hãy kiểm tra máy chủ FastAPI.`); showPlaceholder("Chưa kết nối được dữ liệu", "Khởi động máy chủ bằng start.ps1 rồi tải lại trang."); }
})();
