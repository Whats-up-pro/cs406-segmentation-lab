# CS406 Course Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Hoàn thiện đồ án theo DOCX với thí nghiệm mới có kiểm soát, bằng chứng mạnh hơn và bộ nộp chạy lại được.

**Architecture:** Mở rộng FastAPI/HTML hiện có bằng Compact U-Net huấn luyện từ đầu và các CLI thí nghiệm độc lập. Dữ liệu/artefact lớn nằm ngoài Git; manifest và cấu hình nằm trong Git. Hoàn thiện báo cáo/slide từ số liệu đã chạy.

**Tech Stack:** Python 3.13, PyTorch CPU, NumPy/OpenCV/Pillow, FastAPI, HTML/CSS/JS, pytest, Word/PDF/PowerPoint artifact runtimes.

**Spec:** `docs/superpowers/specs/2026-11-19-course-completion-design.md`

## Global Constraints

- Bám DOCX gốc; phân biệt mandatory/optional/human confirmation; không hứa điểm chấm.
- Không sửa raw dataset hoặc checkpoint gốc; không chọn cấu hình từ test.
- Train đủ dữ liệu, lịch sử/manifest/hash thật; không giả số liệu, danh tính, provenance hoặc nội dung bảo vệ đã được hiểu.
- CPU mặc định 4 luồng; dataset/result cache dưới `Project/results/completion`; không tạo chi phí dịch vụ ngoài.
- Nhánh làm việc `feat/course-completion-20261119`; nguồn gốc `b264cd6`; không force push nhánh chia sẻ.

## Review Focus

1. Nhãn test bị mở trước khi checkpoint/ngưỡng được đóng băng → test CLI phải chứng minh chặn đọc test khi chưa có frozen manifest.
2. Lần chạy lại ghi đè checkpoint hoặc trộn config → output run phải từ chối overwrite, resume chỉ khi manifest tương thích.
3. Mask nhiều annotation và rỗng → test nearest/binary, strict majority và macro/micro xử lý đúng.
4. Dữ liệu tải mới hoặc ảnh khác tỉ lệ → test manifest/hash, output kích thước gốc, không tự báo metric không có nhãn.
5. Artifact stale và đường dẫn local → kiểm tra số liệu, TOC/caption/reference và package links sau render/clean checkout.

---

### Task 1: Rubric, nguồn và kiểm tra dữ liệu

**Files:** Create `docs/completion-status.md`, `config/course-requirements.json`, `scripts/verify_submission.py`, `scripts/audit_leakage.py`, `tests/test_submission.py`, `tests/test_leakage.py`; update provenance notes.
**Interfaces:** requirement entries `{id, clause, kind, evidence, state}`; leakage JSON `{protocol, counts, cross_split_candidates, limitations}`. Existing source data under original `Project/data` is read-only.
- [ ] Extract all DOCX clauses and create atomic mandatory/optional evidence map; document pending identity/provenance separately.
- [ ] Write failing tests for missing evidence, prohibited package content, exact duplicates, transformed duplicate candidates and unrelated synthetic patterns.
- [ ] Implement verifier and pHash/dHash candidate audit; store thresholds/protocol and image-pair review evidence without inventing subject identities.
- [ ] Run tests and real audit, read findings, update completion ledger and commit.

### Task 2: Compact U-Net and controlled full training

**Files:** Create `backend/compact_model.py`, `scripts/train_compact.py`, `tests/test_compact_model.py`, `tests/test_compact_training.py`, `config/experiments/compact_ablation.json`.
**Interfaces:** `CompactUNet(base_channels=12)` maps N×3×H×W to N×1×H×W; `predict_compact(image, checkpoint)` returns original-size float32 probabilities. Training CLI consumes data-dir/output/config/seed and produces config/history/model/frozen-selection with source/checkpoint hashes.
- [ ] Write and run failing tests: non-square sizes; finite backward loss; aligned/seeded augmentation; train update vs validation no-update; no test read until frozen; no overwrite; checkpoint reproducible load and dimensions.
- [ ] Implement compact architecture, uint8 cached train/valid preprocessing, deterministic training, validation checkpoint selection and validation-only threshold selection.
- [ ] Run entire test suite. Benchmark one genuine epoch to quantify throughput; record any resource ruling before scheduled runs.
- [ ] Execute four loss×augmentation configurations at seed406 and combined recipe at seed407/408 for the declared full schedule. Save full histories, counts, weights and frozen manifests. No test scoring until these exist.
- [ ] Inspect completed runs and commit source/config/evidence summaries.

### Task 3: Comparison, ablation, uncertainty and external stress evaluation

**Files:** Create `scripts/evaluate_completion.py`, `scripts/analyze_completion.py`, `scripts/prepare_external.py`, `tests/test_completion_evaluation.py`, `docs/experiments-completion.md`.
**Interfaces:** per-image CSV contains method/sample/subject/TP/FP/FN/TN/five metrics/latency; summary JSON contains macro/micro, seed variability, bootstrap and dataset/protocol hashes.
- [ ] Write/run failing tests for macro versus micro, empty masks, strict consensus and frozen-config enforcement; paired bootstrap groups/seed and corrupt external input.
- [ ] Implement shared evaluation without training imports into inference; validate external source/license/metadata and save immutable download manifest.
- [ ] Run the same 295-image test comparison, all planned ablations/multiple seeds; add reference methods without implying unseen-checkpoint independence.
- [ ] Run external stress test with frozen parameters and document modality/label differences. Produce plots/tables/paired error examples/robustness results from saved CSVs.
- [ ] Verify result tables against per-image data and commit.

### Task 4: Demo, installation and reproducibility

**Files:** Modify `backend/main.py`, frontend files, README and model manifest; add `tests/test_compact_api.py`, `scripts/check_ui.py` as needed and a clean-run/checkpoint download script.
**Interfaces:** new method ID `compact_unet`; explicit availability and source/checkpoint provenance; configure dataset/results roots via documented environment variables.
- [ ] Write/run failing tests for missing model, real upload inference/no invented scores, original-size PNG, selected annotation/error overlay and configurable paths.
- [ ] Implement new model in UI/API and show the correct saved protocol/results; preserve existing methods and error behavior.
- [ ] Verify all tests, JS syntax, live Playwright flows/responsive layouts and downloaded/checksummed model path.
- [ ] Test clean source checkout with documented installation/runtime path, commit.

### Task 5: Report, slides and defense evidence

**Files:** Update original `Project/docs/report.md`, related methods/results/defense docs, report/figure/presentation builders; generate versioned DOCX/PDF/PPTX/HTML deliverables.
**Interfaces:** report and slide tables read only finalized result manifests; identity from user-confirmed metadata; old user-authored files are preserved.
- [ ] Load document/PDF/presentation skills and define a data-driven content contract: nine report chapters, current sources, actual methods/ablation/seeds/stress results, limitations and reproducibility.
- [ ] Fix cover, editorial notes, TOC/list-of-figures page numbers and stale figures; use genuine results and explicit protocol differences.
- [ ] Generate editable Word/PowerPoint and PDF/offline HTML; render and visually inspect every page/slide, iterate on defects.
- [ ] Update speaking script, demo scenarios, member contribution matrix and question bank with confirmed information; distinguish preparation from actual oral competence.
- [ ] Verify content numbers, references, fields, links and captions; commit source and store final artefact hashes.

### Task 6: Final review, package and GitHub

**Files:** Update evidence map, `docs/completion-status.md`, `scripts/verify_submission.py`, repository README/model/download links, deliverable manifests and submission ZIPs.
- [ ] Run requirement verifier against the original DOCX/evidence, inspect every unresolved entry.
- [ ] Run full suite, browser/clean-package checks; obtain independent whole-branch review and fix substantive findings with regression tests.
- [ ] Verify release weights/checksum links, source package exclusions, deliverable render/content consistency and training replication instructions.
- [ ] Publish authorized source/artefact updates with remote verification; binary transport unavailable in sandbox gets a concrete tested upload action, never a false uploaded claim.
- [ ] Mark complete only for achieved evidence-backed scope; otherwise keep explicit remaining human/external requirements while continuing feasible work.
