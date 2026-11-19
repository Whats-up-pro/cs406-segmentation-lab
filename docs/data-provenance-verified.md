# Verified dataset provenance and evaluation limits

Checked 19 November 2026 against primary sources. This note adds evidence to [the existing checkpoint note](data-and-checkpoint.md); it does not identify an undocumented checkpoint training run.

## Original corpus and segmentation extension

[DermaVQA's MICCAI 2024 repository](https://github.com/velvinnn/DermaVQA/blob/main/README.md) describes dermatology questions and images from IIYI and Reddit, supporting free-text question answering. The [original paper, section 3](https://papers.miccai.org/miccai-2024/paper/2444_paper.pdf) describes public forum collection, removal of identifying/inappropriate images and unusable threads, and conversion of conversations into question/answer pairs. These are corpus curation instructions, rather than the later mask authoring protocol.

The [official 2025 organizer repository](https://github.com/wyim/ImageCLEF-MAGIC-2025/blob/main/README.md) documents segmentation and closed QA. [DermaVQA-DAS v2, revised 9 November 2026, table 3 and section 2](https://arxiv.org/html/2512.24340v2) explicitly separates the segmentation splits:

| Split | Published images | Published masks | Supplied local images | Supplied local masks |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,474 | 7,448 | 2,179 | 2,179 |
| Validation | 157 | 472 | 157 | 157 |
| Test | 314 | 944 | 295 | 885 |

The published extension marks clinically relevant problem regions, with three masks described per image, provided by a pool of four **medical annotators**. Two board-certified dermatologists designed the DAS question schema; this does not establish that every mask annotator was a dermatologist. Published mask totals slightly exceed three times image counts. The precise reason, the drawing tool, adjudication process, omitted images, and the transformation to one supplied training/validation mask remain unverified. The local version must be described using its own inventory, without claiming it is the complete official release.

## License and access evidence

The [official OSF project metadata](https://api.osf.io/v2/nodes/72rp3/) links to a [CC BY 4.0 license record](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/). The deposited [LICENSE file](https://osf.io/67n5p/) contains the same license. These were read through the public OSF API/file service because the OSF frontend returned HTTP 403 in the research tool.

This establishes the declared license of the original OSF release. It does not establish which transformed files were supplied locally or independently resolve the supplied checkpoint's redistribution/training terms. Attribution should preserve the source, license link and description of transformations. The Apache-2.0 license for upstream TransUNet source is a separate matter.

[ImageCLEF 2025 registration instructions](https://www.imageclef.org/2025) require the registration form and acceptance of task-specific terms in AI4MediaBench. The [competition endpoint linked by the organizers](https://ai4media-bench.aimultimedialab.ro/competitions/62/) returned HTTP 404 during this check, so that agreement was not verified. The [MEDIQA-MAGIC 2025 task page](https://www.imageclef.org/2025/medical/mediqa) prohibits using the original 2024 answers for training or inference.

## Scoring and checkpoint identity

The [organizer task page](https://www.imageclef.org/2025/medical/mediqa) reports Anastasia submissions with Jaccard 0.645774234 and Dice 0.784766. A submission name and score do not identify the supplied weight file: a matching author-provided checkpoint hash, configuration and training/submission manifest are still needed.

The [official segmentation scoring code](https://github.com/wyim/ImageCLEF-MAGIC-2025/blob/main/evaluation/score_segmentations.py) pools pixel intersections/unions for micro scores. Its vote threshold is `sum >= ceil(number_of_masks / 2)`: even-count ties become foreground. This project uses strict majority, so ties become background. Both agree for the local test images with exactly three masks. Official original-resolution micro results and local 256-square macro results are different protocols and must not be equated.

## Leakage audit and prior test inspection

Run from the repository root:

```powershell
python -m scripts.audit_leakage --data-dir ../Project/data --output ../Project/results/completion/leakage
```

The audit reads original image bytes without changing source data or labels. It records SHA-256 groups, filename encounter overlaps, and transformed-copy candidates requiring both pHash Hamming distance <= 6 and dHash distance <= 8. Exact duplicates are reported separately. A low-information flag identifies potential hash collisions. Candidate CSV/JSON and a ranked contact sheet support visual review; the images remain outside Git. Hashes and thresholds are recorded in the generated protocol.

An `ENC` filename group is an **encounter identifier**, not a verified patient identifier. Absent encounter overlap or perceptual candidates cannot prove patient independence, rule out all transformed copies, or establish checkpoint history. No candidate is automatically removed or declared clinically identical.

The actual 19 November run decoded all **2,631 images**, representing **2,597 distinct SHA-256 values** and **898 filename encounter groups** (train 739, validation 56, test 103). These group counts also differ from the published split counts. Across **1,031,223 cross-split image pairs**, there were **zero cross-split exact duplicate groups**, **zero encounter overlaps**, and **zero perceptual candidates at the declared thresholds**. There were **28 exact duplicate groups confined to individual splits** (train 17, validation 7, test 4). The run had no decode errors. Reproducible evidence is in `Project/results/completion/leakage/summary.json`, `inventory.csv`, and `cross_split_candidates.csv`; no contact sheet was generated because there were no qualifying candidates. These findings preserve the limitations above.

The local test images, annotations and earlier results were already inspected before the current completion experiments; historical evidence includes `Project/results/comparison_test/` and the existing result analysis. The test set is therefore a previously inspected benchmark, not a pristine untouched holdout. Freezing new configurations before their new test runs prevents additional selection from those runs; it cannot erase the earlier inspection.

## External ISIC 2018 validation stress set

The [official collection](https://api.isic-archive.com/collections/62/) contains 100 validation images. The [official download table](https://challenge.isic-archive.com/data/#2018) labels Task 1 CC0, distinct from Task 3's license. Direct official archives, both verified with HTTP 200:

| Payload | Official download | Advertised bytes |
| --- | --- | ---: |
| Images | [ISIC2018_Task1-2_Validation_Input.zip](https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip) | 239,231,159 |
| Masks | [ISIC2018_Task1_Validation_GroundTruth.zip](https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip) | 759,706 |

The [official task specification](https://challenge.isic-archive.com/landing/2018/45/) defines one primary lesion in each dermoscopic image, with same-size grayscale PNG masks using 0 for background and 255 for foreground. Practicing dermatologists reviewed/curated annotations generated by accepted automated results, expert-controlled flood fill or manual polygons. Dermoscopic primary-lesion boundaries differ from clinically contextual problem regions in consumer photographs. Report this as an external modality/semantics stress test with frozen parameters, rather than clinical validation. The unknown supplied checkpoint history also prevents asserting its ISIC images were unseen during pretraining.

Download into `../Project/data_external/isic2018_validation/raw/`, preserving the raw archives, then prepare:

```powershell
python -m scripts.prepare_external --images-archive ../Project/data_external/isic2018_validation/raw/ISIC2018_Task1-2_Validation_Input.zip --masks-archive ../Project/data_external/isic2018_validation/raw/ISIC2018_Task1_Validation_GroundTruth.zip
```

The helper checks advertised sizes, safe archive member paths, duplicate flattened names, ZIP CRC, exactly 100 matched IDs, RGB JPEG decoding, grayscale PNG mask decoding, binary values and matching dimensions before creating prepared folders. It refuses overwrites. It preserves payload bytes and renames mask paths for the existing dataset loader. `manifest.json` records source/license URLs, raw archive and per-payload SHA-256, dimensions and foreground coverage. Locally recorded hashes support reproducibility; no publisher-provided SHA-256 was found for independent archive identity verification.

Both archives were downloaded successfully and preparation completed on 19 November 2026 with **100 images and 100 matching masks**, all passing the checks above. The original ZIPs remain in `Project/data_external/isic2018_validation/raw/`; the preparation manifest is `Project/data_external/isic2018_validation/manifest.json`. Local archive SHA-256 values are:

| Archive | Locally measured SHA-256 |
| --- | --- |
| Images | `0ea920fcfe512d12a6e620b50b50233c059f67b10146e1479c82be58ff15a797` |
| Masks | `f6911e9c0a64e6d687dd3ca466ca927dd5e82145cb2163b7a1e5b37d7a716285` |

The mask archive's own `ATTRIBUTION.txt` and `LICENSE.txt` also declare CC0 1.0. Preparation is complete; model evaluation is a separate operation and no stress-test performance is asserted by this note.

Prepared layout is `imagesTest/ISIC_XXXXXXX.jpg` and `labelsTest/ISIC_XXXXXXX.png`. The manifest retains the original split name **validation** and assigns the local role **external stress test**. No model fitting or threshold selection occurs in preparation. Cite Codella et al., *Skin Lesion Analysis Toward Melanoma Detection 2018*, [arXiv:1902.03368](https://arxiv.org/abs/1902.03368), and Tschandl et al., *The HAM10000 dataset*, [Scientific Data 5, 180161](https://doi.org/10.1038/sdata.2018.161), as requested by the collection.

## Confirmed source split membership of the local test set

**The 295-image local test set is a repartition of the organizer-linked source training images, not the official 314-image held-out test set.** This conclusion follows from exact filename/encounter joins and public archive metadata, independently of the matching image counts. It resolves the earlier uncertainty about source image split membership; it does not resolve local mask construction or supplied checkpoint training history.

The [2025 organizer README pinned at revision `6d60a81e62d376d89d4ddbacb7bdc4f014cbbea6`](https://github.com/wyim/ImageCLEF-MAGIC-2025/blob/6d60a81e62d376d89d4ddbacb7bdc4f014cbbea6/README.md) explicitly links [the source image archive](https://osf.io/p8bfu/), [training queries](https://osf.io/jgbtm/), [updated validation queries](https://osf.io/krj8g/), and [updated test queries](https://osf.io/y82bn/). Only a **524,288-byte HTTP range from the end** of the 1,104,296,867-byte image archive was downloaded to inspect its full ZIP central directory. No source images were extracted from that archive. It lists exactly 2,474 `images_train`, 157 `images_valid`, and 314 `images_test` files.

| Supplied local split | Local images | Local encounter groups | Matching source archive folder | Matches |
| --- | ---: | ---: | --- | ---: |
| Train | 2,179 | 739 | `images_train` | 2,179 |
| Validation | 157 | 56 | `images_valid` | 157 |
| Test | 295 | 103 | `images_train` | 295 |

The union of local train and local test image stems equals **all 2,474 source training image stems exactly**, with neither missing nor additional IDs. Local validation equals all 157 source validation image stems. No local image stem appears among the 314 official source test images. The 842 local train/test encounter groups likewise equal the 842 final training-query encounter groups.

This mapping was independently corroborated using the organizer's original [ENC-to-forum-ID map](https://github.com/wyim/DermaVQA/blob/686a5700089cce366ba940b4e91af1d2e04ce16d/data/iiyi/instanceid2encounterid.json) and [raw split-ID lists](https://github.com/wyim/DermaVQA/blob/686a5700089cce366ba940b4e91af1d2e04ce16d/data/iiyi/split2encounterids.json), pinned at DermaVQA revision `686a5700089cce366ba940b4e91af1d2e04ce16d`. The dictionary direction is `ENC identifier -> original forum instance ID`; split lists contain the latter numeric IDs. Every local test encounter joins to source train, every local training encounter joins to source train, and every local validation encounter joins to source validation. The raw training list contains 851 IDs, whereas the final query release has 842 encounters, so raw-list counts alone should not be mistaken for final released counts.

The [final training-query JSON](https://github.com/wyim/DermaVQA/blob/686a5700089cce366ba940b4e91af1d2e04ce16d/data/iiyi/train.json) lists 2,473 images. Its raw bytes match the OSF training-file SHA-256 `057ad0cb3894b5b0645cffe9a067880477ac8d5fe27275239c96ddf4cbc9c8a6`; its Git blob SHA-1 is `f4df6c8076cec9989f9bdd37fd85148930794110`. All 295 local test image IDs are listed there. The sole image present in the training ZIP but absent from query `image_ids` is `IMG_ENC00714_00005.jpg`, explicitly located at `images_final/images_train/IMG_ENC00714_00005.jpg`. It is in local train. This explains the 2,473-query-image versus 2,474-archive-image difference without guessing which image was missing. The updated 2025 validation/test query files were also checked against their OSF-published SHA-256 values and contain the same 157/314 image IDs as their respective source archive folders. Query/response text was not used for training, inference or retained in the evidence files.

All **2,631 current local image files match both the uncompressed size and CRC32 of their corresponding public ZIP member**. Their freshly calculated local SHA-256 values also match the preceding local audit. This corroborates original image payload consistency. CRC32 is not a cryptographic per-image identity proof; the entire public archive was not downloaded or independently SHA-256 verified, and the public central directory does not provide per-file SHA-256 values. No equivalent original segmentation-mask comparison has yet been established.

Sorted evidence lives outside Git in `Project/results/completion/provenance/`:

- `comparison_summary.json`: source URLs/revisions/hashes, join procedure, set-equality results, integrity counts and limitations.
- `official_zip_image_inventory.csv`: all 2,945 source image paths, original folder, uncompressed length and CRC32.
- `official_query_image_inventory.csv`: normalized train/validation/test image IDs from the linked public query metadata, excluding query/response text.
- `local_official_membership.csv`: one row per local image, local split, official source split, ENC/forum-ID join, current SHA-256, and source/local size and CRC32 comparisons.
- `original_encounter_id_mapping.json` and `original_raw_split_ids.json`: normalized copies of the two original public identifier inventories. The summary records their pinned source Git blob hashes and the normalized evidence-file SHA-256 values separately.

For example, this validates the saved membership evidence without opening image pixels:

```python
import csv
from pathlib import Path
p = Path('../Project/results/completion/provenance')
rows = list(csv.DictReader((p / 'local_official_membership.csv').open()))
official = list(csv.DictReader((p / 'official_zip_image_inventory.csv').open()))
local_training_pool = {r['image_id'] for r in rows if r['local_split'] in {'train', 'test'}}
source_training_pool = {r['image_id'] for r in official if r['official_split'] == 'train'}
assert local_training_pool == source_training_pool
assert all(r['official_archive_split'] == 'train' for r in rows if r['local_split'] == 'test')
assert all(r['archive_size_matches'] == r['archive_crc32_matches'] == 'True' for r in rows)
```

Consequently, call the 295-image evaluation a **local held-out repartition of the source training set**, and retain the previously inspected benchmark qualification. Its results do not reproduce the official held-out leaderboard. The supplied TransUNet might have been exposed to this source training pool, but no checkpoint training manifest proves that it actually was; report the exposure as unresolved rather than asserting contamination. The new from-scratch experiment's own local split and frozen-selection evidence are a separate, verifiable training protocol.

### Remaining mask transformation evidence

The focused follow-up checked the official 2025 repository inventory, the public OSF root/iiyi file inventory, and the organizer-linked competition/API endpoints. The competition and `/api/competitions/62/` endpoints both returned HTTP 404. No original multi-annotator segmentation archive, per-mask hash inventory, or COCO/RLE payload was obtained from these public sources. The later OSF `iiyi_test_extension.json` contains encounter/post identifiers and textual response records, rather than segmentation annotations; only its key schema and checksum were retained in `mask_source_availability.json`.

Therefore, the local one-mask training/validation labels cannot yet be attributed to strict majority, union, intersection, an individual original annotator, or a later edit. Image source membership and matching image checksums do not establish mask transformation identity. This limitation is recorded separately from the now-confirmed image split mapping; it does not justify guessing a merge rule.
