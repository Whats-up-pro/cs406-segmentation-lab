# Private Kaggle GPU run

`notebooks/CS406_Training_Kaggle.ipynb` runs the fixed CompactUNet experiment plan from a private mounted input dataset. It uses the packaged source rather than cloning GitHub, and uses Kaggle's installed PyTorch rather than downloading a runtime. Source notebook cells have no execution counts, outputs, or claimed results before a real run.

The [original OSF DermaVQA (2024) release](https://api.osf.io/v2/nodes/72rp3/) declares [CC BY 4.0](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/). Mapping from that 2024 corpus to the 2025 DermaVQA-DAS masks, the supplied local archive, transformed-mask mapping and checkpoint history remain unconfirmed; see [the verified provenance note](data-provenance-verified.md). The handoff keeps the course images, notebook and generated outputs private and asserts no additional redistribution rights. Kaggle requires a license category in dataset metadata; the command below explicitly selects `other` for this unresolved local mapping. Credentials are read only by the normal Kaggle client outside the bundle.

## Build the local handoff

Run in the repository directory, replacing the account slug with the actual Kaggle account:

```powershell
$KaggleOwner = 'YOUR_KAGGLE_OWNER'
python -m scripts.prepare_kaggle `
  --data-dir ../Project/data `
  --output ../Project/output/completion/kaggle-private `
  --owner $KaggleOwner `
  --dataset-slug cs406-course-private-inputs `
  --kernel-slug cs406-private-gpu-training `
  --license-name other
```

The output directory must be new. The script validates required source entry points, all six nonempty split directories, exact-stem Train/Valid pairs and Test annotations matched by exact stem or the supplied `_mask_annN` suffix before writing. Every supplied Test annotation is preserved byte for byte. It builds:

```text
../Project/output/completion/kaggle-private/
  dataset/
    cs406-source.zip
    cs406-data.zip
    bundle-manifest.json
    dataset-metadata.json
  kernel/
    CS406_Training_Kaggle.ipynb
    kernel-metadata.json
```

The manifest records archive and per-file sizes/hashes and the training protocol. Source includes Python modules under `backend`, `scripts` and `tests`, JSON experiment/team configuration and model manifests, frontend HTML/CSS/JS/SVG, README, this source notebook, runtime/development requirements, and checked-in TransUNet support code with its attribution/license files. Data includes only `imagesTrain`, `labelsTrain`, `imagesValid`, `labelsValid`, `imagesTest`, and `labelsTest`. The 421 MB supplied pretrained checkpoint is excluded from fresh CompactUNet training. Local credentials, environment files, Git metadata, caches and old results are excluded.

Omitting `--license-name` still builds a local review bundle but leaves `dataset-metadata.json` unwritten. The checked-in `kaggle/kernel-metadata.json` is a placeholder; the build writes account-specific metadata into the handoff directory.

## Upload and run with the official client

Use an authenticated official `kaggle` CLI. Keep credentials in its normal user-level configuration or environment; do not copy them into the handoff directory. Dataset creation defaults to private, and the generated notebook metadata explicitly sets `is_private: true`, `enable_gpu: true` and `enable_internet: false`. See the official [dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md), [dataset metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md), and [notebook metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).

```powershell
kaggle datasets create -p ../Project/output/completion/kaggle-private/dataset
kaggle datasets status "$KaggleOwner/cs406-course-private-inputs"
kaggle kernels push -p ../Project/output/completion/kaggle-private/kernel --accelerator NvidiaTeslaT4
kaggle kernels status "$KaggleOwner/cs406-private-gpu-training"
kaggle kernels output "$KaggleOwner/cs406-private-gpu-training" -p ../Project/output/completion/kaggle-results
```

Run the push once the private dataset is ready. `kernels push` normally starts execution; adding `--no-run` saves a notebook version without executing it. Accelerator access depends on the account and available quota. These behaviors and accelerator names are documented in the official [notebook commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

Alternatively, upload the files in `dataset/` as a **private** Kaggle Dataset, import `kernel/CS406_Training_Kaggle.ipynb`, attach that private dataset, and select a GPU accelerator. The notebook expects exactly one `bundle-manifest.json` under `/kaggle/input`. It accepts intact ZIP inputs or server-expanded files at the dataset root or in `cs406-source/` and `cs406-data/` stem folders. Intact archives must match their recorded hashes/sizes; every restored payload must match its per-file manifest hash/size. Expanded input paths are confined to the mounted dataset and links are rejected. ZIP extraction also rejects unsafe paths, links and duplicates. Restored copies go into a new directory under `/kaggle/working/cs406`. Reuse a fresh session or a new `CS406_WORK_ROOT` when restarting restoration.

## Fixed experiments and evaluation

Every run uses 128 × 128 RGB images, 12 base channels, batch size 16, 20 epochs and AdamW with learning rate 0.001 and weight decay 0.0001:

| Loss | Augmentation | Seed |
| --- | --- | --- |
| BCE | off | 406 |
| BCE + Dice | off | 406 |
| BCE | on | 406 |
| BCE + Dice | on | 406 |
| BCE + Dice | on | 407 |
| BCE + Dice | on | 408 |

The notebook runs the actual project commands:

```text
python -m scripts.run_compact_experiments --data-dir DATA --cache-dir CACHE --output RUNS --device DEVICE --epochs 20 --cpu-threads 4
python -m scripts.train_pixel_model --data-dir DATA --output OUTPUT/pixel_classifier --epochs 2 --seed 406
python -m scripts.evaluate_completion --data-dir DATA --runs-dir RUNS --output OUTPUT/evaluation --device DEVICE --pixel-model OUTPUT/pixel_classifier/model.json
```

`DEVICE` is `cuda` only when `torch.cuda.is_available()` is true; otherwise it is `cpu`. `runtime.json` records the chosen device, actual GPU name and package versions. The runner uses Train for optimizer updates and Valid for checkpoint/threshold selection. The notebook checks that all six 20-epoch runs have frozen selections before training a fresh two-epoch pixel logistic baseline at seed 406. The existing pixel trainer uses Train for updates, freezes its Valid-selected threshold, then performs its test evaluation. The common evaluator receives this newly generated model through `--pixel-model`, so the old local logistic weights are never distributed. A failing subprocess raises an error and preserves its real log.

For the optional supplied TransUNet comparison, attach its checkpoint separately as a private input and set `REFERENCE_CHECKPOINT` in the notebook to its mounted path. That file is passed only to evaluation with `--reference-checkpoint`; CompactUNet starts from a seeded fresh initialization.

Download `CS406_Kaggle_Artifacts.zip` from Kaggle Output. It contains `runtime.json`, compact/pixel training and evaluation logs, each compact run's checkpoint/history/frozen selection, the fresh pixel classifier and selection manifest, evaluator outputs, and an artifact manifest with SHA-256 hashes. Retain the Kaggle run URL/version with the downloaded files. GPU execution is established by the runtime record and completed artifacts, not by the source notebook's GPU request.

## Resumable official CLI orchestration

The local helper uses only the official CLI and its normal authentication profile:

```powershell
python -m scripts.kaggle_cloud `
  --bundle ../Project/output/completion/kaggle-private `
  --output ../Project/output/completion/kaggle `
  --kaggle-python PATH/TO/ISOLATED/KAGGLE/python.exe `
  --wait
```

`--check-only` validates local input hashes, private metadata, source counts and CLI availability without creating state or remote resources. The helper is scoped to the configured course account `qucphmtngia`; dataset/notebook IDs, source references and the state fingerprint must agree. It refuses existing unrecorded targets and never publishes a dataset or notebook.

The helper saves `launch-state.json` before each potentially ambiguous create/push. Repeating the command resumes status/download and preserves the existing GPU job. A second job requires `--retry-after-error` and a confirmed failed remote kernel. Polling is bounded: defaults allow 60 dataset polls and 720 notebook polls at 30-second intervals. When the bound expires, the same command resumes safely. Keep the state directory with the bundle.

After completion, it downloads only `CS406_Kaggle_Artifacts.zip`, verifies its input-archive identity and actual runtime device, and records its SHA-256. CPU fallback is reported as `downloaded_cpu` with exit code 2. Kaggle 2.2.5 supports the official `kernels output --file-pattern` option; an older client lacking it stops with an upgrade instruction, preserving the completed job. The helper never falls back to downloading every working file. See the official [filtered output command](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#kaggle-kernels-output).

Kaggle's [upload implementation](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py) and directory-upload options do not guarantee that an uploaded ZIP remains mounted as a ZIP. The notebook supports both preserved and expanded payloads, so no undocumented ingestion setting is required.

The artifact ZIP also retains `training-cache-manifest.json`, copied from the actual training cache after all compact runs freeze. Array caches are excluded from the artifact ZIP.

Local verification:

```powershell
python -m pytest tests/test_kaggle_package.py tests/test_kaggle_cloud.py -q
```

The tests exercise real archive creation/restoration and CLI behavior, including secret exclusion, all six splits, missing inputs, real Test annotation names, unsafe ZIP members, expanded input hashes/links, tampering and overwrite refusal. CLI state tests substitute only the external Kaggle service and verify bounded polling, persisted state, filtered downloads and prevention of duplicate GPU pushes. They do not upload data or start the training plan.
