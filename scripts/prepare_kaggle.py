"""Build and restore a private Kaggle input bundle without credentials or Git access.

This command only writes local files. It does not upload data, start training,
choose a dataset license, or read Kaggle authentication files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIRECTORIES = tuple(prefix + split for split in ("Train", "Valid", "Test") for prefix in ("images", "labels"))
REQUIRED_SOURCE = (
    "backend/compact_model.py", "scripts/train_compact.py",
    "scripts/run_compact_experiments.py", "scripts/evaluate_completion.py",
    "scripts/prepare_kaggle.py", "scripts/train_pixel_model.py", "requirements.txt", "requirements-ml.txt",
    "config/experiments/compact_ablation.json",
)
MAX_EXTRACTED_BYTES = 24 * 1024 ** 3
TRAINING_PROTOCOL = {
    "architecture": "CompactUNet", "image_size": 128, "base_channels": 12,
    "input_preprocessing": "RGB PIL bilinear to256x256 then128x128; masks nearest",
    "epochs": 20, "batch_size": 16, "optimizer": "AdamW",
    "learning_rate": 0.001, "weight_decay": 0.0001,
    "experiments": [
        {"dice_weight": 0, "augment": False, "seed": 406},
        {"dice_weight": 1, "augment": False, "seed": 406},
        {"dice_weight": 0, "augment": True, "seed": 406},
        {"dice_weight": 1, "augment": True, "seed": 406},
        {"dice_weight": 1, "augment": True, "seed": 407},
        {"dice_weight": 1, "augment": True, "seed": 408},
    ],
    "optimizer_split": "train", "checkpoint_and_threshold_selection_split": "valid",
    "test_policy": "Evaluate only after every training selection is frozen.",
    "pixel_baseline": {"trainer": "scripts.train_pixel_model", "epochs": 2, "seed": 406,
                       "input_size": 256, "pixels_per_image": 2048, "from_scratch": True,
                       "start_after": "All six CompactUNet selections are frozen.",
                       "threshold_selection_split": "valid"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _validate_slug(value: str, label: str) -> None:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value):
        raise ValueError(f"Invalid {label}: use a Kaggle URL slug, without slashes or spaces")


def _source_files(repo: Path) -> list[Path]:
    files = []
    for folder in ("backend", "scripts"):
        files.extend(path for path in (repo / folder).glob("*.py") if path.is_file())
    files.extend(path for path in (repo / "tests").rglob("*.py") if path.is_file()
                 and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(repo / "tests").parts))
    files.extend(path for path in (repo / "config").rglob("*.json") if path.is_file()
                 and not any(part.startswith(".") for part in path.relative_to(repo / "config").parts))
    files.extend(path for path in (repo / "models").glob("*.json") if path.is_file())
    files.extend(path for path in (repo / "frontend").rglob("*") if path.is_file()
                 and path.suffix.lower() in {".html", ".css", ".js", ".svg"}
                 and not any(part.startswith(".") for part in path.relative_to(repo / "frontend").parts))
    for name in ("README.md", "notebooks/CS406_Training_Kaggle.ipynb"):
        if (repo / name).is_file():
            files.append(repo / name)
    for name in ("requirements.txt", "requirements-ml.txt", "requirements-dev.txt"):
        if (repo / name).is_file():
            files.append(repo / name)
    # The optional supplied TransUNet evaluator imports its checked-in Python code.
    files.extend(path for path in (repo / "third_party").rglob("*")
                 if path.is_file() and (path.suffix in {".py", ".yaml", ".yml", ".json", ".txt", ".md"}
                                        or path.name in {"LICENSE", "COPYING", "NOTICE"})
                 and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(repo / "third_party").parts))
    names = {path.relative_to(repo).as_posix() for path in files}
    missing = set(REQUIRED_SOURCE) - names
    if missing:
        raise ValueError("Missing required source: " + ", ".join(sorted(missing)))
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(repo.resolve()):
            raise ValueError(f"Unsafe source symlink: {path}")
    return sorted(files)


def _data_files(data_dir: Path) -> list[Path]:
    files = []
    for name in SPLIT_DIRECTORIES:
        folder = data_dir / name
        allowed = {".jpg", ".jpeg", ".png"} if name.startswith("images") else {".tif", ".tiff"}
        selected = sorted(path for path in folder.glob("*") if path.is_file() and path.suffix.lower() in allowed)
        if not selected:
            raise ValueError(f"Missing or empty required split directory: {name}")
        if len({path.stem for path in selected}) != len(selected):
            raise ValueError(f"Duplicate stems in required split directory: {name}")
        for path in selected:
            if path.is_symlink() or not path.resolve().is_relative_to(data_dir.resolve()):
                raise ValueError(f"Unsafe data symlink: {path}")
        files.extend(selected)
    for split in ("Train", "Valid", "Test"):
        images = {path.stem for path in files if path.parent.name == "images" + split}
        masks = {re.sub(r"_mask_ann\d+$", "", path.stem) if split == "Test" else path.stem
                 for path in files if path.parent.name == "labels" + split}
        if not images <= masks:
            rule = "exact-stem or _mask_annN" if split == "Test" else "exact-stem"
            raise ValueError(f"Missing {rule} labels{split} masks for {len(images - masks)} images")
    return files


def _write_archive(destination: Path, root: Path, files: list[Path]) -> dict:
    records = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            archive.write(path, name)
            records.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"file": destination.name, "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination), "uncompressed_bytes": sum(r["bytes"] for r in records),
            "file_count": len(records), "files": records}


def build_bundle(repo: Path, data_dir: Path, output: Path, *, owner: str,
                 dataset_slug: str = "cs406-course-private-inputs",
                 kernel_slug: str = "cs406-private-gpu-training", license_name: str | None = None) -> dict:
    """Create a new local handoff; credentials, prior outputs and weights are excluded."""
    repo, data_dir, output = Path(repo).resolve(), Path(data_dir).resolve(), Path(output).resolve()
    for value, label in ((owner, "owner"), (dataset_slug, "dataset slug"), (kernel_slug, "kernel slug")):
        _validate_slug(value, label)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bundle directory: {output}")
    if license_name is not None and not license_name.strip():
        raise ValueError("An explicit nonempty dataset license name is required")
    source_files, data_files = _source_files(repo), _data_files(data_dir)
    template = PROJECT_ROOT / "notebooks/CS406_Training_Kaggle.ipynb"
    notebook = json.loads(template.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"], cell["outputs"] = None, []
    dataset_dir, kernel_dir = output / "dataset", output / "kernel"
    dataset_dir.mkdir(parents=True)
    kernel_dir.mkdir()
    artifacts = {
        "source": _write_archive(dataset_dir / "cs406-source.zip", repo, source_files),
        "data": _write_archive(dataset_dir / "cs406-data.zip", data_dir, data_files),
    }
    dataset_ref = f"{owner}/{dataset_slug}"
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_ref": dataset_ref, "private": True, "artifacts": artifacts,
        "training_protocol": TRAINING_PROTOCOL,
        "data_provenance": "Private coursework copy. The original OSF DermaVQA (2024) release declares CC BY 4.0; mapping to the 2025 DermaVQA-DAS masks, the supplied local archive, transformed-mask mapping and checkpoint history remain unconfirmed. No additional redistribution rights are asserted.",
        "original_release": {"project": "https://api.osf.io/v2/nodes/72rp3/",
                             "declared_license": "CC BY 4.0",
                             "license_record": "https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/",
                             "local_archive_mapping_verified": False},
        "excluded": ["credentials", "environment files", "prior run outputs", "supplied pretrained checkpoint"],
    }
    _json_write(dataset_dir / "bundle-manifest.json", manifest)
    if license_name is not None:
        _json_write(dataset_dir / "dataset-metadata.json", {
            "title": "CS406 Private Segmentation Inputs", "id": dataset_ref,
            "description": manifest["data_provenance"] + " Restricted to private research coursework.",
            "licenses": [{"name": license_name}],
        })
    _json_write(kernel_dir / "CS406_Training_Kaggle.ipynb", notebook)
    _json_write(kernel_dir / "kernel-metadata.json", {
        "id": f"{owner}/{kernel_slug}", "title": kernel_slug.replace("-", " "),
        "code_file": "CS406_Training_Kaggle.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [dataset_ref], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    })
    return {"output": str(output), "dataset_ref": dataset_ref, "kernel_ref": f"{owner}/{kernel_slug}",
            "source_bytes": artifacts["source"]["bytes"], "data_bytes": artifacts["data"]["bytes"],
            "upload_metadata_ready": license_name is not None, "uploaded": False}


def safe_extract_zip(archive_path: Path, destination: Path, *, max_bytes: int = MAX_EXTRACTED_BYTES) -> None:
    """Preflight all paths before creating files, then stream regular members only."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite extraction directory: {destination}")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        seen = set()
        total = 0
        for info in members:
            # ZipInfo normalizes platform separators on Windows; inspect the
            # stored name as well so the exact same archive is portable.
            raw = info.orig_filename
            parts = raw.rstrip("/").split("/")
            mode = info.external_attr >> 16
            if (not raw or raw.startswith("/") or "\\" in raw or ":" in raw
                    or any(part in {"", ".", ".."} for part in parts)
                    or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR})):
                raise ValueError(f"Unsafe ZIP member: {raw}")
            name = PurePosixPath(*parts).as_posix()
            if name.casefold() in seen:
                raise ValueError(f"Unsafe duplicate ZIP member: {raw}")
            seen.add(name.casefold())
            total += info.file_size
            if total > max_bytes:
                raise ValueError("Unsafe ZIP expansion exceeds the permitted byte limit")
        # File/directory collisions can otherwise depend on extraction order.
        file_names = {i.filename.rstrip("/").casefold() for i in members if not i.is_dir()}
        for info in members:
            parts = PurePosixPath(info.filename.rstrip("/")).parts
            if any(PurePosixPath(*parts[:index]).as_posix().casefold() in file_names for index in range(1, len(parts))):
                raise ValueError(f"Unsafe ZIP file/directory collision: {info.filename}")
        destination.mkdir(parents=True)
        for info in members:
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)


def _manifest_records(artifact: dict) -> list[dict]:
    records = artifact.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Missing per-file bundle manifest records")
    seen, total = set(), 0
    for record in records:
        name = record.get("path")
        if (not isinstance(name, str) or not name or name.startswith("/") or "\\" in name or ":" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or any(ord(character) < 32 for character in name)):
            raise ValueError(f"Unsafe bundle manifest path: {name}")
        if name.casefold() in seen:
            raise ValueError(f"Unsafe duplicate bundle manifest path: {name}")
        seen.add(name.casefold())
        size, digest = record.get("bytes"), record.get("sha256")
        if (type(size) is not int or size < 0 or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError(f"Invalid file integrity record: {name}")
        total += size
    if total > MAX_EXTRACTED_BYTES or artifact.get("file_count") != len(records) or artifact.get("uncompressed_bytes") != total:
        raise ValueError("Invalid bundle manifest file totals")
    for name in seen:
        parts = PurePosixPath(name).parts
        if any(PurePosixPath(*parts[:index]).as_posix() in seen for index in range(1, len(parts))):
            raise ValueError(f"Unsafe bundle manifest file/directory collision: {name}")
    return records


def _confined_file(base: Path, name: str) -> Path:
    """Reject links and paths escaping the selected expanded input root."""
    target = base.joinpath(*PurePosixPath(name).parts)
    for path in (base, *[base.joinpath(*PurePosixPath(name).parts[:index])
                         for index in range(1, len(PurePosixPath(name).parts) + 1)]):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError(f"Unsafe expanded input link: {path}")
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Unsafe expanded input path: {name}")
    return target


def _verify_records(base: Path, records: list[dict]) -> None:
    for record in records:
        path = _confined_file(base, record["path"])
        if not path.is_file():
            raise ValueError(f"Missing recorded bundle file: {record['path']}")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"SHA-256 or byte-size mismatch: {record['path']}")


def restore_bundle(input_dir: Path, work_dir: Path) -> dict:
    """Restore intact ZIPs or provider-expanded inputs using recorded file hashes."""
    input_dir, work_dir = Path(input_dir).resolve(), Path(work_dir).resolve()
    if work_dir.exists():
        raise FileExistsError(f"Refusing to overwrite restored bundle: {work_dir}")
    manifest = json.loads((input_dir / "bundle-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("private") is not True:
        raise ValueError("Unsupported or nonprivate Kaggle bundle manifest")
    inputs, layouts = {}, {}
    for role, filename in (("source", "cs406-source.zip"), ("data", "cs406-data.zip")):
        artifact = manifest["artifacts"][role]
        records = _manifest_records(artifact)
        if artifact.get("file") != filename:
            raise ValueError(f"Unexpected {role} archive name")
        path = _confined_file(input_dir, filename)
        if path.is_file():
            if path.stat().st_size != artifact.get("bytes") or sha256_file(path) != artifact.get("sha256"):
                raise ValueError(f"SHA-256 or byte-size mismatch: {filename}")
            with zipfile.ZipFile(path) as archive:
                names = [info.filename for info in archive.infolist() if not info.is_dir()]
            if len(names) != len(records) or set(names) != {record["path"] for record in records}:
                raise ValueError(f"Archive file list differs from manifest: {filename}")
            inputs[role], layouts[role] = path, "zip"
        else:
            for candidate, layout in ((input_dir / Path(filename).stem, "expanded-stem"), (input_dir, "expanded-flat")):
                files = [_confined_file(candidate, record["path"]) for record in records]
                if all(file.is_file() for file in files):
                    _verify_records(candidate, records)
                    inputs[role], layouts[role] = candidate, layout
                    break
            else:
                raise ValueError(f"Missing required {role} ZIP or complete expanded input files")
    # Expanded files for both roles are verified before any output is created.
    for role in ("source", "data"):
        destination = work_dir / role
        records = manifest["artifacts"][role]["files"]
        if layouts[role] == "zip":
            safe_extract_zip(inputs[role], destination)
        else:
            destination.mkdir(parents=True)
            for record in records:
                target = destination.joinpath(*PurePosixPath(record["path"]).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_confined_file(inputs[role], record["path"]), target)
        _verify_records(destination, records)
    _source_files(work_dir / "source")
    _data_files(work_dir / "data")
    _json_write(work_dir / "bundle-manifest.json", manifest)
    return {"source_dir": work_dir / "source", "data_dir": work_dir / "data", "manifest": manifest,
            "input_layouts": layouts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True, help="Actual Kaggle account URL slug")
    parser.add_argument("--dataset-slug", default="cs406-course-private-inputs")
    parser.add_argument("--kernel-slug", default="cs406-private-gpu-training")
    parser.add_argument("--license-name", help="Explicit Kaggle license category; omission leaves dataset upload metadata unwritten")
    args = parser.parse_args(argv)
    result = build_bundle(args.repo, args.data_dir, args.output, owner=args.owner,
                          dataset_slug=args.dataset_slug, kernel_slug=args.kernel_slug, license_name=args.license_name)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
