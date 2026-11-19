"""Behavior tests for the private, credential-free Kaggle handoff."""
import ast
import json
from pathlib import Path
import stat
import subprocess
import sys
import textwrap
import zipfile

import pytest


REPO = Path(__file__).resolve().parents[1]


def make_inputs(tmp_path):
    source = tmp_path / "source"
    for name, content in {
        "scripts/train_compact.py": "# train entry point\n",
        "scripts/run_compact_experiments.py": "# experiment entry point\n",
        "scripts/evaluate_completion.py": "# evaluation entry point\n",
        "scripts/train_pixel_model.py": "# pixel training entry point\n",
        "scripts/prepare_kaggle.py": (REPO / "scripts/prepare_kaggle.py").read_text() if (REPO / "scripts/prepare_kaggle.py").exists() else "# packaging entry point\n",
        "backend/compact_model.py": "# model implementation\n",
        "tests/contracts/test_portable.py": "# portable verification\n",
        "config/experiments/compact_ablation.json": '{"epochs":20}',
        "models/transunet-checkpoint.json": '{"training_data_history_verified":false}',
        "third_party/TransUNet/LICENSE": "upstream license retained\n",
        "frontend/index.html": "<main>Private segmentation interface</main>",
        "frontend/assets/app.js": "console.log('ready');",
        "frontend/styles.css": "main { color: black; }",
        "frontend/favicon.svg": '<svg xmlns="http://www.w3.org/2000/svg"/>',
        "README.md": "# Portable source\n",
        "notebooks/CS406_Training_Kaggle.ipynb": (REPO / "notebooks/CS406_Training_Kaggle.ipynb").read_text(),
        "requirements.txt": "numpy>=2\nPillow>=11\n",
        "requirements-ml.txt": "torch>=2.6\n",
        "requirements-dev.txt": "pytest>=8,<10\n",
        ".env": "KAGGLE_KEY=must-not-leave-machine\n",
        ".kaggle/kaggle.json": '{"key":"must-not-leave-machine"}',
        "scripts/private.key": "must-not-leave-machine",
        "backend/__pycache__/private.pyc": "must-not-leave-machine",
        "results/frozen-selection.json": "must-not-leave-machine",
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    data = tmp_path / "data"
    for split in ("Train", "Valid", "Test"):
        for prefix, suffix in (("images", ".png"), ("labels", ".tiff")):
            path = data / (prefix + split) / (split.lower() + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((split + prefix).encode())
    (data / "supplied.pth").write_bytes(b"not needed for fresh compact training")
    return source, data


def test_bundle_round_trip_excludes_credentials_and_preserves_all_six_splits(tmp_path):
    from scripts.prepare_kaggle import build_bundle, restore_bundle

    source, data = make_inputs(tmp_path)
    output = tmp_path / "handoff"
    result = build_bundle(source, data, output, owner="courseowner", dataset_slug="course-private-data")
    assert result["dataset_ref"] == "courseowner/course-private-data"
    assert not (output / "dataset/dataset-metadata.json").exists()
    metadata = json.loads((output / "kernel/kernel-metadata.json").read_text())
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == ["courseowner/course-private-data"]
    with zipfile.ZipFile(output / "dataset/cs406-source.zip") as archive:
        assert set(archive.namelist()) == {
            "scripts/train_compact.py", "scripts/run_compact_experiments.py", "scripts/evaluate_completion.py", "scripts/prepare_kaggle.py",
            "scripts/train_pixel_model.py",
            "backend/compact_model.py", "requirements.txt", "requirements-ml.txt", "requirements-dev.txt",
            "tests/contracts/test_portable.py", "config/experiments/compact_ablation.json",
            "models/transunet-checkpoint.json", "third_party/TransUNet/LICENSE",
            "frontend/index.html", "frontend/assets/app.js", "frontend/styles.css", "frontend/favicon.svg",
            "README.md", "notebooks/CS406_Training_Kaggle.ipynb",
        }
        assert b"must-not-leave-machine" not in b"".join(archive.read(n) for n in archive.namelist())
    restored = restore_bundle(output / "dataset", tmp_path / "restored")
    assert (restored["source_dir"] / "backend/compact_model.py").read_text() == "# model implementation\n"
    assert (restored["data_dir"] / "labelsTest/test.tiff").read_bytes() == b"Testlabels"
    assert not (restored["data_dir"] / "supplied.pth").exists()
    notebook = json.loads((output / "kernel/CS406_Training_Kaggle.ipynb").read_text())
    assert all(c["execution_count"] is None and c["outputs"] == [] for c in notebook["cells"] if c["cell_type"] == "code")


def test_missing_split_fails_before_any_bundle_is_written(tmp_path):
    from scripts.prepare_kaggle import build_bundle

    source, data = make_inputs(tmp_path)
    (data / "labelsTest/test.tiff").unlink()
    with pytest.raises(ValueError, match="labelsTest"):
        build_bundle(source, data, tmp_path / "out", owner="courseowner", dataset_slug="course-private-data")
    assert not (tmp_path / "out").exists()


def test_bundle_preserves_all_test_annotations_with_supplied_suffix_naming(tmp_path):
    from scripts.prepare_kaggle import build_bundle, restore_bundle

    source, data = make_inputs(tmp_path)
    (data / "labelsTest/test.tiff").unlink()
    for index in range(1, 4):
        (data / f"labelsTest/test_mask_ann{index}.tiff").write_bytes(f"annotation {index}".encode())
    output = tmp_path / "handoff"
    build_bundle(source, data, output, owner="courseowner", dataset_slug="course-private-data")
    restored = restore_bundle(output / "dataset", tmp_path / "restored")
    labels = sorted((restored["data_dir"] / "labelsTest").iterdir())
    assert [path.name for path in labels] == ["test_mask_ann1.tiff", "test_mask_ann2.tiff", "test_mask_ann3.tiff"]
    assert [path.read_bytes() for path in labels] == [b"annotation 1", b"annotation 2", b"annotation 3"]


@pytest.mark.parametrize("layout", ["flat", "stem"])
def test_server_expanded_bundle_restores_verified_files_without_original_zip(tmp_path, layout):
    from scripts.prepare_kaggle import build_bundle, restore_bundle

    source, data = make_inputs(tmp_path)
    output = tmp_path / "handoff"
    build_bundle(source, data, output, owner="courseowner", dataset_slug="course-private-data")
    input_dir = output / "dataset"
    for name in ("cs406-source", "cs406-data"):
        archive_path = input_dir / (name + ".zip")
        destination = input_dir if layout == "flat" else input_dir / name
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)  # controlled fixture models provider ingestion
        archive_path.unlink()
    restored = restore_bundle(input_dir, tmp_path / "restored")
    assert (restored["source_dir"] / "backend/compact_model.py").read_bytes() == b"# model implementation\n"
    assert (restored["data_dir"] / "labelsTest/test.tiff").read_bytes() == b"Testlabels"
    assert restored["input_layouts"] == {"source": "expanded-" + layout, "data": "expanded-" + layout}


def test_server_expanded_tampered_file_fails_before_restoration(tmp_path):
    from scripts.prepare_kaggle import build_bundle, restore_bundle

    source, data = make_inputs(tmp_path)
    output = tmp_path / "handoff"
    build_bundle(source, data, output, owner="courseowner", dataset_slug="course-private-data")
    input_dir = output / "dataset"
    for name in ("cs406-source", "cs406-data"):
        path = input_dir / (name + ".zip")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(input_dir / name)
        path.unlink()
    # Same byte length proves the integrity check uses the recorded digest.
    (input_dir / "cs406-data/labelsTest/test.tiff").write_bytes(b"fakepixels")
    with pytest.raises(ValueError, match="SHA-256"):
        restore_bundle(input_dir, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_server_expanded_manifest_cannot_reference_outside_input(tmp_path):
    from scripts.prepare_kaggle import build_bundle, restore_bundle

    source, data = make_inputs(tmp_path)
    output = tmp_path / "handoff"
    build_bundle(source, data, output, owner="courseowner", dataset_slug="course-private-data")
    path = output / "dataset/bundle-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["data"]["files"][0]["path"] = "../outside.png"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe.*path"):
        restore_bundle(output / "dataset", tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_server_expanded_links_are_rejected_before_any_copy(tmp_path):
    from scripts.prepare_kaggle import build_bundle, restore_bundle

    source, data = make_inputs(tmp_path)
    output = tmp_path / "handoff"
    build_bundle(source, data, output, owner="courseowner", dataset_slug="course-private-data")
    input_dir = output / "dataset"
    path = input_dir / "cs406-data.zip"
    with zipfile.ZipFile(path) as archive:
        archive.extractall(input_dir / "cs406-data")
    path.unlink()
    linked = input_dir / "cs406-data/labelsTest/test.tiff"
    linked.unlink()
    target = tmp_path / "outside.tiff"
    target.write_bytes(b"Testlabels")
    try:
        linked.symlink_to(target)
    except OSError as error:
        pytest.skip(f"Runtime cannot create a filesystem symlink: {error}")
    with pytest.raises(ValueError, match="Unsafe.*link"):
        restore_bundle(input_dir, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("bad_name", ["../outside.txt", "/outside.txt", "C:/outside.txt", "folder\\outside.txt", "a/../../outside.txt"])
def test_zip_traversal_is_rejected_before_writing_any_member(tmp_path, bad_name):
    from scripts.prepare_kaggle import safe_extract_zip

    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("safe.txt", "safe")
        member = zipfile.ZipInfo("member")
        member.filename = bad_name  # preserve a raw backslash member on Windows
        archive.writestr(member, "bad")
    destination = tmp_path / "extract"
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        safe_extract_zip(archive_path, destination)
    assert not destination.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_zip_symlink_and_duplicate_paths_are_rejected(tmp_path):
    from scripts.prepare_kaggle import safe_extract_zip

    archive_path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "../outside.txt")
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        safe_extract_zip(archive_path, tmp_path / "extract-link")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("folder/a", "one")
        archive.writestr("folder/./a", "two")
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        safe_extract_zip(archive_path, tmp_path / "extract-duplicate")


def test_modified_archive_is_rejected_before_restoration(tmp_path):
    from scripts.prepare_kaggle import build_bundle, restore_bundle

    source, data = make_inputs(tmp_path)
    output = tmp_path / "handoff"
    build_bundle(source, data, output, owner="courseowner", dataset_slug="course-private-data")
    with (output / "dataset/cs406-data.zip").open("ab") as stream:
        stream.write(b"unexpected modification")
    with pytest.raises(ValueError, match="SHA-256"):
        restore_bundle(output / "dataset", tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_cli_builds_upload_metadata_only_for_explicit_license_and_refuses_overwrite(tmp_path):
    source, data = make_inputs(tmp_path)
    output = tmp_path / "handoff"
    command = [sys.executable, "-m", "scripts.prepare_kaggle", "--repo", str(source), "--data-dir", str(data),
               "--output", str(output), "--owner", "courseowner", "--dataset-slug", "course-private-data",
               "--license-name", "other"]
    first = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    metadata = json.loads((output / "dataset/dataset-metadata.json").read_text())
    assert metadata["licenses"] == [{"name": "other"}]
    assert "unconfirmed" in metadata["description"].lower()
    marker = output / "reviewed.txt"
    marker.write_text("keep")
    second = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert second.returncode != 0
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("layout", ["zip", "flat", "stem"])
def test_notebook_restores_real_bundle_and_stops_on_failed_training(tmp_path, monkeypatch, layout):
    from scripts.prepare_kaggle import build_bundle

    source, data = make_inputs(tmp_path)
    (source / "scripts/run_compact_experiments.py").write_text(
        'raise RuntimeError("controlled training failure")\n', encoding="utf-8")
    output = tmp_path / "handoff"
    build_bundle(source, data, output, owner="courseowner", dataset_slug="course-private-data")
    if layout != "zip":
        for name in ("cs406-source", "cs406-data"):
            path = output / "dataset" / (name + ".zip")
            with zipfile.ZipFile(path) as archive:
                archive.extractall(output / "dataset" if layout == "flat" else output / "dataset" / name)
            path.unlink()
    monkeypatch.setenv("CS406_INPUT_ROOT", str(output / "dataset"))
    monkeypatch.setenv("CS406_WORK_ROOT", str(tmp_path / "working"))
    # Bootstrap intentionally changes cwd/sys.path so notebook imports resolve
    # from restored source. Keep that normal behavior in its own interpreter.
    script = textwrap.dedent("""
        from pathlib import Path
        import json
        import os
        import subprocess

        notebook_path = Path(os.environ["CS406_INPUT_ROOT"]).parent / "kernel/CS406_Training_Kaggle.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        namespace = {}
        exec(compile(cells["restore-private-inputs"], "notebook-bootstrap", "exec"), namespace)
        assert (namespace["DATA_DIR"] / "imagesTest/test.png").read_bytes() == b"Testimages"
        exec(compile(cells["runtime-and-protocol"], "notebook-runtime", "exec"), namespace)
        try:
            exec(compile(cells["train-six-experiments"], "notebook-training", "exec"), namespace)
        except subprocess.CalledProcessError:
            pass
        else:
            raise AssertionError("Failed training must stop notebook execution")
    """)
    result = subprocess.run([sys.executable, "-"], input=script, cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    output_dir = tmp_path / "working/outputs"
    assert "controlled training failure" in (output_dir / "training.log").read_text()
    assert not (output_dir / "pixel_classifier").exists()
    assert not (output_dir / "evaluation").exists()
    assert not (output_dir / "artifact-manifest.json").exists()


def test_notebook_trains_fresh_pixel_baseline_only_after_six_frozen_compact_runs(tmp_path):
    import numpy as np
    from PIL import Image

    notebook = json.loads((REPO / "notebooks/CS406_Training_Kaggle.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    pixel_cell = cells["train-pixel-baseline"]
    data = tmp_path / "data"
    for index, split in enumerate(("Train", "Valid", "Test"), 1):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[6:24, 4 + index:20 + index] = 1
        image = np.repeat((mask * 220 + index * 5)[:, :, None], 3, axis=2)
        name = f"IMG_ENC{index}_1"
        for prefix, pixels, suffix in (("images", image, ".png"), ("labels", mask, ".tiff")):
            folder = data / (prefix + split)
            folder.mkdir(parents=True)
            Image.fromarray(pixels).save(folder / (name + suffix))
    output = tmp_path / "outputs"
    output.mkdir()
    namespace = {"SOURCE_DIR": REPO, "DATA_DIR": data, "OUTPUT_DIR": output,
                 "sys": sys, "json": json, "frozen_paths": []}
    parsed = ast.parse(cells["train-six-experiments"])
    helpers = ast.Module(body=[node for node in parsed.body if isinstance(node, (ast.Import, ast.FunctionDef))], type_ignores=[])
    exec(compile(helpers, "notebook-subprocess-helper", "exec"), namespace)
    with pytest.raises(RuntimeError, match="six|Six"):
        exec(compile(pixel_cell, "notebook-pixel-training", "exec"), namespace)
    assert not (output / "pixel_classifier").exists()
    namespace["frozen_paths"] = []
    for index in range(6):
        path = tmp_path / "runs" / str(index) / "frozen-selection.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"status":"frozen"}')
        namespace["frozen_paths"].append(path)
    exec(compile(pixel_cell, "notebook-pixel-training", "exec"), namespace)
    model = json.loads((output / "pixel_classifier/model.json").read_text())
    manifest = json.loads((output / "pixel_classifier/manifest.json").read_text())
    assert model["training_origin"]["epochs"] == 2
    assert model["training_origin"]["seed"] == 406
    assert model["training_origin"]["optimizer_splits"] == ["train"]
    assert model["training_origin"]["threshold_split"] == "valid"
    assert manifest["status"] == "complete" and len(manifest["history"]) == 2
