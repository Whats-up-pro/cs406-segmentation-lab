"""Launch this private course bundle once through the official Kaggle CLI.

Authentication stays entirely in the official client's normal user profile.
The local state records operation outcomes and hashes, never credentials.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time
import zipfile

from scripts.prepare_kaggle import REQUIRED_SOURCE, SPLIT_DIRECTORIES, TRAINING_PROTOCOL, _manifest_records, sha256_file


EXPECTED_OWNER = "qucphmtngia"
ARTIFACT_NAME = "CS406_Kaggle_Artifacts.zip"
OUTPUT_PATTERN = r"(^|/)CS406_Kaggle_Artifacts[.]zip$"
ERROR_STATUSES = {"ERROR", "FAILED", "CANCELLED", "CANCELED"}
COMPLETE_STATUSES = {"COMPLETE", "COMPLETED", "SUCCESS"}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bundle_file(bundle: Path, name: str) -> Path:
    path = bundle / name
    for index in range(1, len(PurePosixPath(name).parts) + 1):
        component = bundle.joinpath(*PurePosixPath(name).parts[:index])
        if component.is_symlink() or getattr(component, "is_junction", lambda: False)():
            raise ValueError(f"Unsafe bundle link: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(bundle):
        raise ValueError(f"Missing or unsafe bundle file: {name}")
    return path


def validate_bundle(bundle: Path) -> dict:
    """Check private targets, archive identity, required source and declared counts."""
    bundle = Path(bundle).resolve()
    manifest_path = _bundle_file(bundle, "dataset/bundle-manifest.json")
    dataset_path = _bundle_file(bundle, "dataset/dataset-metadata.json")
    kernel_path = _bundle_file(bundle, "kernel/kernel-metadata.json")
    notebook_path = _bundle_file(bundle, "kernel/CS406_Training_Kaggle.ipynb")
    manifest, dataset, kernel = map(_read_json, (manifest_path, dataset_path, kernel_path))
    if manifest.get("schema_version") != 1 or manifest.get("private") is not True:
        raise ValueError("Expected a private supported bundle manifest")
    if (kernel.get("is_private") is not True or kernel.get("enable_gpu") is not True
            or kernel.get("enable_internet") is not False):
        raise ValueError("Notebook must explicitly request private GPU execution without Internet")
    if ("id_no" in dataset or "id_no" in kernel or dataset.get("is_private") is False
            or dataset.get("isPublic") is True or dataset.get("public") is True):
        raise ValueError("Ambiguous numeric targets or public dataset metadata are forbidden")
    dataset_ref, kernel_ref = dataset.get("id"), kernel.get("id")
    for value in (dataset_ref, kernel_ref):
        if not isinstance(value, str) or not re.fullmatch(r"qucphmtngia/[a-z0-9][a-z0-9-]*", value):
            raise ValueError(f"Both resource owners must be the configured course account {EXPECTED_OWNER}")
    if manifest.get("dataset_ref") != dataset_ref or kernel.get("dataset_sources") != [dataset_ref]:
        raise ValueError("Dataset references differ between bundle and notebook metadata")
    if kernel.get("code_file") != notebook_path.name or kernel.get("kernel_type") != "notebook" or kernel.get("language") != "python":
        raise ValueError("Notebook entry point metadata differs from the packaged source")
    if dataset.get("licenses") != [{"name": "other"}]:
        raise ValueError("Keep the explicitly selected other category for the unresolved local mapping")
    if manifest.get("training_protocol") != TRAINING_PROTOCOL:
        raise ValueError("Bundle does not declare the fixed six-run course protocol")
    counts = {}
    for role, filename in (("source", "cs406-source.zip"), ("data", "cs406-data.zip")):
        path = _bundle_file(bundle, "dataset/" + filename)
        artifact = manifest["artifacts"][role]
        records = _manifest_records(artifact)
        if (artifact.get("file") != filename or path.stat().st_size != artifact.get("bytes")
                or sha256_file(path) != artifact.get("sha256")):
            raise ValueError(f"Archive byte-size or SHA-256 mismatch: {filename}")
        expected = {record["path"]: record for record in records}
        with zipfile.ZipFile(path) as archive:
            entries = [info for info in archive.infolist() if not info.is_dir()]
            if len(entries) != len(expected) or {info.filename for info in entries} != set(expected):
                raise ValueError(f"Archive file counts/paths differ from its manifest: {filename}")
            for info in entries:
                mode = info.external_attr >> 16
                if (info.orig_filename != info.filename or stat.S_ISLNK(mode)
                        or stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
                    raise ValueError(f"Unsafe archive member: {info.orig_filename}")
                if info.file_size != expected[info.filename]["bytes"]:
                    raise ValueError(f"Archive member byte-size mismatch: {info.filename}")
            if role == "source":
                if not set(REQUIRED_SOURCE) <= set(expected):
                    raise ValueError("Required runtime source is missing from the archive")
                import hashlib
                for info in entries:
                    if hashlib.sha256(archive.read(info)).hexdigest() != expected[info.filename]["sha256"]:
                        raise ValueError(f"Source member SHA-256 mismatch: {info.filename}")
            else:
                folders = [PurePosixPath(name).parts for name in expected]
                if any(len(parts) != 2 or parts[0] not in SPLIT_DIRECTORIES for parts in folders):
                    raise ValueError("Data archive must contain only the six declared split folders")
                if any(not any(parts[0] == name for parts in folders) for name in SPLIT_DIRECTORIES):
                    raise ValueError("Data archive is missing a required split")
        counts[role + "_files"] = len(records)
    notebook = _read_json(notebook_path)
    if any(cell.get("outputs") or cell.get("execution_count") is not None
           for cell in notebook["cells"] if cell["cell_type"] == "code"):
        raise ValueError("Upload the unexecuted source notebook, without claimed execution outputs")
    return {"owner": EXPECTED_OWNER, "dataset_ref": dataset_ref, "kernel_ref": kernel_ref,
            "manifest_sha256": sha256_file(manifest_path), "notebook_sha256": sha256_file(notebook_path),
            "dataset_metadata_sha256": sha256_file(dataset_path), "kernel_metadata_sha256": sha256_file(kernel_path),
            "counts": counts, "input_artifacts": {key: {field: entry[field] for field in ("file", "bytes", "sha256")}
                                                 for key, entry in manifest["artifacts"].items()}}


def _cli(python: Path, arguments: list[str], timeout=60):
    return subprocess.run([str(python), "-m", "kaggle", *map(str, arguments)],
                          capture_output=True, text=True, timeout=timeout, check=False)


def _checked(python, arguments, timeout=60):
    result = _cli(python, arguments, timeout)
    if result.returncode:
        # Do not persist or echo arbitrary client tracebacks containing local configuration.
        raise RuntimeError(f"Kaggle {' '.join(arguments[:2])} failed (exit {result.returncode}); inspect the official CLI's authentication/network settings")
    return result.stdout


def _status(python, kind, reference):
    result = _cli(python, [kind, "status", reference])
    if result.returncode:
        message = result.stdout + result.stderr
        if re.search(r"\b404\b|not found", message, re.I) and not re.search(r"\b401\b|\b403\b", message):
            return None
        raise RuntimeError(f"Kaggle {kind} status failed (exit {result.returncode}); no mutation was retried")
    text = result.stdout.strip()
    match = re.search(r'has status\s+[\"\x27]([^\"\x27]+)', text, re.I)
    if match:
        value = match.group(1)
    else:
        try:
            value = json.loads(text)["status"]
        except (ValueError, TypeError, KeyError):
            value = text.splitlines()[-1] if text else "unknown"
    return value.rsplit(".", 1)[-1].strip().upper()


def _save(path: Path, state: dict) -> None:
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _launch_lock(output: Path):
    path = output / "launch.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write("One launcher owns this output directory.\n")
    except FileExistsError:
        raise RuntimeError("Another launcher owns launch.lock; inspect that process before removing a stale lock") from None
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _download(python, output, state, state_path):
    help_text = _checked(python, ["kernels", "output", "--help"])
    if "--file-pattern" not in help_text:
        state["status"] = "awaiting_cli_filter"
        _save(state_path, state)
        raise RuntimeError("Official CLI lacks --file-pattern; install the provided Kaggle 2.2.5 wheel, then rerun to download without relaunching")
    download_dir = output / "downloads"
    existing = list(download_dir.rglob(ARTIFACT_NAME)) if download_dir.exists() else []
    if not existing:
        print("Downloading only", ARTIFACT_NAME, flush=True)
        _checked(python, ["kernels", "output", state["kernel_ref"], "-p", str(download_dir),
                          "--file-pattern", OUTPUT_PATTERN], timeout=3600)
        existing = list(download_dir.rglob(ARTIFACT_NAME))
    if len(existing) != 1:
        raise RuntimeError("Kaggle output did not contain exactly one requested artifact ZIP")
    path = existing[0]
    if path.is_symlink() or not path.resolve().is_relative_to(output):
        raise ValueError("Unsafe downloaded artifact path")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("artifact-manifest.json"))
        runtime = json.loads(archive.read("runtime.json"))
    if manifest.get("status") != "completed" or manifest.get("device") != runtime.get("device"):
        raise ValueError("Downloaded artifacts do not establish a completed consistent execution")
    if manifest.get("input_archives") != state["input_artifacts"]:
        raise ValueError("Downloaded artifacts were produced from a different input bundle")
    state["artifact"] = {"path": str(path.resolve()), "bytes": path.stat().st_size,
                         "sha256": sha256_file(path), "device": runtime["device"]}
    state["status"] = "downloaded" if runtime["device"] == "cuda" else "downloaded_cpu"
    _save(state_path, state)
    print("Artifacts saved:", path, "device:", runtime["device"], flush=True)
    return state


def run_cloud(bundle: Path, output: Path, kaggle_python: Path, *, wait=False, check_only=False,
              retry_after_error=False, poll_interval=30, max_polls=720, dataset_max_polls=60) -> dict:
    bundle, output, kaggle_python = Path(bundle).resolve(), Path(output).resolve(), Path(kaggle_python).resolve()
    identity = validate_bundle(bundle)
    if not kaggle_python.is_file():
        raise FileNotFoundError("The isolated official Kaggle CLI Python executable is missing")
    if output.is_relative_to(bundle):
        raise ValueError("Cloud state/output must be outside the upload bundle")
    if not 1 <= poll_interval <= 60 or min(max_polls, dataset_max_polls) < 1:
        raise ValueError("Use bounded positive polling, with an interval between 1 and 60 seconds")
    _checked(kaggle_python, ["--version"])
    if check_only:
        _checked(kaggle_python, ["kernels", "output", "--help"])
        return {"status": "checked", **identity}
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "launch-state.json"
    with _launch_lock(output):
        if state_path.exists():
            state = _read_json(state_path)
            if any(state.get(key) != value for key, value in identity.items()):
                raise ValueError("Existing cloud state belongs to a different bundle, owner or target")
            artifact = state.get("artifact")
            if artifact:
                path = Path(artifact["path"])
                if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                    raise FileExistsError("Recorded downloaded artifact is missing or changed; preserve it and inspect before retrying")
                return state
        else:
            state = {**identity, "status": "initialized", "dataset_create_attempted": False,
                     "push_attempted": False, "push_succeeded": False, "push_attempts": 0}
            _save(state_path, state)
        try:
            current = _status(kaggle_python, "kernels", state["kernel_ref"])
            retry = current in ERROR_STATUSES and retry_after_error and state["push_attempted"]
            if not state["push_attempted"]:
                if current is not None:
                    raise FileExistsError("An unrecorded notebook already exists at this target; refusing to overwrite it")
                if not state["dataset_create_attempted"]:
                    if _status(kaggle_python, "datasets", state["dataset_ref"]) is not None:
                        raise FileExistsError("An unrecorded dataset already exists at this target; refusing to replace it")
                    state["dataset_create_attempted"], state["status"], state["operation"] = True, "creating_dataset", "datasets create"
                    _save(state_path, state)
                    print("Creating private dataset:", state["dataset_ref"], flush=True)
                    _checked(kaggle_python, ["datasets", "create", "-p", str(bundle / "dataset")], timeout=3600)
            elif current is None and not state["push_succeeded"]:
                raise RuntimeError("Previous push outcome is uncertain and no job is visible; inspect the target before any retry")
            if not state["push_attempted"] or retry:
                for index in range(dataset_max_polls):
                    dataset_status = _status(kaggle_python, "datasets", state["dataset_ref"])
                    print(f"Dataset {state['dataset_ref']}: {dataset_status or 'not visible'} ({index + 1}/{dataset_max_polls})", flush=True)
                    if dataset_status == "READY":
                        break
                    if dataset_status in ERROR_STATUSES:
                        raise RuntimeError(f"Kaggle dataset reported {dataset_status}; no notebook was pushed")
                    if index + 1 < dataset_max_polls:
                        time.sleep(poll_interval)
                else:
                    state["status"] = "waiting_dataset"
                    _save(state_path, state)
                    return state
                state["push_attempted"], state["push_succeeded"] = True, False
                state["push_attempts"] += 1
                state["status"], state["operation"] = "pushing", "kernels push"
                _save(state_path, state)  # persist intent before a potentially ambiguous network result
                print("Starting private GPU notebook:", state["kernel_ref"], flush=True)
                _checked(kaggle_python, ["kernels", "push", "-p", str(bundle / "kernel"),
                                         "--accelerator", "NvidiaTeslaT4"], timeout=600)
                state["push_succeeded"] = True
                _save(state_path, state)
                current = None
            for index in range(max_polls if wait else 1):
                if current is None or index:
                    current = _status(kaggle_python, "kernels", state["kernel_ref"])
                state["last_kernel_status"] = current or "PENDING_VISIBLE"
                print(f"Notebook {state['kernel_ref']}: {state['last_kernel_status']} ({index + 1}/{max_polls if wait else 1})", flush=True)
                if current in ERROR_STATUSES:
                    state["status"] = "error"
                    _save(state_path, state)
                    raise RuntimeError(f"Kaggle kernel reported {current}; an explicit --retry-after-error is required for another GPU job")
                if current in COMPLETE_STATUSES:
                    return _download(kaggle_python, output, state, state_path)
                state["status"] = "running"
                _save(state_path, state)
                if not wait:
                    return state
                if index + 1 < max_polls:
                    time.sleep(poll_interval)
                current = None
            state["status"] = "waiting"
            _save(state_path, state)
            print("Bounded wait ended; rerun the same command to resume status/download without another push.", flush=True)
            return state
        except Exception as error:
            state["last_error"] = {"kind": type(error).__name__, "operation": state.get("operation")}
            _save(state_path, state)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kaggle-python", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--retry-after-error", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--max-polls", type=int, default=720)
    parser.add_argument("--dataset-max-polls", type=int, default=60)
    result = run_cloud(**vars(parser.parse_args(argv)))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] != "downloaded_cpu" else 2


if __name__ == "__main__":
    raise SystemExit(main())
