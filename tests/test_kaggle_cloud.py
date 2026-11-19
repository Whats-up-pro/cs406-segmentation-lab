"""State-machine tests; only the external official Kaggle CLI is substituted."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle(tmp_path):
    from scripts.prepare_kaggle import build_bundle

    data = tmp_path / "data"
    for split in ("Train", "Valid", "Test"):
        for prefix, extension in (("images", ".png"), ("labels", ".tiff")):
            path = data / (prefix + split) / (split.lower() + extension)
            path.parent.mkdir(parents=True)
            path.write_bytes((prefix + split).encode())
    output = tmp_path / "bundle"
    build_bundle(ROOT, data, output, owner="qucphmtngia", license_name="other")
    return output


class KaggleService:
    """Model network effects at the subprocess boundary, including ambiguous failures."""

    def __init__(self):
        self.dataset_exists = False
        self.kernel_exists = False
        self.kernel_status = "running"
        self.calls = []
        self.push_error_once = False
        self.has_filter = True
        self.input_artifacts = None

    def __call__(self, command, **kwargs):
        assert isinstance(command, list) and command[:3] == [sys.executable, "-m", "kaggle"]
        args = command[3:]
        self.calls.append(args)
        code, stdout, stderr = 0, "", ""
        if args == ["--version"]:
            stdout = "Kaggle API 2.2.5"
        elif args == ["kernels", "output", "--help"]:
            stdout = "--file-pattern REGEX" if self.has_filter else "--path PATH"
        elif args[:2] == ["datasets", "status"]:
            if self.dataset_exists:
                stdout = "ready"
            else:
                code, stderr = 1, "404 Client Error: Not Found"
        elif args[:2] == ["kernels", "status"]:
            if self.kernel_exists:
                stdout = f'{args[2]} has status "{self.kernel_status}"'
            else:
                code, stderr = 1, "404 Client Error: Not Found"
        elif args[:2] == ["datasets", "create"]:
            assert "--public" not in args and "-u" not in args
            self.dataset_exists = True
            manifest = json.loads((Path(args[args.index("-p") + 1]) / "bundle-manifest.json").read_text())
            self.input_artifacts = {role: {key: entry[key] for key in ("file", "bytes", "sha256")}
                                    for role, entry in manifest["artifacts"].items()}
            stdout = "Your private Dataset is being created"
        elif args[:2] == ["kernels", "push"]:
            assert args[-2:] == ["--accelerator", "NvidiaTeslaT4"]
            self.kernel_exists = True
            self.kernel_status = "running"
            if self.push_error_once:
                self.push_error_once = False
                code, stderr = 1, "Connection lost after server accepted the push"
            else:
                stdout = "Kernel version pushed"
        elif args[:2] == ["kernels", "output"]:
            assert "--file-pattern" in args
            assert args[args.index("--file-pattern") + 1] == r"(^|/)CS406_Kaggle_Artifacts[.]zip$"
            destination = Path(args[args.index("-p") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination / "CS406_Kaggle_Artifacts.zip", "w") as archive:
                archive.writestr("artifact-manifest.json", json.dumps({"status": "completed", "device": "cuda", "files": [],
                                                                        "input_archives": self.input_artifacts}))
                archive.writestr("runtime.json", '{"device":"cuda"}')
        else:
            raise AssertionError(f"Unexpected external operation: {args}")
        return subprocess.CompletedProcess(command, code, stdout, stderr)

    def count(self, command):
        return sum(args[:2] == command and "--help" not in args for args in self.calls)


def install_service(monkeypatch, service):
    from scripts import kaggle_cloud

    monkeypatch.setattr(kaggle_cloud.subprocess, "run", service)
    monkeypatch.setattr(kaggle_cloud.time, "sleep", lambda _: None)
    return kaggle_cloud


def launch(cloud, bundle, output, **kwargs):
    return cloud.run_cloud(bundle, output, Path(sys.executable), poll_interval=1, max_polls=2,
                           dataset_max_polls=2, **kwargs)


def test_launch_resume_download_records_real_state_without_duplicate_gpu_job(bundle, tmp_path, monkeypatch):
    service = KaggleService()
    cloud = install_service(monkeypatch, service)
    output = tmp_path / "cloud"
    first = launch(cloud, bundle, output)
    assert first["status"] == "running"
    assert json.loads((output / "launch-state.json").read_text())["push_attempts"] == 1
    service.kernel_status = "complete"
    second = launch(cloud, bundle, output, wait=True)
    assert second["status"] == "downloaded"
    assert second["artifact"]["device"] == "cuda"
    assert Path(second["artifact"]["path"]).is_file()
    assert len(second["artifact"]["sha256"]) == 64
    launch(cloud, bundle, output, wait=True)
    assert service.count(["datasets", "create"]) == 1
    assert service.count(["kernels", "push"]) == 1
    assert service.count(["kernels", "output"]) == 1
    state = json.loads((output / "launch-state.json").read_text())
    assert state["owner"] == "qucphmtngia" and state["status"] == "downloaded"
    assert not {"key", "token", "credentials", "environment"} & state.keys()


def test_ambiguous_push_failure_is_recovered_by_status_without_a_second_push(bundle, tmp_path, monkeypatch):
    service = KaggleService()
    service.push_error_once = True
    cloud = install_service(monkeypatch, service)
    output = tmp_path / "cloud"
    with pytest.raises(RuntimeError, match="push"):
        launch(cloud, bundle, output)
    state = json.loads((output / "launch-state.json").read_text())
    assert state["push_attempted"] is True and state["push_attempts"] == 1
    assert launch(cloud, bundle, output)["status"] == "running"
    assert service.count(["kernels", "push"]) == 1


def test_failed_job_requires_explicit_retry_and_never_recreates_dataset(bundle, tmp_path, monkeypatch):
    service = KaggleService()
    cloud = install_service(monkeypatch, service)
    output = tmp_path / "cloud"
    launch(cloud, bundle, output)
    service.kernel_status = "error"
    with pytest.raises(RuntimeError, match="ERROR|error"):
        launch(cloud, bundle, output, wait=True)
    assert service.count(["kernels", "push"]) == 1
    # A confirmed remote ERROR is the only condition permitting explicit retry.
    launch(cloud, bundle, output, retry_after_error=True)
    assert service.count(["kernels", "push"]) == 2
    assert service.count(["datasets", "create"]) == 1


def test_preexisting_unrecorded_kernel_is_never_overwritten(bundle, tmp_path, monkeypatch):
    service = KaggleService()
    service.kernel_exists = True
    cloud = install_service(monkeypatch, service)
    with pytest.raises(FileExistsError):
        launch(cloud, bundle, tmp_path / "cloud")
    assert service.count(["kernels", "push"]) == service.count(["datasets", "create"]) == 0


@pytest.mark.parametrize("change", ["public", "other_owner", "modified_archive"])
def test_invalid_local_bundle_is_rejected_before_any_cli_operation(bundle, tmp_path, monkeypatch, change):
    path = bundle / "kernel/kernel-metadata.json"
    metadata = json.loads(path.read_text())
    if change == "public":
        metadata["is_private"] = False
    elif change == "other_owner":
        metadata["id"] = "otheraccount/cs406-private-gpu-training"
    else:
        with (bundle / "dataset/cs406-data.zip").open("ab") as stream:
            stream.write(b"tampered")
    path.write_text(json.dumps(metadata))
    service = KaggleService()
    cloud = install_service(monkeypatch, service)
    with pytest.raises(ValueError):
        launch(cloud, bundle, tmp_path / "cloud")
    assert service.calls == []


def test_check_only_does_not_create_state_or_touch_remote_resources(bundle, tmp_path, monkeypatch):
    service = KaggleService()
    cloud = install_service(monkeypatch, service)
    output = tmp_path / "cloud"
    result = launch(cloud, bundle, output, check_only=True)
    assert result["status"] == "checked"
    assert not output.exists()
    assert all(args == ["--version"] or args == ["kernels", "output", "--help"] for args in service.calls)


def test_wait_is_bounded_and_resumable_without_relaunch(bundle, tmp_path, monkeypatch):
    service = KaggleService()
    cloud = install_service(monkeypatch, service)
    output = tmp_path / "cloud"
    result = launch(cloud, bundle, output, wait=True)
    assert result["status"] == "waiting"
    assert result["last_kernel_status"] == "RUNNING"
    launch(cloud, bundle, output, wait=True)
    assert service.count(["kernels", "push"]) == 1


def test_older_cli_never_downloads_unfiltered_outputs(bundle, tmp_path, monkeypatch):
    service = KaggleService()
    cloud = install_service(monkeypatch, service)
    output = tmp_path / "cloud"
    launch(cloud, bundle, output)
    service.kernel_status = "complete"
    service.has_filter = False
    with pytest.raises(RuntimeError, match="file-pattern"):
        launch(cloud, bundle, output, wait=True)
    assert service.count(["kernels", "output"]) == 0
    assert service.count(["kernels", "push"]) == 1
