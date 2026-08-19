#!/usr/bin/env python3
"""Idempotently prefetch all public model artifacts into persistent storage."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import hf_hub_download, snapshot_download


_SNAPSHOT_SHA256 = {
    "VLM_MODEL_ID": {
        "model-00001-of-00002.safetensors": "4f75e3de726546ee43620d1227d3596cd3ba0fdd19f11faeea71de578d2d1052",
        "model-00002-of-00002.safetensors": "dae4128bbfd2b8d489e838048edc0bbe6e31f269d9b96fa3effe11cc534b8f0c",
    },
    "GDINO_MODEL_ID": {
        "model.safetensors": "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21",
    },
}

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_with_retry(label: str, operation):
    attempts = int(os.environ.get("MODEL_DOWNLOAD_ATTEMPTS", "30"))
    base_delay = float(os.environ.get("MODEL_DOWNLOAD_RETRY_DELAY_S", "5"))
    if attempts < 1 or attempts > 100:
        raise RuntimeError("MODEL_DOWNLOAD_ATTEMPTS must be within [1,100]")
    if not 0.0 <= base_delay <= 60.0:
        raise RuntimeError("MODEL_DOWNLOAD_RETRY_DELAY_S must be within [0,60]")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(base_delay * attempt, 60.0)
            print(
                f"retrying {label} after {type(exc).__name__} "
                f"({attempt}/{attempts}, delay={delay:g}s)",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable retry loop")


def _prefetch_pinned_artifact(
    repo: str,
    revision: str,
    relative_path: str,
    expected_sha: str,
) -> None:
    """Resume a large, content-addressed Hub artifact with bounded parallel HTTP."""
    transport = os.environ.get("MODEL_DOWNLOAD_TRANSPORT", "aria2")
    if transport not in {"aria2", "hub"}:
        raise RuntimeError("MODEL_DOWNLOAD_TRANSPORT must be aria2 or hub")
    if transport == "hub":
        return
    if not _REPO_ID_RE.fullmatch(repo):
        raise RuntimeError("unsafe Hugging Face repository id")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("model revision must be a full lowercase Git SHA")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", relative_path):
        raise RuntimeError("unsafe pinned artifact filename")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise RuntimeError("invalid pinned artifact SHA-256")
    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c is required for resumable model recovery")

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    if not endpoint.startswith("https://"):
        raise RuntimeError("HF_ENDPOINT must use HTTPS")
    cache_root = Path(os.environ.get("HF_HUB_CACHE", "/data/huggingface/hub"))
    blob_dir = cache_root / ("models--" + repo.replace("/", "--")) / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob = blob_dir / expected_sha
    partial = blob_dir / (expected_sha + ".incomplete")
    if blob.is_file():
        if blob.is_symlink() or _sha256(blob) != expected_sha:
            raise RuntimeError(f"cached artifact SHA-256 mismatch: {relative_path}")
        return
    if partial.exists() and (not partial.is_file() or partial.is_symlink()):
        raise RuntimeError(f"unsafe partial artifact path: {relative_path}")

    connections = int(os.environ.get("MODEL_DOWNLOAD_CONNECTIONS", "8"))
    if connections < 1 or connections > 16:
        raise RuntimeError("MODEL_DOWNLOAD_CONNECTIONS must be within [1,16]")
    url = (
        f"{endpoint}/{quote(repo, safe='/')}/resolve/{revision}/"
        f"{quote(relative_path, safe='')}"
    )

    def download() -> None:
        subprocess.run(
            [
                "aria2c",
                "--allow-overwrite=false",
                "--auto-file-renaming=false",
                "--check-certificate=true",
                "--connect-timeout=30",
                "--console-log-level=warn",
                "--continue=true",
                "--file-allocation=none",
                "--http-accept-gzip=false",
                f"--max-connection-per-server={connections}",
                "--max-tries=10",
                "--min-split-size=16M",
                "--retry-wait=5",
                f"--split={connections}",
                "--summary-interval=30",
                "--timeout=120",
                f"--dir={blob_dir}",
                f"--out={partial.name}",
                url,
            ],
            check=True,
        )

    _download_with_retry(f"pinned artifact {repo}/{relative_path}", download)
    if _sha256(partial) != expected_sha:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded artifact SHA-256 mismatch: {relative_path}")
    partial.replace(blob)
    print(f"ready pinned artifact: {repo}/{relative_path}", flush=True)


def _snapshot(repo_env: str, revision_env: str) -> None:
    repo = os.environ[repo_env]
    revision = os.environ[revision_env]
    for relative_path, expected_sha in _SNAPSHOT_SHA256.get(repo_env, {}).items():
        _prefetch_pinned_artifact(repo, revision, relative_path, expected_sha)
    options: dict[str, object] = {}
    # Transformers loads the safetensors artifact. Avoid caching the duplicate
    # legacy PyTorch pickle from Grounding DINO during disaster recovery.
    if repo_env == "GDINO_MODEL_ID":
        options["ignore_patterns"] = ["*.bin"]
    max_workers = int(os.environ.get("HF_HUB_MAX_WORKERS", "2"))
    if max_workers < 1 or max_workers > 8:
        raise RuntimeError("HF_HUB_MAX_WORKERS must be within [1,8]")
    path = _download_with_retry(
        f"snapshot {repo}@{revision}",
        lambda: snapshot_download(
            repo_id=repo,
            revision=revision,
            max_workers=max_workers,
            **options,
        ),
    )
    snapshot = Path(path)
    for relative_path, expected_sha in _SNAPSHOT_SHA256.get(repo_env, {}).items():
        artifact = snapshot / relative_path
        if not artifact.is_file() or _sha256(artifact) != expected_sha:
            raise RuntimeError(f"snapshot artifact SHA-256 mismatch: {relative_path}")
    print(f"ready {repo}@{revision}: {snapshot}")


def _sam2() -> None:
    target = Path(os.environ.get("SAM2_CHECKPOINT", "/data/checkpoints/sam2.1_hiera_small.pt"))
    target.parent.mkdir(parents=True, exist_ok=True)
    sam2_repo = os.environ.get("SAM2_REPO_ID", "facebook/sam2.1-hiera-small")
    sam2_revision = os.environ["SAM2_MODEL_REVISION"]
    expected_sha = os.environ["SAM2_CHECKPOINT_SHA256"]
    _prefetch_pinned_artifact(
        sam2_repo,
        sam2_revision,
        "sam2.1_hiera_small.pt",
        expected_sha,
    )
    downloaded = Path(
        _download_with_retry(
            f"SAM2 checkpoint {sam2_repo}@{sam2_revision}",
            lambda: hf_hub_download(
                repo_id=sam2_repo,
                revision=sam2_revision,
                filename="sam2.1_hiera_small.pt",
            ),
        )
    )
    if _sha256(downloaded) != expected_sha:
        raise RuntimeError("downloaded SAM2 checkpoint SHA-256 mismatch")
    if target.exists() and _sha256(target) == expected_sha:
        print(f"ready SAM2 checkpoint: {target}")
        return
    temporary = target.with_suffix(target.suffix + ".partial")
    shutil.copyfile(downloaded, temporary)
    if _sha256(temporary) != expected_sha:
        raise RuntimeError("copied SAM2 checkpoint SHA-256 mismatch")
    temporary.replace(target)
    print(f"ready SAM2 checkpoint: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--perception", action="store_true")
    args = parser.parse_args()
    if not (args.all or args.vlm or args.perception):
        parser.error("select --all, --vlm or --perception")
    if args.all or args.vlm:
        _snapshot("VLM_MODEL_ID", "VLM_MODEL_REVISION")
    if args.all or args.perception:
        _snapshot("GDINO_MODEL_ID", "GDINO_MODEL_REVISION")
        _sam2()


if __name__ == "__main__":
    main()
