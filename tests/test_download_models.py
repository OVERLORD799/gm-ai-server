from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "download_models.py"
SPEC = importlib.util.spec_from_file_location("download_models", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
download_models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_models)


class DownloadModelsTests(unittest.TestCase):
    def test_retry_resumes_after_transient_errors(self) -> None:
        calls = 0

        def operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ConnectionError("transient")
            return "ready"

        with mock.patch.dict(
            os.environ,
            {"MODEL_DOWNLOAD_ATTEMPTS": "3", "MODEL_DOWNLOAD_RETRY_DELAY_S": "0"},
        ):
            self.assertEqual(
                download_models._download_with_retry("fixture", operation), "ready"
            )
        self.assertEqual(calls, 3)

    def test_parallel_prefetch_verifies_and_promotes_blob(self) -> None:
        content = b"content-addressed fixture"
        digest = hashlib.sha256(content).hexdigest()
        revision = "1" * 40
        with tempfile.TemporaryDirectory() as cache:
            environment = {
                "HF_ENDPOINT": "https://huggingface.co",
                "HF_HUB_CACHE": cache,
                "MODEL_DOWNLOAD_ATTEMPTS": "1",
                "MODEL_DOWNLOAD_CONNECTIONS": "4",
                "MODEL_DOWNLOAD_TRANSPORT": "aria2",
            }

            def fake_run(arguments: list[str], *, check: bool) -> None:
                self.assertTrue(check)
                directory = Path(next(x[6:] for x in arguments if x.startswith("--dir=")))
                output = next(x[6:] for x in arguments if x.startswith("--out="))
                (directory / output).write_bytes(content)

            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(download_models.shutil, "which", return_value="/usr/bin/aria2c"),
                mock.patch.object(download_models.subprocess, "run", side_effect=fake_run),
            ):
                download_models._prefetch_pinned_artifact(
                    "owner/model", revision, "model.safetensors", digest
                )

            blob = Path(cache) / "models--owner--model" / "blobs" / digest
            self.assertEqual(blob.read_bytes(), content)
            self.assertFalse(blob.with_name(digest + ".incomplete").exists())

    def test_rejects_non_https_endpoint(self) -> None:
        with (
            tempfile.TemporaryDirectory() as cache,
            mock.patch.dict(
                os.environ,
                {
                    "HF_ENDPOINT": "http://example.invalid",
                    "HF_HUB_CACHE": cache,
                    "MODEL_DOWNLOAD_TRANSPORT": "aria2",
                },
                clear=False,
            ),
            mock.patch.object(download_models.shutil, "which", return_value="/usr/bin/aria2c"),
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTPS"):
                download_models._prefetch_pinned_artifact(
                    "owner/model", "1" * 40, "model.safetensors", "2" * 64
                )

    def test_rejects_unknown_transport(self) -> None:
        with mock.patch.dict(
            os.environ, {"MODEL_DOWNLOAD_TRANSPORT": "typo"}, clear=False
        ):
            with self.assertRaisesRegex(RuntimeError, "aria2 or hub"):
                download_models._prefetch_pinned_artifact(
                    "owner/model", "1" * 40, "model.safetensors", "2" * 64
                )


if __name__ == "__main__":
    unittest.main()
