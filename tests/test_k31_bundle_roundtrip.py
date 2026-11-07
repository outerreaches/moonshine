#!/usr/bin/env python3
"""Round-trip a generated K3.1 bundle through the native loader."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

FIXTURES_PATH = Path(__file__).resolve().parent / "test_build_mzg2_bundle.py"
SPEC = importlib.util.spec_from_file_location("k31_bundle_fixtures", FIXTURES_PATH)
assert SPEC and SPEC.loader
fixtures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixtures
SPEC.loader.exec_module(fixtures)
link_mzg2_store = fixtures.bundle.link_mzg2_store
synthetic_source = fixtures.synthetic_source
write_bundle_manifest = fixtures.bundle.write_bundle_manifest
write_mzg2_store = fixtures.write_mzg2_store


def main() -> None:
    binary = Path(__file__).resolve().parent / "test_k3_bundle"
    if not binary.is_file():
        raise RuntimeError("build tests/test_k3_bundle before this round-trip")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = synthetic_source()
        store = root / "source-mzg2"
        linked = root / "bundle" / "expert-store-mzg2"
        output = linked.parent
        mzg2_manifest = write_mzg2_store(store, source)
        link_mzg2_store(store, linked, 2)
        static_file = output / "model-static.safetensors"
        static_file.write_bytes(b"static")
        static_manifest = {
            "path": static_file.name,
            "tensor_count": source["static_tensor_count"],
            "payload_bytes": source["static_bytes"],
            "file_bytes": len(b"static"),
            "sha256": hashlib.sha256(b"static").hexdigest(),
            "source_layout_sha256": source["source_layout_sha256"],
        }
        manifest = write_bundle_manifest(
            output, static_manifest, mzg2_manifest, {}, source
        )
        assert manifest["source_model_layout_crc64"] == \
            source["source_model_layout_crc64"]
        completed = subprocess.run(
            [str(binary), str(output)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"native generated-bundle load failed: "
                f"{completed.stdout}{completed.stderr}"
            )
        print(completed.stdout, end="")


if __name__ == "__main__":
    main()
