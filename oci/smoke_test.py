#!/usr/bin/env python3
"""End-to-end pretrained inference against the MegaSeg OCI container."""

import json
import os

os.environ.setdefault("NAHUAL_IPC_TIMEOUT_MS", "1800000")

import numpy as np
from nahual.process import dispatch_setup_process


def main() -> None:
    address = os.environ.get("NAHUAL_ADDRESS", "tcp://127.0.0.1:5555")
    device = os.environ.get("NAHUAL_DEVICE", "cpu")
    setup, process = dispatch_setup_process("megaseg")
    info = setup(
        {"device": device, "overlap": 0.0, "sw_batch_size": 1},
        address=address,
    )
    pixels = np.random.default_rng(42).random((1, 1, 64, 128, 128), dtype=np.float32)
    result = process(pixels, address=address)
    assert info["device"] == device, info
    assert info["model_type"] == "megaseg", info
    assert result.shape == pixels.shape, result.shape
    assert result.dtype == np.uint8, result.dtype
    assert set(np.unique(result)).issubset({0, 1})
    print(
        json.dumps(
            {
                "setup": info,
                "shape": list(result.shape),
                "foreground_voxels": int(result.sum()),
            }
        )
    )


if __name__ == "__main__":
    main()
