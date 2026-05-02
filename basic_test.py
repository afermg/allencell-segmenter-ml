"""Standalone smoke test for the MegaSeg Nahual wrap.

Loads the model the same way `server.py` does and runs a forward pass on a
small synthetic 3-D volume. Does NOT spin up the IPC server.

Run from the repo root:
    nix develop --impure --command python basic_test.py

Expected: prints info dict with `device: cuda:0` and a uint8 mask shape.
"""

import sys

# server.py reads sys.argv[1] at import time; inject a placeholder so that
# importing it from here doesn't crash.
if len(sys.argv) < 2:
    sys.argv.append("ipc:///tmp/megaseg_basic_test.ipc")

import numpy  # noqa: E402

from server import setup  # noqa: E402


def main() -> None:
    processor, info = setup()
    print(f"setup: {info}")
    assert "cuda" in info["device"], (
        f"Not on GPU! info['device']={info['device']!r}"
    )

    # MegaSeg is 3-D. Use a small (1, 1, Z, Y, X) volume that's at least the
    # patch shape (64, 128, 128) so sliding-window inference has something to
    # do.
    numpy.random.seed(0)
    data = numpy.random.random_sample((1, 1, 64, 128, 128)).astype(numpy.float32)
    out = processor(data)
    print(f"process: {type(out).__name__} dtype={out.dtype} shape={out.shape}")


if __name__ == "__main__":
    main()
