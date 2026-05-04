"""
This example uses a server within the environment defined on
`https://github.com/afermg/allencell-segmenter-ml.git`.

Run `nix run github:afermg/allencell-segmenter-ml -- ipc:///tmp/megaseg.ipc`
from any directory, or
`nix develop --command bash -c "python server.py ipc:///tmp/megaseg.ipc"` from
the root of that repository.

MegaSeg is the structure-agnostic 3-D fluorescence segmentation model from the
Allen Institute (DynUNet via cyto-dl). It expects a 5-D NCZYX float32 volume
with C=1 (a single fluorescent channel) and returns a binary uint8 mask of the
same shape.
"""

import numpy

from nahual.process import dispatch_setup_process

# `megaseg` isn't in the built-in registry, so pass the signature explicitly.
setup, process = dispatch_setup_process("megaseg", signature=("dict", "numpy"))
address = "ipc:///tmp/megaseg.ipc"

# %% Load model server-side
parameters = {
    # All optional. Server downloads the MegaSeg checkpoint from S3 on first
    # call and caches it in $MEGASEG_CACHE_DIR (default ~/.cache/megaseg).
    # "checkpoint_path": "/path/to/epoch_650.ckpt",
    # "device": 0,
    # "overlap": 0.0,
    # "sw_batch_size": 1,
}
response = setup(parameters, address=address)
print(response)
# Expected:
# {'device': 'cuda:0', 'model_type': 'megaseg', ..., 'patch_shape': [64, 128, 128],
#  'input_axes': 'NCZYX', 'output_axes': 'NCZYX', 'spatial_dims': 3, ...}

# %% Define custom data
# 3-D segmentation: (N, C=1, Z, Y, X). Z/Y/X must be at least the patch shape
# (64, 128, 128) for sliding-window inference to behave well.
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 1, 64, 128, 128)).astype(numpy.float32)

result = process(data, address=address)
print(f"Shape: {result.shape}, dtype: {result.dtype}, max: {result.max()}")
# Expected: Shape: (1, 1, 64, 128, 128), dtype: uint8
