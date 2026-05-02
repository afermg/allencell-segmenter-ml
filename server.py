"""Nahual server for the AllenCell MegaSeg / cyto-dl segmentation model.

Wraps the upstream `allencell-segmenter-ml` napari plugin, but bypasses the
napari/Qt UI and the file-CSV inference path. Loads the MegaSeg DynUNet
checkpoint directly via PyTorch Lightning, then runs MONAI sliding-window
inference on in-memory NCZYX numpy volumes.

Run with:
    nix run --impure . -- ipc:///tmp/megaseg.ipc
or:
    python server.py ipc:///tmp/megaseg.ipc
"""

import os
import sys
import urllib.request
import zipfile
from functools import partial
from pathlib import Path
from typing import Callable

import numpy
import pynng
import torch
import trio
from monai.inferers import sliding_window_inference
from nahual.server import responder

# server.py captures argv[1] at import time; basic_test.py injects a
# placeholder before importing this module.
address = sys.argv[1]


# Default S3 endpoint for MegaSeg, copied from
# allencell_ml_segmenter.utils.s3.s3_bucket_constants.PROD_BUCKET (no Qt import).
_MEGASEG_URL = (
    "https://production-aics-ml-segmenter-models.s3.us-west-2.amazonaws.com/"
    "megaSeg.zip"
)
_DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "MEGASEG_CACHE_DIR",
        Path.home() / ".cache" / "megaseg",
    )
)


def _ensure_megaseg_checkpoint(cache_dir: Path) -> Path:
    """Download + unzip the MegaSeg checkpoint if missing. Returns the
    absolute path to ``epoch_650.ckpt``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cache_dir / "megaSeg" / "checkpoints" / "epoch_650.ckpt"
    if ckpt.exists():
        return ckpt
    zip_path = cache_dir / "megaSeg.zip"
    if not zip_path.exists():
        print(f"Downloading MegaSeg checkpoint to {zip_path}", flush=True)
        urllib.request.urlretrieve(_MEGASEG_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected checkpoint at {ckpt} after extraction")
    return ckpt


def _build_model(checkpoint_path: Path, torch_device: torch.device):
    """Construct the MegaSeg ``MultiTaskIm2Im`` LightningModule and load the
    checkpoint weights.

    Configuration mirrors ``megaSeg/train_config.yaml`` but is hard-wired here
    so we don't have to instantiate Hydra / pyrootutils on the server side.
    """
    # Local imports keep the import surface of this module small and prove
    # cyto_dl never pulls napari/Qt.
    from cyto_dl.models.im2im import MultiTaskIm2Im
    from cyto_dl.models.im2im.utils.postprocessing import ActThreshLabel
    from cyto_dl.nn.head.mask_head import MaskHead
    from monai.losses import MaskedDiceLoss
    from monai.networks.nets import DynUNet

    spatial_dims = 3
    raw_im_channels = 1
    patch_shape = [64, 128, 128]

    backbone = DynUNet(
        spatial_dims=spatial_dims,
        in_channels=raw_im_channels,
        out_channels=1,
        strides=[1, 2, 2, 2, 2],
        kernel_size=[3, 3, 3, 3, 3],
        upsample_kernel_size=[2, 2, 2, 2],
        dropout=0.0,
        res_block=True,
    )

    head = MaskHead(
        loss=MaskedDiceLoss(sigmoid=True),
        mask_key="exclude_mask",
        postprocess={
            "input": ActThreshLabel(rescale_dtype=numpy.uint8),
            "prediction": ActThreshLabel(
                activation=torch.nn.Sigmoid(),
                rescale_dtype=numpy.uint8,
            ),
        },
    )

    model = MultiTaskIm2Im(
        backbone=backbone,
        task_heads={"seg": head},
        x_key="raw",
        save_dir="./",
        save_images_every_n_epochs=1,
        inference_args={
            "sw_batch_size": 1,
            "roi_size": patch_shape,
            "overlap": 0,
            "mode": "gaussian",
        },
        optimizer={},
        lr_scheduler={},
    )

    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=False)
    model.to(torch_device).eval()
    return model, patch_shape


def setup(
    checkpoint_path: str | None = None,
    cache_dir: str | None = None,
    device: int | None = None,
    overlap: float = 0.0,
    sw_batch_size: int = 1,
) -> tuple[Callable, dict]:
    """Load the MegaSeg model and return ``(processor_partial, info_dict)``.

    Parameters
    ----------
    checkpoint_path
        Path to a ``.ckpt`` file. If ``None``, downloads MegaSeg from S3 into
        ``cache_dir``.
    cache_dir
        Directory to store / look up the auto-downloaded MegaSeg bundle.
        Defaults to ``$MEGASEG_CACHE_DIR`` or ``~/.cache/megaseg``.
    device
        CUDA device index. ``0`` by default. CPU fallback if no CUDA.
    overlap, sw_batch_size
        MONAI sliding-window inference knobs the client may override.
    """
    if device is None:
        device = 0
    if torch.cuda.is_available():
        torch_device = torch.device(int(device))
    else:
        torch_device = torch.device("cpu")

    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    if checkpoint_path is None:
        ckpt = _ensure_megaseg_checkpoint(cache)
    else:
        ckpt = Path(checkpoint_path)

    model, patch_shape = _build_model(ckpt, torch_device)
    # Patch in any client-side overrides for the sliding-window inferer.
    model.hparams.inference_args["overlap"] = float(overlap)
    model.hparams.inference_args["sw_batch_size"] = int(sw_batch_size)

    info = {
        "device": str(torch_device),
        "model_type": "megaseg",
        "checkpoint": str(ckpt),
        "patch_shape": patch_shape,
        "input_axes": "NCZYX",
        "output_axes": "NCZYX",
        "spatial_dims": 3,
        "raw_im_channels": 1,
    }

    processor = partial(
        process,
        model=model,
        device=torch_device,
        patch_shape=patch_shape,
    )
    return processor, info


def _normalize_intensity(t: torch.Tensor) -> torch.Tensor:
    """MONAI ``NormalizeIntensityd`` (channel-wise) equivalent for the predict
    transform pipeline in MegaSeg's train config.
    """
    # t shape: (N, C, Z, Y, X). Normalize per (sample, channel).
    spatial = tuple(range(2, t.ndim))
    mean = t.mean(dim=spatial, keepdim=True)
    std = t.std(dim=spatial, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (t - mean) / std


def process(
    pixels: numpy.ndarray,
    model,
    device: torch.device,
    patch_shape,
) -> numpy.ndarray:
    """Run MegaSeg inference on a 5-D NCZYX numpy volume.

    Returns a 5-D ``(N, 1, Z, Y, X)`` ``uint8`` mask array (one channel for
    the single segmentation head).
    """
    if pixels.ndim != 5:
        raise ValueError(f"Expected NCZYX (5D) array, got shape {pixels.shape}")
    n, c, z, y, x = pixels.shape
    if c != 1:
        raise ValueError(
            f"MegaSeg expects a single channel (raw_im_channels=1); got C={c}"
        )

    tensor = torch.from_numpy(numpy.ascontiguousarray(pixels)).to(device).float()
    tensor = _normalize_intensity(tensor)

    with torch.no_grad():
        raw = sliding_window_inference(
            inputs=tensor,
            roi_size=patch_shape,
            sw_batch_size=int(model.hparams.inference_args["sw_batch_size"]),
            predictor=model.forward,
            overlap=float(model.hparams.inference_args["overlap"]),
            mode=str(model.hparams.inference_args["mode"]),
            run_heads=["seg"],
        )

    seg_logits = raw["seg"]  # (N, 1, Z, Y, X)
    seg = (torch.sigmoid(seg_logits) > 0.5).to(torch.uint8)
    return seg.cpu().numpy()


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"MegaSeg server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(responder, setup=setup), sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
