# Allen Cell MegaSeg Nahual OCI image

Build the reproducible archive and load it into Podman or Docker:

```console
nix build .#oci-image
podman load < result                         # or: docker load < result
```

The image is tagged `nahual/megaseg:local` and listens on TCP port 5555. The
released MegaSeg checkpoint is downloaded from Allen Cell's S3 bucket on first
setup, so persist `/tmp/nahual` as a model cache:

```console
podman run --rm --device nvidia.com/gpu=all -p 5555:5555 \
  -v nahual-megaseg-cache:/tmp/nahual nahual/megaseg:local
```

For Docker, replace the CDI device option with `--gpus all`. CPU operation is
supported, although 3-D inference is substantially faster on a GPU. With Nahual
and NumPy installed on the host, run pretrained end-to-end inference with:

```console
NAHUAL_DEVICE=cpu python oci/smoke_test.py
```

A custom trusted Lightning checkpoint can instead be mounted read-only and
selected with the setup request's `checkpoint_path` parameter.
