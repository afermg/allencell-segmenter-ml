{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  pdm-pep517,
  setuptools,
  # Runtime deps actually used by the inference path (MultiTaskIm2Im + MaskHead).
  torch,
  torchvision,
  numpy,
  monai,
  lightning,
  hydra-core,
  omegaconf,
  torchmetrics,
  scikit-image,
  scikit-learn,
  matplotlib,
  pandas,
  tqdm,
  pyyaml,
  einops,
  fire,
  rich,
  pyarrow,
}:
buildPythonPackage rec {
  pname = "cyto-dl";
  version = "0.6.2";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "AllenCellModeling";
    repo = "cyto-dl";
    rev = "v${version}";
    sha256 = "sha256-WDpxGrDM+aFS0S6e+xOLcuH0f1/O1bsuGXpDMwJrS1s=";
  };

  build-system = [
    pdm-pep517
    setuptools
  ];

  # We only ship the bits the MegaSeg inference path needs. Heavy upstream
  # deps (bioio, mlflow, ome-zarr, hydra-optuna-sweeper, edt, online-stats,
  # pyrootutils, opencv, positional-encodings, anndata, astropy, ...) are
  # only used by training/IO code we never enter from server.py.
  dependencies = [
    torch
    torchvision
    numpy
    monai
    lightning
    hydra-core
    omegaconf
    torchmetrics
    scikit-image
    scikit-learn
    matplotlib
    pandas
    tqdm
    pyyaml
    einops
    fire
    rich
    pyarrow
  ];

  pythonImportsCheck = [
    "cyto_dl"
    "cyto_dl.models.im2im"
    "cyto_dl.nn.head"
  ];

  # Upstream pyproject pins runtime versions we don't satisfy with nixpkgs
  # (bioio, opencv-python, etc). The inference path doesn't import them.
  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;
  doCheck = false;

  meta = {
    description = "AllenCellModeling collection of representation learning models (inference subset).";
    homepage = "https://github.com/AllenCellModeling/cyto-dl";
    license = lib.licenses.bsd3;
  };
}
