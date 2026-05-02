{
  inputs = {
    # MONAI 1.5.2, lightning 2.6.2, hydra-core 1.3.2 are all in nixos-unstable
    # — exactly the versions cyto-dl wants. No fallback to 24.11 needed.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    pynng-flake.url = "github:afermg/pynng";
    pynng-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            # GPU is mandatory for MegaSeg.
            cudaSupport = true;
          };
        };
        cytoDl = pkgs.python3.pkgs.callPackage ./nix/cyto-dl.nix { };
      in
      with pkgs;
      rec {
        apps.default =
          let
            python_with_pkgs = python3.withPackages (pp: [
              packages.nahual
              cytoDl
              pp.torch
              pp.torchvision
              pp.numpy
              pp.monai
              pp.lightning
              pp.hydra-core
              pp.omegaconf
              pp.torchmetrics
              pp.scikit-image
            ]);
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/megaseg.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        packages = {
          nahual = pkgs.python3.pkgs.callPackage ./nix/nahual.nix {
            pynng = inputs.pynng-flake.packages.${system}.pynng;
          };
          cyto-dl = cytoDl;
        };

        devShells = {
          default =
            let
              python_with_pkgs = python3.withPackages (pp: [
                packages.nahual
                cytoDl
                pp.torch
                pp.torchvision
                pp.numpy
                pp.monai
                pp.lightning
                pp.hydra-core
                pp.omegaconf
                pp.torchmetrics
                pp.scikit-image
                # Dev-only extras
                pp.tifffile
                pp.scikit-learn
                pp.pyyaml
                pp.requests
              ]);
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
              ];
              shellHook = ''
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
                export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
              '';
            };
        };
      }
    );
}
