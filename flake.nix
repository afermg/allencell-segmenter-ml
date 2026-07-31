{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        cytoDl = pkgs.python3.pkgs.callPackage ./nix/cyto-dl.nix {};
        python_with_pkgs = pkgs.python3.withPackages (pp: [
          inputs.nahual-flake.packages.${system}.nahual
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
        runServer = pkgs.writeScriptBin "nahual-megaseg" ''
          #!${pkgs.bash}/bin/bash
          : "''${MEGASEG_CACHE_DIR:=''${XDG_CACHE_HOME:-$HOME/.cache}/megaseg}"
          export MEGASEG_CACHE_DIR
          mkdir -p "$MEGASEG_CACHE_DIR"
          exec ${python_with_pkgs}/bin/python ${self}/server.py \
            "''${1:-tcp://0.0.0.0:5555}"
        '';
        megasegApp = {
          type = "app";
          program = "${runServer}/bin/nahual-megaseg";
        };
      in
        with pkgs; rec {
          packages =
            {cyto-dl = cytoDl;}
            // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              oci-image = import ./nix/oci-image.nix {
                inherit pkgs;
                name = "megaseg";
                title = "Nahual Allen Cell MegaSeg";
                description = "Allen Cell MegaSeg 3-D segmentation served through Nahual";
                source = "https://github.com/afermg/allencell-segmenter-ml";
                revision = self.rev or self.dirtyRev or "unknown";
                server = runServer;
                entrypoint = megasegApp.program;
              };
            };
          inherit python_with_pkgs;
          scripts.runServer = runServer;
          apps = rec {
            megaseg = megasegApp;
            default = megaseg;
          };
          devShells.default = mkShell {
            packages = [
              python_with_pkgs
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              python3Packages.tifffile
              python3Packages.scikit-learn
              python3Packages.pyyaml
              python3Packages.requests
            ];
            shellHook = ''
              export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
              : "''${MEGASEG_CACHE_DIR:=''${XDG_CACHE_HOME:-$HOME/.cache}/megaseg}"
              export MEGASEG_CACHE_DIR
            '';
          };
        }
    );
}
