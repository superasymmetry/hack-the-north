#!/usr/bin/env bash
# Everything that has to be layered on top of a fresh `pip install minestudio`.
# Run once per fresh conda env (new machine, SLURM node, minestudio upgrade). Safe to re-run.
# Assumes the env exists and is active -- see docs/setup.md for creating it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install psutil "cuda-python==12.9.7" "pyglet==2.1.16"

# Source patches for real bugs in minestudio 1.1.6 (GPU rendering, the play GUI).
python "$HERE/apply_patches.py"

# ROCKET-2's policy is not part of the minestudio package -- fetch the two files we import
# from the CraftJarvis repo and patch them for this timm version. See docs/rocket2.md.
python "$HERE/vendor_rocket2.py"
