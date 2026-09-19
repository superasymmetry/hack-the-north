"""Bake the city into a world zip, once, so no run ever pays for it again.

    python -m mcagents.cli.bake_city                       # plains.zip -> city.zip
    python -m mcagents.cli.bake_city --source worlds/seed0.zip --out worlds/seed0_city.zip

The result is an ordinary cached world: `MC_WORLD=worlds/city.zip` with the city callback
*off*. See `mcagents.minecraft.bake`.
"""
import argparse
from pathlib import Path

from mcagents.minecraft.bake import CityBake
from mcagents.minecraft.city import CityLayout
from mcagents.minecraft.world import WORLDS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=WORLDS_DIR / "plains.zip",
                        help="the terrain to build on (default worlds/plains.zip)")
    parser.add_argument("--out", type=Path, default=WORLDS_DIR / "city.zip",
                        help="where to write the baked world (default worlds/city.zip)")
    parser.add_argument("--seed", type=int, default=7, help="city layout seed")
    parser.add_argument("--lots", type=int, default=CityLayout.lots, help="lots per side")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"{args.source} does not exist. Build it first: "
                         f"bash scripts/build_world.sh")

    CityBake(source=args.source, destination=args.out,
             layout=CityLayout(lots=args.lots), seed=args.seed).run()


if __name__ == "__main__":
    main()
