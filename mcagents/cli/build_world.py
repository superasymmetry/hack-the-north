"""Generate a Minecraft world once and cache it, so later runs load it instead of building it.

    bash scripts/build_world.sh          # writes worlds/plains.zip, then exits

Run it once, or again after changing the biome -- the spawn biome is baked into the save, so
it is part of the cache filename rather than a per-run setting.
"""
import argparse

from mcagents.minecraft.world import DEFAULT_BIOME, WorldCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--biome", default=DEFAULT_BIOME, help="spawn biome to generate")
    WorldCache(biome=parser.parse_args().biome).build()


if __name__ == "__main__":
    main()
