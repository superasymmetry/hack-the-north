"""Check the offline world editor against a real world zip, without launching Minecraft.

Every mistake this layer can make is silent. A block index packed one bit wide writes a
wall of the wrong material; a chunk written back with a field dropped loads as empty air;
a `/fill` misread as inclusive-exclusive leaves the city one block short on two sides. None
of that raises -- you find out by looking at a screenshot twenty minutes later.

So the test does the two things that catch it: it round-trips real chunk NBT byte for byte,
and it bakes the real city into a real world zip and reads the blocks back out.

    python tests/test_bake_city.py
"""
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcagents.minecraft import nbt
from mcagents.minecraft.anvil import (AIR_BLOCK, Block, Region, World, bits_for, pack_states,
                                      unpack_states)
from mcagents.minecraft.world import WORLDS_DIR, baked_city_path, terrain_biome

SOURCE = WORLDS_DIR / "plains.zip"
failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'ok  ' if condition else 'FAIL'} {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


# ---------------------------------------------------------------- pure pieces

def test_bit_packing() -> None:
    """Indices must survive a round trip at every width the palette can reach."""
    rng = random.Random(0)
    for bits in range(4, 13):
        values = [rng.randrange(1 << bits) for _ in range(4096)]
        check(f"pack/unpack at {bits} bits", unpack_states(pack_states(values, bits), bits) == values)
    check("palette of 16 still fits 4 bits", bits_for(16) == 4)
    check("palette of 17 needs 5 bits", bits_for(17) == 5)
    # 1.16 does not straddle longs, so 5-bit indices waste the top 4 bits of every long.
    check("5-bit indices use 12 per long", len(pack_states([0] * 4096, 5)) == 342)


def test_block_parsing() -> None:
    door = Block.parse("minecraft:oak_door[facing=west,half=lower,hinge=left,open=false]")
    check("block name parsed", door.name == "minecraft:oak_door")
    check("properties parsed", dict(door.properties)["facing"] == "west")
    check("property order does not matter",
          Block.parse("a[y=2,x=1]") == Block.parse("a[x=1,y=2]"))
    check("bare name has no properties", Block.parse("minecraft:stone").properties == ())
    check("round trip through nbt",
          Block.from_nbt(door.to_nbt()) == door)


def test_naming() -> None:
    check("city.zip is plains underneath", terrain_biome("city") == "plains")
    check("desert_city.zip is desert underneath", terrain_biome("desert_city") == "desert")
    check("plains.zip is untouched", terrain_biome("plains") == "plains")
    check("baked path for the default", baked_city_path().name == "city.zip")
    check("baked path for another biome", baked_city_path("desert").name == "desert_city.zip")


# ---------------------------------------------------------------- against the real world

def test_nbt_round_trip(save_dir: Path) -> None:
    """Every chunk must re-serialise byte for byte, or the fields this repo does not model
    are being quietly dropped from worlds it writes."""
    total = mismatched = 0
    for path in sorted((save_dir / "region").glob("*.mca")):
        region = Region.load(path)
        for raw in region._raw.values():
            body = nbt.decompress(raw[1:])
            total += 1
            mismatched += nbt.dumps(nbt.loads(body)) != body
    check("chunk NBT round-trips byte for byte", mismatched == 0 and total > 0,
          f"{total - mismatched}/{total} chunks")


def test_bake(save_dir: Path, tmp: Path) -> None:
    """Bake the city and read it back: the streets, a building, and the spawn."""
    from mcagents.minecraft.bake import CityBake

    out = tmp / "city.zip"
    origin = CityBake(source=SOURCE, destination=out, verbose=False)
    # Reach past run() so the test can inspect the save directory before it is zipped.
    workspace = Path(tempfile.mkdtemp(prefix="test-bake-"))
    try:
        baked_dir = origin.unpack(workspace)
        x, ground, z = origin.build(baked_dir)
        world = World(baked_dir)

        check("no chunk under the plot is missing", not world.missing_chunks,
              f"{len(world.missing_chunks)} missing")
        check("street level is solid under the crossroads",
              world.get_block(x, ground - 1, z).name == "minecraft:stone")
        check("the crossroads is paved",
              "concrete" in world.get_block(x, ground, z).name)
        check("the plot was cleared above the street",
              world.get_block(x, ground + 3, z) == AIR_BLOCK)

        # Fourteen buildings, each with a two-block door; the park and plaza have none.
        doors = [(bx, bz) for bx in range(x - 50, x + 52) for bz in range(z - 50, z + 52)
                 if "door" in world.get_block(bx, ground + 1, bz).name]
        check("every building has a door", len(doors) == 14, f"{len(doors)} found")
        check("doors are two blocks tall",
              all(world.get_block(bx, ground + 2, bz).name
                  == world.get_block(bx, ground + 1, bz).name for bx, bz in doors))

        materials = {world.get_block(bx, ground, bz).name
                     for bx in range(x - 50, x + 52) for bz in range(z - 50, z + 52)}
        check("roads and lane markings are down",
              {"minecraft:gray_concrete", "minecraft:white_concrete"} <= materials)
        check("sidewalks are down", "minecraft:smooth_stone" in materials)

        level = nbt.read_compressed(baked_dir / "level.dat")["Data"]
        check("the clock is frozen at noon",
              int(level["DayTime"]) == 6000 and level["GameRules"]["doDaylightCycle"] == "false")
        check("the weather is frozen clear",
              int(level["raining"]) == 0 and level["GameRules"]["doWeatherCycle"] == "false")
        check("world spawn is the crossroads",
              (int(level["SpawnX"]), int(level["SpawnZ"])) == (x, z))

        player = nbt.read_compressed(next((baked_dir / "playerdata").glob("*.dat")))
        check("the agent comes up on the crossroads",
              [float(v) for v in player["Pos"]] == [x + 0.5, ground + 1, z + 0.5])
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_packed_zip(tmp: Path) -> None:
    """`loadWorldFromZip` takes the world name from `entries[0].split("/")[2]`, so the
    entry layout is load-bearing and not merely tidy."""
    from mcagents.minecraft.bake import CityBake

    out = CityBake(source=SOURCE, destination=tmp / "packed.zip", verbose=False).run()
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
    check("zip entries start with ./saves/<world>", names[0].split("/")[0:2] == [".", "saves"])
    check("the world name is recoverable", len({n.split("/")[2] for n in names}) == 1,
          names[0])
    check("region files came along", any("/region/" in name for name in names))


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} is missing. Build it once: bash scripts/build_world.sh")

    test_bit_packing()
    test_block_parsing()
    test_naming()

    tmp = Path(tempfile.mkdtemp(prefix="test-city-"))
    try:
        with zipfile.ZipFile(SOURCE) as archive:
            archive.extractall(tmp / "source")
        save_dir = next((tmp / "source" / "saves").iterdir())
        test_nbt_round_trip(save_dir)
        test_bake(save_dir, tmp)
        test_packed_zip(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        raise SystemExit(f"{len(failures)} failed: {', '.join(failures)}")
    print("all good")


if __name__ == "__main__":
    main()
