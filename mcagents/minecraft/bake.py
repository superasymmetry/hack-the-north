"""Bake the city into a world zip, so it costs nothing at reset.

`mcagents.minecraft.city` builds its city by firing ~430 chat commands into a *running*
world, because that is the only channel the environment offers, and it re-fires them after
every reset because nothing in the stack can write a modified world back to disk. That is
~20s of every episode, forever, for a city that is identical every time.

This module takes the other route into the same world: not through the game, but through the
save files, while nothing has them open. It unpacks `worlds/plains.zip`, runs the *same*
command list against the region data with `mcagents.minecraft.anvil`, and repacks the result
as `worlds/city.zip`. After that the city is terrain -- `MC_WORLD=worlds/city.zip` with no
`--city` loads it in the 9s a cached world already costs.

    bash scripts/bake_city.sh                     # plains.zip -> city.zip
    MC_WORLD=worlds/city.zip bash scripts/rocket2.sh --no-city

Reading `CityBuilder.commands()` rather than reimplementing the layout is the point. The
command list stays the single definition of what the city is, so the baked world and the
live callback cannot drift apart, and a change to a building style shows up in both.

Two things the game would have done that this has to do itself:

- **Where the city goes.** The callback centres on wherever the player came up, which it
  learns at runtime. Offline there is no player, so the origin comes from the position
  saved in the world's own `playerdata` -- the spot the agent will spawn on when this world
  is loaded -- and the street level from the highest solid block under it.
- **What a command means outside a world.** `/fill` and `/setblock` are block writes.
  `/gamerule`, `/time` and `/weather` are `level.dat` fields. `/forceload` exists only to
  work around chunks not being in memory, which is not a problem a file has.
"""
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mcagents.minecraft import nbt
from mcagents.minecraft.anvil import AIR_BLOCK, Block, World
from mcagents.minecraft.city import CityBuilder, CityLayout

#: `/time set <these>`, in ticks. The city asks for noon and freezes the clock there.
TIME_OF_DAY = {"day": 1000, "noon": 6000, "night": 13000, "midnight": 18000}

#: Blocks that a solid-ground search must not stop on. Leaves and logs are why: the spawn
#: column can have a tree in it, and taking street level from the canopy would bury the city.
NON_GROUND = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air", "minecraft:water",
              "minecraft:lava", "minecraft:grass", "minecraft:tall_grass", "minecraft:snow",
              "minecraft:oak_leaves", "minecraft:oak_log", "minecraft:birch_leaves",
              "minecraft:birch_log", "minecraft:spruce_leaves", "minecraft:spruce_log",
              "minecraft:poppy", "minecraft:dandelion", "minecraft:fern", "minecraft:seagrass"}


# ---------------------------------------------------------------- applying commands

@dataclass
class BakeStats:
    blocks: int = 0
    outside: int = 0
    commands: int = 0
    ignored: List[str] = field(default_factory=list)


class CommandBaker:
    """Applies the subset of commands `CityBuilder` emits to a save on disk."""

    def __init__(self, world: World, level: Dict, player: Optional[Dict] = None):
        self.world = world
        self.level = level          # the "Data" compound of level.dat
        self.player = player        # the playerdata compound, if there is one
        self.stats = BakeStats()

    def run(self, commands: List[str]) -> BakeStats:
        for command in commands:
            self.apply(command)
            self.stats.commands += 1
        return self.stats

    def apply(self, command: str) -> None:
        parts = command.lstrip("/").split()
        handler = getattr(self, f"_do_{parts[0]}", None)
        if handler is None:
            self.stats.ignored.append(parts[0])
            return
        handler(parts[1:])

    def unhandled(self) -> List[str]:
        """Command names this baker had no handler for, deduplicated.

        The city is whatever `CityBuilder.commands()` says it is, and a command this module
        does not know is a piece of the city missing from the baked world -- quietly, and
        only in the baked one, so the callback would still look right. The caller shouts.
        """
        return sorted(set(self.stats.ignored))

    # -- blocks ----------------------------------------------------------

    def _do_fill(self, args: List[str]) -> None:
        """`fill x1 y1 z1 x2 y2 z2 <block> [hollow]`.

        `hollow` is the only mode the city uses: the six faces in the named block, the
        inside emptied. Anything else would be a silent wrong shape, so it raises.
        """
        (x1, y1, z1, x2, y2, z2), block = self._coords(args[:6]), Block.parse(args[6])
        mode = args[7] if len(args) > 7 else None
        if mode not in (None, "hollow"):
            raise ValueError(f"/fill mode {mode!r} is not implemented")

        (x1, x2), (y1, y2), (z1, z2) = sorted((x1, x2)), sorted((y1, y2)), sorted((z1, z2))
        for y in range(y1, y2 + 1):
            for z in range(z1, z2 + 1):
                edge_zy = y in (y1, y2) or z in (z1, z2)
                for x in range(x1, x2 + 1):
                    if mode == "hollow" and not (edge_zy or x in (x1, x2)):
                        self._set(x, y, z, AIR_BLOCK)
                    else:
                        self._set(x, y, z, block)

    def _do_setblock(self, args: List[str]) -> None:
        x, y, z = self._coords(args[:3])
        self._set(x, y, z, Block.parse(args[3]))

    def _set(self, x: int, y: int, z: int, block: Block) -> None:
        if self.world.set_block(x, y, z, block):
            self.stats.blocks += 1
        elif block != AIR_BLOCK:
            # Air outside a generated chunk is already air; anything else is a real loss.
            self.stats.outside += 1

    @staticmethod
    def _coords(args: List[str]) -> Tuple[int, ...]:
        """Absolute coordinates only -- `~` needs an executor, and the city emits none."""
        if any(arg.startswith(("~", "^")) for arg in args):
            raise ValueError(f"relative coordinates {args} have no meaning offline")
        return tuple(int(float(arg)) for arg in args)

    # -- level.dat -------------------------------------------------------

    def _do_gamerule(self, args: List[str]) -> None:
        self.level.setdefault("GameRules", {})[args[0]] = args[1]

    def _do_time(self, args: List[str]) -> None:
        """`time set <name|ticks>`. DayTime is the clock; Time is the world's age."""
        if args[0] != "set":
            raise ValueError(f"/time {args[0]} cannot be baked")
        when = args[1]
        self.level["DayTime"] = nbt.Long(int(when) if when.isdigit() else TIME_OF_DAY[when])

    def _do_weather(self, args: List[str]) -> None:
        clear = args[0] == "clear"
        self.level["raining"] = nbt.Byte(not clear)
        self.level["thundering"] = nbt.Byte(0)
        self.level["rainTime"] = nbt.Int(1_000_000 if clear else 0)
        self.level["thunderTime"] = nbt.Int(1_000_000)
        self.level["clearWeatherTime"] = nbt.Int(1_000_000 if clear else 0)

    def _do_setworldspawn(self, args: List[str]) -> None:
        x, y, z = self._coords(args[:3])
        self.level["SpawnX"], self.level["SpawnY"], self.level["SpawnZ"] = (
            nbt.Int(x), nbt.Int(y), nbt.Int(z))

    def _do_tp(self, args: List[str]) -> None:
        """`tp @p x y z [yaw pitch]` -- the saved player position, which is where the agent
        comes up when this world is loaded."""
        values = [float(arg) for arg in args[1:]]
        for target in filter(None, [self.player, self.level.get("Player")]):
            target["Pos"] = nbt.TagList(nbt.DOUBLE, [nbt.Double(v) for v in values[:3]])
            if len(values) >= 5:
                target["Rotation"] = nbt.TagList(
                    nbt.FLOAT, [nbt.Float(values[3]), nbt.Float(values[4])])

    def _do_forceload(self, args: List[str]) -> None:
        """A running server's problem, not a file's."""


# ---------------------------------------------------------------- the save directory

def player_files(save_dir: Path) -> List[Path]:
    return sorted((save_dir / "playerdata").glob("*.dat"))


def spawn_position(save_dir: Path, level: Dict) -> Tuple[float, float, float]:
    """Where the agent will come up: its saved player position, or the world spawn.

    The player position wins because that is what the engine actually restores, and it is
    what `CityCallback` reads at runtime -- so a baked city lands in the same place a live
    one would.
    """
    for path in player_files(save_dir):
        pos = nbt.read_compressed(path).get("Pos")
        if pos:
            return tuple(float(v) for v in pos)
    if "Player" in level and "Pos" in level["Player"]:
        return tuple(float(v) for v in level["Player"]["Pos"])
    return float(level["SpawnX"]), float(level["SpawnY"]), float(level["SpawnZ"])


def ground_level(world: World, x: int, z: int, start_y: int) -> int:
    """The highest solid block at or below `start_y` in this column.

    `CityCallback` gets this for free -- the player is standing on it. Offline it has to be
    looked up, and the honest answer is a downward scan that skips the things you can stand
    inside: grass, flowers, a tree the spawn point happens to sit under.
    """
    for y in range(min(start_y, 255), 0, -1):
        if world.get_block(x, y, z).name not in NON_GROUND:
            return y
    return start_y - 1


def prune_contents(world: World, box: Tuple[int, int, int, int, int, int]) -> int:
    """Drop entities and block entities standing inside the cleared plot.

    The plot is levelled and emptied before anything is built on it, but that only moves
    *blocks*. A cow that generated where a building now is would spawn inside its wall, and
    a chest's block entity would outlive the chest and confuse the client.
    """
    x1, y1, z1, x2, y2, z2 = box
    removed = 0
    for region in world.regions.values():
        if region is None:
            continue
        for chunk in region.chunks.values():
            level = chunk.level
            for key, position in (("Entities", None), ("TileEntities", ("x", "y", "z"))):
                items = level.get(key)
                if not items:
                    continue
                kept = []
                for item in items:
                    if position:
                        x, y, z = (int(item[axis]) for axis in position)
                    else:
                        x, y, z = (float(v) for v in item["Pos"])
                    if x1 <= x <= x2 and y1 <= y <= y2 and z1 <= z <= z2:
                        removed += 1
                    else:
                        kept.append(item)
                if len(kept) != len(items):
                    level[key] = nbt.TagList(items.element_type, kept)
                    chunk.dirty = True
    return removed


# ---------------------------------------------------------------- top level

@dataclass
class CityBake:
    """Unpack a world zip, build the city into it, and pack it back out."""

    source: Path
    destination: Path
    layout: CityLayout = field(default_factory=CityLayout)
    seed: int = 7
    verbose: bool = True

    def run(self) -> Path:
        from mcagents.minecraft.world import WorldCache

        workspace = Path(tempfile.mkdtemp(prefix="bake-city-"))
        try:
            save_dir = self.unpack(workspace)
            origin = self.build(save_dir)
            out = WorldCache.from_path(self.destination).pack(save_dir)
            self._log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB); "
                      f"the city is at {origin}, and costs nothing to load")
            return out
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def unpack(self, workspace: Path) -> Path:
        """Extract the source zip and return the one save directory inside it."""
        import zipfile

        with zipfile.ZipFile(self.source) as archive:
            archive.extractall(workspace)
        saves = [path for path in (workspace / "saves").iterdir() if path.is_dir()]
        if len(saves) != 1:
            raise SystemExit(f"expected one world in {self.source}, found {len(saves)}")
        return saves[0]

    def build(self, save_dir: Path) -> Tuple[int, int, int]:
        """Run the city's command list into the save. Returns (x, ground y, z)."""
        level_path = save_dir / "level.dat"
        root = nbt.read_compressed(level_path)
        level = root["Data"]
        world = World(save_dir)

        px, py, pz = spawn_position(save_dir, level)
        # int(round(...)) and floor(y) - 1: the same arithmetic CityCallback does on
        # `info["player_pos"]`, so both routes centre the city on the same block.
        x, z = int(round(px)), int(round(pz))
        ground = ground_level(world, x, z, int(math.floor(py)) - 1)

        builder = CityBuilder(x, z, ground, layout=self.layout, seed=self.seed)
        commands = builder.commands()
        self._log(f"baking a {self.layout.span}x{self.layout.span} city at ({x}, {z}), "
                  f"street level y={ground} -- {len(commands)} commands")

        players = [nbt.read_compressed(path) for path in player_files(save_dir)]
        baker = CommandBaker(world, level, players[0] if players else None)
        stats = baker.run(commands)

        x0, z0, x1, z1 = builder.bounds
        pruned = prune_contents(
            world, (x0, ground, z0, x1, ground + self.layout.clear, z1))
        self._log(f"{stats.blocks} blocks set, {pruned} entities removed"
                  + (f", {stats.outside} outside the generated chunks" if stats.outside else ""))
        if baker.unhandled():
            self._log(f"WARNING: no handler for {', '.join(baker.unhandled())} -- that much "
                      f"of the city is in the callback but not in this world")
        if world.missing_chunks:
            self._log(f"WARNING: {len(world.missing_chunks)} chunks under the plot were never "
                      f"generated and will regenerate without the city on top of them")

        world.save()
        nbt.write_compressed(level_path, root)
        for path, player in zip(player_files(save_dir), players):
            nbt.write_compressed(path, player)
        return x, ground, z

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[bake] {message}", flush=True)
