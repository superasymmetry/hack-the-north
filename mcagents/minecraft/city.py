"""Build a grid-plan city into the world at reset, so there is somewhere to go.

The default world is a plains biome: grass, a few trees, nothing to navigate *around*. That
is fine for "mine a log" and useless for anything that looks like getting somewhere. This
module lays a 102x102 city over the spawn point -- streets with lane markings, fourteen
hollow buildings you can walk into through real doors, a park and a fountain plaza -- as a
callback that runs on every reset:

    sim = MinecraftSim(..., callbacks=[CityCallback(), PrevActionCallback()])

There is no block-placement API. The one channel into a running world is
`sim.env.execute_cmd`, which stuffs a string into the chat action of a single env step, so
the city is ~430 `/fill` and `/setblock` commands at one game tick each -- about 20 seconds
on top of the ~9s reset. Three things constrain how those commands are written:

- `/fill` refuses regions over 32768 blocks, so `fill()` halves a box until each piece fits.
- `/fill` does nothing to chunks the server has not loaded, and the plot is 7x7 chunks while
  the player stands in the middle of it. `/forceload add` over the whole plot comes first.
- `/fill ... outline` is a trap; see `fill()`. Use `ring()`.

**Prefer the baked world to this callback.** Nothing here can write the modified world back
to disk -- `EnvServer` has `loadWorldFromZip` and nothing that writes one, `/save-all` is not
a registered command, and ending the mission does not flush either (building a marker,
resetting, and diffing the save directory rewrote 0 of 4 region files). But the *files* can
be edited while no Minecraft has them open, which is what `mcagents.minecraft.bake` does: it
runs this same command list into a world zip's region data offline, so the city arrives as
terrain and a reset costs 5s instead of 15s.

    bash scripts/bake_city.sh        # once: worlds/plains.zip -> worlds/city.zip

`Session` picks that up on its own when it exists. What is left for this callback is the
case where it does not -- a fresh checkout, or a layout being iterated on, where rebuilding
every reset beats re-baking every edit.

The city is deterministic given its seed -- same buildings, same heights, same materials --
so runs are comparable. `python -m mcagents.cli.city` dry-runs the command list without
launching Minecraft, which is how to iterate on the layout.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from minestudio.simulator.callbacks.callback import MinecraftCallback

Commands = List[str]

#: `/fill` refuses anything larger than this many blocks.
FILL_LIMIT = 32768

ROAD_BLOCK = "minecraft:gray_concrete"
LANE_BLOCK = "minecraft:white_concrete"
WALK_BLOCK = "minecraft:smooth_stone"
LAWN_BLOCK = "minecraft:grass_block"

#: One entry per building, drawn without replacement until the list runs out and reshuffles.
#: The mismatch between a wall, its roof and its trim is deliberate -- uniform buildings read
#: as a texture rather than as a place with landmarks in it.
STYLES: List[Dict[str, str]] = [
    {"wall": "minecraft:bricks",              "roof": "minecraft:dark_prismarine", "trim": "minecraft:brick_wall",       "glass": "minecraft:glass_pane",               "door": "minecraft:oak_door"},
    {"wall": "minecraft:stone_bricks",        "roof": "minecraft:cobblestone",     "trim": "minecraft:stone_brick_wall", "glass": "minecraft:glass",                    "door": "minecraft:spruce_door"},
    {"wall": "minecraft:smooth_sandstone",    "roof": "minecraft:cut_sandstone",   "trim": "minecraft:sandstone_wall",   "glass": "minecraft:glass_pane",               "door": "minecraft:birch_door"},
    {"wall": "minecraft:white_concrete",      "roof": "minecraft:gray_concrete",   "trim": "minecraft:stone_brick_wall", "glass": "minecraft:light_blue_stained_glass", "door": "minecraft:dark_oak_door"},
    {"wall": "minecraft:light_gray_concrete", "roof": "minecraft:black_concrete",  "trim": "minecraft:cobblestone_wall", "glass": "minecraft:gray_stained_glass",       "door": "minecraft:oak_door"},
    {"wall": "minecraft:oak_planks",          "roof": "minecraft:dark_oak_planks", "trim": "minecraft:oak_fence",        "glass": "minecraft:glass_pane",               "door": "minecraft:oak_door"},
    {"wall": "minecraft:terracotta",          "roof": "minecraft:red_terracotta",  "trim": "minecraft:brick_wall",       "glass": "minecraft:glass",                    "door": "minecraft:acacia_door"},
    {"wall": "minecraft:quartz_block",        "roof": "minecraft:smooth_quartz",   "trim": "minecraft:stone_brick_wall", "glass": "minecraft:light_blue_stained_glass", "door": "minecraft:jungle_door"},
]


# ---------------------------------------------------------------- command primitives

def fill(x1, y1, z1, x2, y2, z2, block, mode=None) -> Commands:
    """One or more `/fill` commands covering the box, split to stay under the 32768 limit.

    The split halves the longest axis recursively, which keeps the pieces cube-ish and the
    count near minimal. `mode` is passed through: None fills every block, "hollow" builds the
    six faces and empties the interior. `hollow` does not survive being split into pieces
    (each piece would grow its own faces), so that asserts rather than silently building
    something else.

    `outline` is deliberately unsupported. Minecraft calls a block an "edge" if it sits at
    the min or max of *any* axis, so on a region one block tall every block qualifies and the
    outline is a solid slab -- which silently turned every window band into a glass floor the
    first time this ran. Use `ring()` for a rectangle at one height.
    """
    (x1, x2), (y1, y2), (z1, z2) = sorted((x1, x2)), sorted((y1, y2)), sorted((z1, z2))
    volume = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
    suffix = f" {mode}" if mode else ""

    if volume <= FILL_LIMIT:
        return [f"/fill {x1} {y1} {z1} {x2} {y2} {z2} {block}{suffix}"]

    assert mode is None, f"a {mode} fill of {volume} blocks cannot be split without changing shape"
    _, axis = max([(x2 - x1, 0), (y2 - y1, 1), (z2 - z1, 2)])
    lo, hi = [(x1, x2), (y1, y2), (z1, z2)][axis]
    mid = (lo + hi) // 2
    low_corner, high_corner = [x1, y1, z1], [x2, y2, z2]
    high_corner[axis] = mid
    head = fill(*low_corner, *high_corner, block, mode)
    low_corner = [x1, y1, z1]
    low_corner[axis] = mid + 1
    return head + fill(*low_corner, x2, y2, z2, block, mode)


def ring(x1, y, z1, x2, z2, block) -> Commands:
    """A hollow rectangle at a single height, as four edge fills.

    Four commands is the honest price of a window band, a parapet or a fountain rim -- the
    one-command version would be `/fill ... outline`, which does not do what its name
    suggests on a flat region (see `fill`).
    """
    return (fill(x1, y, z1, x2, y, z1, block) + fill(x1, y, z2, x2, y, z2, block)
            + fill(x1, y, z1 + 1, x1, y, z2 - 1, block) + fill(x2, y, z1 + 1, x2, y, z2 - 1, block))


def setblock(x, y, z, block) -> Commands:
    return [f"/setblock {x} {y} {z} {block}"]


# ---------------------------------------------------------------- layout

@dataclass(frozen=True)
class CityLayout:
    """The grid. Everything else in this file is derived from these numbers.

    `lots` lots per side, each `lot` blocks square, separated and surrounded by `road`-wide
    streets, so the plot is lots*lot + (lots+1)*road blocks square, centred on spawn.
    """

    lots: int = 4
    lot: int = 18
    road: int = 6
    #: Everything from the surface up to `clear` is emptied (hills, trees, the odd overhang),
    #: and `depth` blocks of stone are packed underneath so the plot cannot be undercut by a
    #: cave that happened to generate there.
    clear: int = 45
    depth: int = 4
    #: Two lots are not buildings. Both sit diagonally off the central intersection where the
    #: player spawns, so the first frame has trees on one side and towers on the others.
    park: Tuple[int, int] = (1, 1)
    plaza: Tuple[int, int] = (0, 3)

    @property
    def span(self) -> int:
        return self.lots * self.lot + (self.lots + 1) * self.road

    @property
    def low(self) -> int:
        """The plot runs low .. low + span - 1, i.e. -50..51 by default."""
        return -(self.span // 2) + 1

    def lot_bounds(self, index: int) -> Tuple[int, int]:
        """The inclusive [start, end] block range of lot `index` along one axis.

        Lot i is preceded by i+1 roads and i lots, hence the offset.
        """
        start = self.low + self.road + index * (self.lot + self.road)
        return start, start + self.lot - 1

    def road_bands(self) -> List[Tuple[int, int]]:
        """The inclusive [start, end] range of each street along one axis.

        lots+1 of them: one before the first lot, one after the last, one between each pair.
        """
        step = self.lot + self.road
        return [(self.low + i * step, self.low + i * step + self.road - 1)
                for i in range(self.lots + 1)]


# ---------------------------------------------------------------- the builder

@dataclass
class CityBuilder:
    """Emits the command list for one city, centred on (x, z) with street level at y.

    Deterministic given `seed`: same buildings, same heights, same materials, so two runs are
    comparable. Coordinates are absolute rather than `~` relative because the player keeps
    falling and drifting between the ticks the commands run on.
    """

    origin_x: int
    origin_z: int
    ground_y: int
    layout: CityLayout = field(default_factory=CityLayout)
    seed: int = 7

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self._deck: List[Dict[str, str]] = []

    # -- plot coordinates ------------------------------------------------

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """(x0, z0, x1, z1) of the whole plot, in world coordinates."""
        span, low = self.layout.span, self.layout.low
        return (self.origin_x + low, self.origin_z + low,
                self.origin_x + low + span - 1, self.origin_z + low + span - 1)

    def lot_box(self, i: int, j: int, inset: int = 0) -> Tuple[int, int, int, int]:
        """(x0, z0, x1, z1) of lot (i, j) in world coordinates, optionally inset."""
        x1, x2 = self.layout.lot_bounds(i)
        z1, z2 = self.layout.lot_bounds(j)
        return (self.origin_x + x1 + inset, self.origin_z + z1 + inset,
                self.origin_x + x2 - inset, self.origin_z + z2 - inset)

    # -- the whole city --------------------------------------------------

    def commands(self) -> Commands:
        """The whole command list, in build order."""
        x0, z0, x1, z1 = self.bounds
        # Force-load first: /fill is a silent no-op on chunks the server has not got in
        # memory, and the plot is seven chunks across while the player holds only the middle.
        cmds = [f"/forceload add {x0} {z0} {x1} {z1}"]
        # Noon forever. A vision agent in a city at night sees nothing, and frozen weather
        # keeps rain off the camera.
        cmds += ["/gamerule doDaylightCycle false", "/time set noon",
                 "/gamerule doWeatherCycle false", "/weather clear"]
        cmds += self.ground()
        cmds += self.streets()
        cmds += self.lots()
        cmds += self.lamps()
        # Land the player in the middle of the central crossroads looking down a street, and
        # move world spawn there too so a death does not strand them outside the city.
        cmds += [f"/setworldspawn {self.origin_x} {self.ground_y + 1} {self.origin_z}",
                 f"/tp @p {self.origin_x + 0.5} {self.ground_y + 1} {self.origin_z + 0.5} 45 0"]
        # `/forceload remove all` throws on this server; name the same rectangle back instead,
        # so the run does not leave 49 chunks pinned for the rest of the episode.
        cmds += [f"/forceload remove {x0} {z0} {x1} {z1}"]
        return cmds

    def ground(self) -> Commands:
        """Flatten the plot: stone underneath, a lawn at street level, nothing above.

        Clearing upward is what makes the city legible -- a plains world still has trees and
        a few metres of relief, and a building half-buried in a hillside is worse than none.
        """
        x0, z0, x1, z1 = self.bounds
        g, layout = self.ground_y, self.layout
        return (fill(x0, g - layout.depth, z0, x1, g - 1, z1, "minecraft:stone")
                + fill(x0, g, z0, x1, g, z1, LAWN_BLOCK)
                + fill(x0, g + 1, z0, x1, g + layout.clear, z1, "minecraft:air"))

    def streets(self) -> Commands:
        """Pave the road bands, run a lane line down the middle of each, and pave every lot.

        The lane markings are the cheapest thing in the file and do more for "this is a city"
        than another two buildings would: they give the streets a direction to be walked along.
        """
        x0, z0, x1, z1 = self.bounds
        g, road = self.ground_y, self.layout.road
        cmds: Commands = []

        for start, end in self.layout.road_bands():
            cmds += fill(x0, g, self.origin_z + start, x1, g, self.origin_z + end, ROAD_BLOCK)
            cmds += fill(self.origin_x + start, g, z0, self.origin_x + end, g, z1, ROAD_BLOCK)
        for start, _ in self.layout.road_bands():
            mid_a, mid_b = start + road // 2 - 1, start + road // 2
            cmds += fill(x0, g, self.origin_z + mid_a, x1, g, self.origin_z + mid_b, LANE_BLOCK)
            cmds += fill(self.origin_x + mid_a, g, z0, self.origin_x + mid_b, g, z1, LANE_BLOCK)

        # Pave each lot end to end. Whatever goes on it -- a building, the park, the plaza --
        # covers the middle afterwards and leaves this showing around the edge, which is a
        # sidewalk for rather less arithmetic than drawing one.
        for i in range(self.layout.lots):
            for j in range(self.layout.lots):
                bx0, bz0, bx1, bz1 = self.lot_box(i, j)
                cmds += fill(bx0, g, bz0, bx1, g, bz1, WALK_BLOCK)
        return cmds

    def lots(self) -> Commands:
        """Fill every lot: the park, the plaza, and a building on each of the rest."""
        cmds: Commands = []
        for i in range(self.layout.lots):
            for j in range(self.layout.lots):
                # Inset by 2, so the paving streets() laid down survives as a kerb.
                box = self.lot_box(i, j, inset=2)
                if (i, j) == self.layout.park:
                    cmds += self.park(*box)
                elif (i, j) == self.layout.plaza:
                    cmds += self.plaza(*box)
                else:
                    cmds += self.building(i, j)
        return cmds

    def lamps(self) -> Commands:
        """Two lamps per lot, at the midpoints of one pair of edges, alternating which pair.

        Not the corners: a lot corner is four blocks diagonally from the spawn intersection,
        and a lamp post there fills a quarter of the agent's very first frame. Midpoints keep
        the crossroads clear and the sight lines open.
        """
        cmds: Commands = []
        for i in range(self.layout.lots):
            for j in range(self.layout.lots):
                x0, z0, x1, z1 = self.lot_box(i, j)
                cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
                if (i + j) % 2 == 0:
                    cmds += self.lamp(cx, z0) + self.lamp(cx, z1)
                else:
                    cmds += self.lamp(x0, cz) + self.lamp(x1, cz)
        return cmds

    # -- pieces ----------------------------------------------------------

    def building(self, i: int, j: int) -> Commands:
        """One hollow box you can walk into, jittered inside its lot.

        Downtown in the middle, low-rise at the edges -- the skyline is what tells you which
        way the centre is from anywhere in the city.
        """
        if not self._deck:
            self._deck = self.rng.sample(STYLES, len(STYLES))
        style = self._deck.pop()

        lots = self.layout.lots
        edge = max(abs(i - (lots - 1) / 2), abs(j - (lots - 1) / 2))
        height = self.rng.randint(14, 22) if edge < 1 else self.rng.randint(5, 11)

        # Inside the kerb, then jittered so the street wall is not a straight line.
        zone = self.layout.lot - 4
        side = self.rng.randint(zone - 4, zone)
        x1, z1 = self.layout.lot_bounds(i)[0], self.layout.lot_bounds(j)[0]
        bx = x1 + 2 + self.rng.randint(0, zone - side)
        bz = z1 + 2 + self.rng.randint(0, zone - side)
        return self._shell(self.origin_x + bx, self.origin_z + bz,
                           self.origin_x + bx + side - 1, self.origin_z + bz + side - 1,
                           height, style)

    def _shell(self, x1, z1, x2, z2, height, style) -> Commands:
        """The building itself: ribbon windows, a parapet, a lit floor and a door.

        Order matters twice. The window rings run before the door is cut, or one of them
        would paint over its upper half. And the floor lanterns go in after the shell,
        because they sit *in* the floor the shell's own bottom face just laid down.

        The shell starts at street level rather than one above it, so its solid bottom face
        replaces the ground instead of sitting on top of it and the doorway ends up flush
        with the pavement. A one-block lip is the difference between an agent walking in and
        an agent standing outside holding forward.
        """
        g = self.ground_y
        cmds = fill(x1, g, z1, x2, g + height, z2, style["wall"], mode="hollow")

        for y in range(g + 2, g + height - 1, 3):
            cmds += ring(x1, y, z1, x2, z2, style["glass"])

        # A roof in the trim material rather than the wall material, and a parapet standing
        # on it, so the skyline has edges instead of a row of flat-topped boxes.
        cmds += fill(x1, g + height, z1, x2, g + height, z2, style["roof"])
        cmds += ring(x1, g + height + 1, z1, x2, z2, style["trim"])

        # Sea lanterns set into the floor. Permanent noon keeps the streets safe, but the
        # inside of a fourteen-block box is pitch dark and would be spawning mobs in a minute.
        cx, cz = (x1 + x2) // 2, (z1 + z2) // 2
        cmds += fill(cx - 1, g, cz - 1, cx, g, cz, "minecraft:sea_lantern")

        # The door goes in the wall facing the city centre, so the fronts all look inward at
        # the spawn intersection. `facing` is the direction a player was looking when they
        # placed the door, i.e. the inward normal of the wall it sits in.
        if abs(cx - self.origin_x) > abs(cz - self.origin_z):
            dx, dz = (x2 if cx < self.origin_x else x1), cz
            facing = "west" if cx < self.origin_x else "east"
        else:
            dx, dz = cx, (z2 if cz < self.origin_z else z1)
            facing = "north" if cz < self.origin_z else "south"

        hinge = self.rng.choice(["left", "right"])
        door = style["door"]
        cmds += fill(dx, g + 1, dz, dx, g + 2, dz, "minecraft:air")
        cmds += setblock(dx, g + 1, dz, f"{door}[facing={facing},half=lower,hinge={hinge},open=false]")
        cmds += setblock(dx, g + 2, dz, f"{door}[facing={facing},half=upper,hinge={hinge},open=false]")
        return cmds

    def park(self, x1, z1, x2, z2) -> Commands:
        """Lawn, gravel paths, a pond and five oaks.

        The oaks are why the park sits next to spawn: a plan that points a targeter at "tree"
        breaks in a city with no tree in the first frame.
        """
        g = self.ground_y
        cx, cz = (x1 + x2) // 2, (z1 + z2) // 2
        cmds = fill(x1, g, z1, x2, g, z2, LAWN_BLOCK)
        cmds += fill(x1, g, cz, x2, g, cz + 1, "minecraft:gravel")
        cmds += fill(cx, g, z1, cx + 1, g, z2, "minecraft:gravel")
        cmds += fill(cx - 4, g - 2, cz - 5, cx - 1, g, cz - 3, "minecraft:water")
        for x, z in [(x1 + 3, z2 - 3), (x2 - 3, z2 - 3), (x2 - 3, z1 + 3),
                     (cx + 4, cz + 4), (x1 + 3, z1 + 6)]:
            cmds += self.tree(x, z, self.rng.randint(4, 6))
        for index, (x, z) in enumerate([(cx - 3, cz + 3), (cx + 4, cz - 4)]):
            facing = "north" if index else "south"
            cmds += fill(x, g + 1, z, x + 2, g + 1, z, f"minecraft:oak_stairs[facing={facing}]")
        return cmds

    def plaza(self, x1, z1, x2, z2) -> Commands:
        """Paved square with a quartz fountain. Pure landmark: something to say "go to" about
        that is neither a building nor a tree, so a plan can name it without ambiguity."""
        g = self.ground_y
        cx, cz = (x1 + x2) // 2, (z1 + z2) // 2
        return (fill(x1, g, z1, x2, g, z2, "minecraft:polished_andesite")
                + fill(cx - 3, g, cz - 3, cx + 3, g, cz + 3, "minecraft:quartz_block")
                + ring(cx - 3, g + 1, cz - 3, cx + 3, cz + 3, "minecraft:quartz_slab")
                + fill(cx - 2, g + 1, cz - 2, cx + 2, g + 1, cz + 2, "minecraft:water")
                + fill(cx, g + 1, cz, cx, g + 3, cz, "minecraft:quartz_pillar")
                + setblock(cx, g + 4, cz, "minecraft:sea_lantern"))

    def tree(self, x, z, height) -> Commands:
        """A hand-built oak: a sapling would need days of game time to grow.

        `persistent=true` on the leaves, or they decay within minutes of the world loading
        and the park turns into a field of bare posts.
        """
        leaf = "minecraft:oak_leaves[persistent=true]"
        y = self.ground_y
        top = y + height
        return (fill(x - 2, top - 2, z - 2, x + 2, top - 1, z + 2, leaf)
                + fill(x - 1, top, z - 1, x + 1, top + 1, z + 1, leaf)
                + fill(x, y + 1, z, x, top, z, "minecraft:oak_log"))

    def lamp(self, x, z) -> Commands:
        """Cobblestone post with a sea lantern on it."""
        g = self.ground_y
        return (fill(x, g + 1, z, x, g + 4, z, "minecraft:cobblestone_wall")
                + setblock(x, g + 5, z, "minecraft:sea_lantern"))


# ---------------------------------------------------------------- the callback

class CityCallback(MinecraftCallback):
    """Builds the city after every reset, around wherever the player came up.

    Must be listed *before* PrevActionCallback: `MinecraftSim.reset` runs callbacks in order
    and this one steps the env several hundred times, which would leave a prev-action seeded
    by an earlier callback stale.
    """

    def __init__(self, layout: CityLayout = CityLayout(), seed: int = 7, verbose: bool = True):
        super().__init__()
        self.layout = layout
        self.seed = seed
        self.verbose = verbose
        self.origin = None       # (x, z, street level y) of the city just built

    def after_reset(self, sim, obs, info):
        pos = info["player_pos"]
        x, z = int(round(pos["x"])), int(round(pos["z"]))
        # The block the player is standing *on*: their feet are at floor(y), ground is below.
        y = int(math.floor(pos["y"])) - 1
        self.origin = (x, z, y)

        builder = CityBuilder(x, z, y, layout=self.layout, seed=self.seed)
        cmds = builder.commands()
        span = self.layout.span
        self._log(f"building a {span}x{span} city at ({x}, {z}), street level y={y} "
                  f"-- {len(cmds)} commands, about {len(cmds) / 20:.0f}s")

        for number, cmd in enumerate(cmds, 1):
            new_obs, _, _, info = sim.env.execute_cmd(cmd)
            obs.update(new_obs)
            if number % 100 == 0:
                self._log(f"  {number}/{len(cmds)}")

        # Let the client render the finished city before the agent sees its first frame.
        noop = sim.env.action_space.no_op()
        for _ in range(20):
            new_obs, _, _, info = sim.env.step(noop)
            obs.update(new_obs)

        self._log(f"done -- spawned at the central crossroads ({x}, {y + 1}, {z})")
        return sim._wrap_obs_info(obs, info)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[city] {message}", flush=True)

    def __repr__(self) -> str:
        return f"CityCallback(seed={self.seed}, span={self.layout.span})"
