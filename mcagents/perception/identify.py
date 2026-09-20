"""What a pointed pixel is, and therefore what ROCKET-2 should do about it.

A pinch gives a pixel. ROCKET-2 wants a pixel *and* an interaction (Hunt, Mine, Approach --
see mcagents/agents/rocket2.py), and nothing in a pinch says which. This closes that gap
without a model: `info['voxels']` already carries the block grid around the player, so
casting the pointed pixel through it (mcagents/minecraft/ranging.py) answers "what block is
under that pixel" exactly, for free, from the observation the sim was already producing.

    choice = identify.choose(sim, agent.info, point, frame.shape)
    agent.run(point=point, interaction=choice.interaction, stop=choice.stop)

The one thing the grid cannot see is mobs: it is a *block* grid, so a ray aimed at a cow
goes straight through it and lands on the grass behind. That miss is the signal. Pinching
something that casts to terrain -- ground, path, plants -- means the interesting thing was
standing in front of the terrain, because nobody pinches a grass block on purpose. So:

    hit a built or natural structure   ->  it is that block          ->  Mine / Switch / Use
    hit bare terrain                   ->  something was in the way  ->  Hunt
    hit nothing inside the grid        ->  further than ~7 blocks    ->  Approach

The middle rule is the one that trades correctness for having no second model. It is right
in a world whose only mobs are the cows click_rocket2 summons, and wrong the moment you want
to pinch the ground to walk there. When that day comes the fix is a real classifier over the
pinched region -- OWLv2 (mcagents/perception/owlv2.py) run over a fixed label vocabulary,
picking the box that contains the point -- dropped in behind the same `choose()` signature.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from mcagents.minecraft import ranging

#: Blocks that mean "this is just the world", i.e. the ray found no target and whatever was
#: pinched is standing on them. Ground, the plants growing out of it, and the surfaces a city
#: paves with -- CityCallback lays streets, and a cow on a street casts to the street.
TERRAIN = frozenset({
    "grass_block", "dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium", "mud",
    "dirt_path", "grass_path", "farmland", "sand", "red_sand", "gravel", "clay",
    "snow", "snow_block", "powder_snow", "ice", "packed_ice", "blue_ice", "moss_block",
    "grass", "short_grass", "tall_grass", "fern", "large_fern", "dead_bush", "seagrass",
    "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "oxeye_daisy",
    "cornflower", "lily_of_the_valley", "sunflower", "lilac", "rose_bush", "peony",
    "red_tulip", "orange_tulip", "white_tulip", "pink_tulip", "sweet_berry_bush",
})

#: Suffixes of blocks you operate rather than break. Suffix matching rather than a list
#: because Minecraft spells the same fitting once per wood type, and a version you have not
#: seen adds more: "warped_door" should behave like "oak_door" without being enumerated.
SWITCHES = ("_door", "_trapdoor", "_fence_gate", "_button", "lever", "_pressure_plate")

#: Blocks whose point is the screen they open.
CONTAINERS = frozenset({
    "chest", "trapped_chest", "ender_chest", "barrel", "shulker_box", "hopper",
    "furnace", "blast_furnace", "smoker", "brewing_stand", "enchanting_table", "anvil",
    "dispenser", "dropper", "beacon", "lectern", "smithing_table", "grindstone",
})

#: The one container ROCKET-2 has a dedicated interaction for.
CRAFTING = frozenset({"crafting_table", "cartography_table", "stonecutter", "loom"})

#: Steps a goal gets when nothing better can end it. Generous enough to walk somewhere and
#: swing a few times, short enough that a wrong guess costs seconds rather than a minute.
BUDGET = 200

#: How close an Approach has to get. Wider than the agent's own `arrive_distance` default
#: because an Approach chosen by this module is the "I do not know what that is" answer, and
#: stopping short of an unknown thing beats walking into it.
ARRIVE = 8


@dataclass
class Choice:
    """An interaction and a stop for it, plus what the ray saw to justify them."""
    interaction: str
    stop: Dict[str, Any] = field(default_factory=dict)
    block: Optional[str] = None       # what the ray landed on; None if it landed on nothing
    reason: str = ""

    def __str__(self) -> str:
        return f"{self.interaction} ({self.reason})"


def block_at(sim, info: Dict[str, Any], point: Sequence[float], shape: Sequence[int],
             fov: float = 70.0) -> Optional[str]:
    """The block the pointed pixel lands on, or None if the ray leaves the grid first.

    None covers three cases that behave the same way -- further than the grid reaches, a sim
    built without VoxelsCallback, and a sim not reporting a position -- and all three mean
    the same thing here: nothing is known about the target, so walk toward it.
    """
    landed = ranging.hit(ranging.voxels(sim, info),
                         ranging.view_ray(info, point, shape, fov))
    return None if landed is None else landed[1]


def interaction_for(block: Optional[str]) -> Tuple[str, str]:
    """(interaction, why) for a block name. See the module docstring for the three rules."""
    if not block:
        return "Approach", "nothing within reach"
    if block in TERRAIN:
        return "Hunt", f"a mob standing on {block}"
    if block in CRAFTING:
        return "Craft", block
    if block in CONTAINERS or block.endswith("_shulker_box"):
        return "Use", block
    if any(block.endswith(suffix) for suffix in SWITCHES):
        return "Switch", block
    return "Mine", block


def stop_for(interaction: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """How that interaction ends.

    A stat stop is the right end for Hunt and Mine -- "something died", "a block broke" --
    but only where the sim reports statistics, and whether it does depends on how it was
    built. So each one is used only when its counter is actually in `info`, and falls back
    to a step budget otherwise: a budget that expires is a worse goal, a stat that never
    arrives is a goal that cannot end at all.

    Note the empty `match`: the stop compiler tests `match in key`, and every key contains
    "", so this means "any kill" / "any block" -- which is what we want, since the whole
    point is that we may not know the mob's name.
    """
    if interaction == "Approach":
        return {"steps": BUDGET, "arrive": {"distance": ARRIVE}}
    counter = {"Hunt": "kill_entity", "Mine": "mine_block"}.get(interaction)
    if counter and info.get(counter) is not None:
        return {"stat": counter, "count": 1}
    return {"steps": BUDGET}


def choose(sim, info: Dict[str, Any], point: Sequence[float], shape: Sequence[int],
           fov: float = 70.0) -> Choice:
    """The whole decision: cast the pointed pixel, name what it hit, pick a goal for it."""
    block = block_at(sim, info, point, shape, fov)
    interaction, reason = interaction_for(block)
    return Choice(interaction, stop_for(interaction, info), block, reason)
