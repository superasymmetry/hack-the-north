"""What a pointed pixel is, and therefore what ROCKET-2 should do about it.

A pinch gives a pixel. ROCKET-2 wants a pixel *and* an interaction (Hunt, Mine, Approach --
see mcagents/agents/rocket2.py), and nothing in a pinch says which. This closes that gap
without a model: `info['voxels']` already carries the block grid around the player, so
casting the pointed pixel through it (mcagents/minecraft/ranging.py) answers "what block is
under that pixel" exactly, for free, from the observation the sim was already producing.

    choice = identify.choose(sim, agent.info, point, frame.shape)
    agent.run(point=point, interaction=choice.interaction, stop=choice.stop)

The one thing the grid cannot see is mobs: it is a *block* grid, so a ray aimed at a cow
goes straight through it and lands on the ground behind. That miss is the signal. A ray that
ends on the floor found nothing on the way to it, and the pinch was aimed at whatever was
standing there. So:

    hit something at or above your feet  ->  it is that block          ->  Mine / Switch / Use
    hit the floor below your feet        ->  something was in the way  ->  Hunt
    hit nothing inside the grid          ->  further than ~7 blocks    ->  Approach

Height rather than a list of floor-looking blocks, because the material does not divide the
way the question does: `CityCallback` paves its streets in `gray_concrete` and roofs its
buildings with the same block, and walks them in `smooth_stone` while the hills behind are
stone too. What a floor has in common is not what it is made of but where it is -- under
you. A wall is not, a chest standing on the floor is not, and both stay minable and usable.

The middle rule is what buys us no second model, and it costs the gesture "pinch the ground
to walk there", which now reads as Hunt. When that matters the fix is a real classifier over
the pinched region -- OWLv2 (mcagents/perception/owlv2.py) run over a fixed label vocabulary,
picking the box that contains the point -- dropped in behind the same `choose()` signature.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from mcagents.minecraft import ranging

#: Ground even when it is not below you: the plants a mob stands *in* rather than on, which
#: a flat-ish ray reaches at your own height and which nobody pinches on purpose. The blocks
#: you walk on need no list at all -- see the height rule in `is_floor`.
TERRAIN = frozenset({
    "grass", "short_grass", "tall_grass", "fern", "large_fern", "dead_bush", "seagrass",
    "snow", "powder_snow", "sweet_berry_bush", "sugar_cane", "wheat", "vine",
    "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "oxeye_daisy",
    "cornflower", "lily_of_the_valley", "sunflower", "lilac", "rose_bush", "peony",
    "red_tulip", "orange_tulip", "white_tulip", "pink_tulip",
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


def probe(sim, info: Dict[str, Any], point: Sequence[float], shape: Sequence[int],
          fov: float = 70.0) -> Optional[Tuple[np.ndarray, str]]:
    """(block centre, block name) for the pointed pixel, or None if the ray hits nothing.

    None covers three cases that behave the same way -- further than the grid reaches, a sim
    built without VoxelsCallback, and a sim not reporting a position -- and all three mean
    the same thing here: nothing is known about the target, so walk toward it.
    """
    return ranging.hit(ranging.voxels(sim, info),
                       ranging.view_ray(info, point, shape, fov))


def is_floor(centre: Optional[np.ndarray], info: Dict[str, Any]) -> bool:
    """Whether a hit block is ground the player could be standing on rather than a target.

    Strictly *below* the feet, not at them: `player_pos` reports the feet, so the block you
    stand on is one lower, and a block level with your feet is something sitting on the
    floor -- a chest, the bottom course of a wall -- which is a target like any other. The
    comparison is on block coordinates, so a cow up a step or down a kerb still reads as a
    cow; only a target a whole block above your feet reads as itself.
    """
    feet = ranging.position(info)
    if centre is None or feet is None:
        return False
    return math.floor(centre[1]) < math.floor(feet[1])


def interaction_for(block: Optional[str], floor: bool = False) -> Tuple[str, str]:
    """(interaction, why) for a hit. See the module docstring for the three rules."""
    if not block:
        return "Approach", "nothing within reach"
    if floor or block in TERRAIN:
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
    landed = probe(sim, info, point, shape, fov)
    block = None if landed is None else landed[1]
    interaction, reason = interaction_for(block, is_floor(landed and landed[0], info))
    return Choice(interaction, stop_for(interaction, info), block, reason)
