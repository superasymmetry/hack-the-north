"""Check that a pinched pixel picks the interaction it should, without Minecraft.

`identify` is the piece that decides what ROCKET-2 does with a pinch, and it decides from
one thing only: the block name the ray lands on. That makes it testable the same way
`ranging` is -- put a known block in a synthetic grid, point at it, and check the verdict.

    python tests/test_identify.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcagents.minecraft import ranging
from mcagents.perception import identify

FRAME = (360, 640, 3)
REACH = 7
CENTRE = (FRAME[1] / 2, FRAME[0] / 2)     # straight ahead, where a pinch at the crosshair is


class StubSim:
    """Just enough sim to say what voxel range was asked for."""

    def __init__(self, reach: int = REACH):
        self.callbacks = [type("VoxelsCallback", (), {"voxels_ins": [-reach, reach] * 3})()]


def scene(block=None, at=(0, 1, 3), reach=REACH, stats=()):
    """A player looking down +z with one named block in front of them, or empty air.

    `at` is a block offset from the player's feet, so y=1 is eye level -- where a ray cast
    straight ahead lands. Which offset a real target sits at does not matter to `identify`:
    the verdict comes from the block's *name*, and the geometry is `ranging`'s problem.
    """
    names = np.full((2 * reach + 1,) * 3, "minecraft:air", dtype=object)
    if block is not None:
        names[tuple(np.array(at) + reach)] = f"minecraft:{block}"
    info = {"player_pos": {"x": 100.5, "y": 64.0, "z": 200.5, "yaw": 0.0, "pitch": 0.0},
            "voxels": {"block_name": names}}
    for stat in stats:
        info[stat] = {}
    return info


def choose(block, **kwargs):
    return identify.choose(StubSim(), scene(block, **kwargs), CENTRE, FRAME)


def check_block_names_come_back() -> None:
    """The whole design rests on the cast reporting *what* it hit, not just where."""
    info = scene("oak_log")
    landed = ranging.hit(ranging.voxels(StubSim(), info),
                         ranging.view_ray(info, CENTRE, FRAME))
    assert landed is not None
    position, name = landed
    assert name == "oak_log", name
    assert np.allclose(position, [100.5, 65.5, 203.5]), position
    # The flat observation shape carries names too, and must agree with the array shape.
    flat = {"player_pos": info["player_pos"],
            "voxels": [{"type": "minecraft:oak_log", "x": 0, "y": 1, "z": 3}]}
    landed = ranging.hit(ranging.voxels(StubSim(), flat),
                         ranging.view_ray(flat, CENTRE, FRAME))
    assert landed is not None and landed[1] == "oak_log", landed


def check_the_three_rules() -> None:
    """Structure is itself, terrain means something stood in front of it, nothing means walk."""
    assert choose("oak_log").interaction == "Mine"
    assert choose("iron_ore").interaction == "Mine"
    assert choose("stone_bricks").interaction == "Mine"
    # The headline case: a cow occupies no block, so the ray goes through it to the ground.
    grass = choose("grass_block")
    assert grass.interaction == "Hunt", grass
    assert grass.block == "grass_block"
    # Nothing within the grid at all -- further away than the voxels reach.
    far = choose(None)
    assert far.interaction == "Approach" and far.block is None, far


def check_operables() -> None:
    """Things you open or flip are not things you break, whatever wood they are made of."""
    for block in ("oak_door", "warped_trapdoor", "lever", "birch_fence_gate", "stone_button"):
        assert choose(block).interaction == "Switch", block
    for block in ("chest", "furnace", "blue_shulker_box", "anvil"):
        assert choose(block).interaction == "Use", block
    assert choose("crafting_table").interaction == "Craft"


def check_stops_match_the_interaction() -> None:
    """Each goal ends the way that interaction can actually end."""
    assert choose(None).stop == {"steps": identify.BUDGET,
                                 "arrive": {"distance": identify.ARRIVE}}
    # No statistics in this sim's info, so a stat stop would never fire: budget instead.
    assert choose("oak_log").stop == {"steps": identify.BUDGET}
    assert choose("grass_block").stop == {"steps": identify.BUDGET}
    # With the counters present, the goal ends on the event rather than the clock. The empty
    # match is deliberate -- see stop_for -- so "any kill" ends a Hunt.
    mining = identify.choose(StubSim(), scene("oak_log", stats=("mine_block",)), CENTRE, FRAME)
    assert mining.stop == {"stat": "mine_block", "count": 1}, mining.stop
    hunting = identify.choose(StubSim(), scene("grass_block", stats=("kill_entity",)),
                              CENTRE, FRAME)
    assert hunting.stop == {"stat": "kill_entity", "count": 1}, hunting.stop


def check_degrades_without_voxels() -> None:
    """No grid, no position, no callback: all three mean "I know nothing", i.e. Approach."""
    info = scene("oak_log")
    del info["voxels"]
    assert identify.choose(StubSim(), info, CENTRE, FRAME).interaction == "Approach"
    bare = type("Bare", (), {"callbacks": []})()
    assert identify.choose(bare, scene("oak_log"), CENTRE, FRAME).interaction == "Approach"
    assert identify.choose(StubSim(), {}, CENTRE, FRAME).interaction == "Approach"


def main() -> int:
    for check in (check_block_names_come_back, check_the_three_rules, check_operables,
                  check_stops_match_the_interaction, check_degrades_without_voxels):
        check()
        print(f"  {check.__name__} ok")
    print("\nALL IDENTIFY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
