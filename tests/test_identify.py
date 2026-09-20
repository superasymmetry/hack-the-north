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


def scene(block=None, at=(0, 1, 3), reach=REACH, stats=(), floor=None, pitch=0.0):
    """A player looking down +z with one named block in front of them, or empty air.

    `at` is a block offset from the player's feet, so y=1 is eye level -- where a ray cast
    straight ahead lands, and where a target standing in front of you is. `floor` paves the
    whole y=-1 layer, which is the block you are standing on: a ray aimed downward reaches
    it wherever it is pointed, exactly as one aimed at a cow does.
    """
    names = np.full((2 * reach + 1,) * 3, "minecraft:air", dtype=object)
    if floor is not None:
        names[:, reach - 1, :] = f"minecraft:{floor}"
    if block is not None:
        names[tuple(np.array(at) + reach)] = f"minecraft:{block}"
    info = {"player_pos": {"x": 100.5, "y": 64.0, "z": 200.5, "yaw": 0.0, "pitch": pitch},
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
    # The ground here is what CityCallback walks its streets with, and the buildings in the
    # same city are concrete -- so this cannot be decided by the block's name.
    cow = identify.choose(StubSim(), scene(floor="smooth_stone", pitch=25.0), CENTRE, FRAME)
    assert cow.interaction == "Hunt", cow
    assert cow.block == "smooth_stone"
    # Plants are the exception the name list still exists for: a mob stands *in* them, so
    # they come back at eye height rather than under your feet.
    assert choose("tall_grass").interaction == "Hunt"
    # Nothing within the grid at all -- further away than the voxels reach.
    far = choose(None)
    assert far.interaction == "Approach" and far.block is None, far


def check_height_beats_material() -> None:
    """The same block is floor under your feet and a target at them. Nothing else changes."""
    paving = "gray_concrete"        # CityCallback's roads -- and its roofs
    under = identify.choose(StubSim(), scene(floor=paving, pitch=25.0), CENTRE, FRAME)
    assert under.interaction == "Hunt", under
    # Level with the feet, i.e. standing on the floor rather than being it: a target.
    on = identify.choose(StubSim(), scene(paving, at=(0, 0, 3), floor=paving, pitch=20.0),
                         CENTRE, FRAME)
    assert on.block == paving and on.interaction == "Mine", on
    # A chest on the floor stays a chest, and a door in a wall stays a door.
    for block, expected in (("chest", "Use"), ("oak_door", "Switch")):
        hit = identify.choose(StubSim(), scene(block, at=(0, 0, 3), floor=paving, pitch=20.0),
                              CENTRE, FRAME)
        assert hit.block == block and hit.interaction == expected, hit


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
    assert choose("tall_grass").stop == {"steps": identify.BUDGET}
    # With the counters present, the goal ends on the event rather than the clock. The empty
    # match is deliberate -- see stop_for -- so "any kill" ends a Hunt.
    mining = identify.choose(StubSim(), scene("oak_log", stats=("mine_block",)), CENTRE, FRAME)
    assert mining.stop == {"stat": "mine_block", "count": 1}, mining.stop
    hunting = identify.choose(StubSim(), scene(floor="smooth_stone", pitch=25.0,
                                                stats=("kill_entity",)), CENTRE, FRAME)
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
    for check in (check_block_names_come_back, check_the_three_rules,
                  check_height_beats_material, check_operables,
                  check_stops_match_the_interaction, check_degrades_without_voxels):
        check()
        print(f"  {check.__name__} ok")
    print("\nALL IDENTIFY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
