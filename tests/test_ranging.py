"""Check the world-coordinate arithmetic without a GPU, a world, or Minecraft.

`ranging` is the piece that turns a pointed pixel into a block address, so everything it
gets wrong is silent: a sign flipped in the yaw formula still returns a plausible number,
and the agent simply walks the wrong way. A synthetic grid with one known wall in it is an
unambiguous oracle -- the ray either lands on the wall or it does not.

    python tests/test_ranging.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcagents.minecraft import ranging

FRAME = (360, 640, 3)
REACH = 7


class StubSim:
    """Just enough sim to say what voxel range was asked for."""

    def __init__(self, reach: int = REACH):
        self.callbacks = [type("VoxelsCallback", (), {"voxels_ins": [-reach, reach] * 3})()]


def scene(x=100.5, y=64.0, z=200.5, yaw=0.0, pitch=0.0, wall=None, form="array",
          reach=REACH):
    """A player at a known spot, with an optional solid wall at a block offset from them."""
    names = np.full((2 * reach + 1,) * 3, "minecraft:air", dtype=object)
    if wall is not None and max(abs(v) for v in wall) <= reach:
        names[tuple(np.array(wall) + reach)] = "minecraft:stone"
    info = {"player_pos": {"x": x, "y": y, "z": z, "yaw": yaw, "pitch": pitch}}
    if form == "array":
        info["voxels"] = {"block_name": names}
    else:  # the flat shape, in player-relative coordinates
        info["voxels"] = [{"type": "minecraft:stone", "x": wall[0], "y": wall[1], "z": wall[2]}] \
            if wall is not None else []
    return info


def check_position_and_facing() -> None:
    info = scene(yaw=90.0, pitch=-20.0)
    assert np.allclose(ranging.position(info), [100.5, 64.0, 200.5])
    assert ranging.facing(info) == (90.0, -20.0)
    # location_stats is the same numbers under other names, and the only source in a replay.
    other = {"location_stats": {"xpos": 1.0, "ypos": 2.0, "zpos": 3.0, "yaw": 0.0, "pitch": 0.0}}
    assert np.allclose(ranging.position(other), [1.0, 2.0, 3.0])
    assert ranging.position({}) is None and ranging.facing({}) is None


def check_direction() -> None:
    """Minecraft's convention: yaw 0 faces +z, yaw 90 faces -x, pitch 90 faces down."""
    for yaw, expected in ((0, [0, 0, 1]), (90, [-1, 0, 0]), (180, [0, 0, -1]), (270, [1, 0, 0])):
        assert np.allclose(ranging.direction(yaw, 0), expected, atol=1e-9), yaw
    assert np.allclose(ranging.direction(0, 90), [0, -1, 0], atol=1e-9)


def check_view_ray() -> None:
    info = scene(yaw=0.0, pitch=0.0)
    eye, centre = ranging.view_ray(info, (320, 180), FRAME)
    assert np.allclose(eye, [100.5, 64.0 + ranging.EYE, 200.5])
    assert np.allclose(centre, [0, 0, 1], atol=1e-9), centre   # the crosshair is where you face

    # A pixel on the right edge is half the horizontal field of view away, and to the right
    # means +yaw, which at yaw 0 means -x.
    _, edge = ranging.view_ray(info, (639, 180), FRAME)
    assert edge[0] < 0 and edge[2] > 0, edge
    half = math.degrees(math.atan2(320, (180) / math.tan(math.radians(35))))
    assert abs(math.degrees(math.atan2(-edge[0], edge[2])) - half) < 0.2

    _, low = ranging.view_ray(info, (320, 359), FRAME)
    assert low[1] < 0, low                                     # down the frame is down the world


def check_cast() -> None:
    """A wall four blocks ahead is found at four blocks; empty air is not found at all.

    The wall sits one block above the feet because that is where the eyes are: a level ray
    from y + 1.62 travels through the block above the one the player is standing level with.
    """
    for form in ("array", "list"):
        info = scene(wall=(0, 1, 4), form=form)
        grid = ranging.voxels(StubSim(), info)
        hit = ranging.cast(grid, ranging.view_ray(info, (320, 180), FRAME))
        assert hit is not None, form
        assert np.allclose(hit, [100.5, 65.5, 204.5]), (form, hit)
        assert abs(ranging.horizontal(ranging.position(info), hit) - 4.0) < 0.01, form

    empty = scene()
    assert ranging.cast(ranging.voxels(StubSim(), empty),
                        ranging.view_ray(empty, (320, 180), FRAME)) is None
    # Beyond the grid there is nothing to hit, however solid the real world is out there.
    far = scene(wall=(0, 1, 7))
    ray = ranging.view_ray(far, (320, 180), FRAME)
    assert ranging.cast(ranging.voxels(StubSim(), far), ray) is not None
    near_sighted = scene(wall=(0, 1, 7), reach=3)
    assert ranging.cast(ranging.voxels(StubSim(reach=3), near_sighted),
                        ranging.view_ray(near_sighted, (320, 180), FRAME)) is None


def check_turning_does_not_move_the_target() -> None:
    """The point of all of this: the anchor is fixed, so facing away changes only the view."""
    info = scene(wall=(0, 1, 4))
    anchor = ranging.cast(ranging.voxels(StubSim(), info),
                          ranging.view_ray(info, (320, 180), FRAME))
    spun = scene(yaw=180.0)
    assert ranging.horizontal(ranging.position(spun), anchor) == 4.0
    # Walking two blocks toward it does move it, by two blocks.
    closer = scene(z=202.5)
    assert abs(ranging.horizontal(ranging.position(closer), anchor) - 2.0) < 1e-9


def check_no_voxels() -> None:
    """Without the callback there is nothing to cast through, and nothing pretends otherwise."""
    info = scene()
    del info["voxels"]
    assert ranging.voxels(StubSim(), info) is None
    assert ranging.voxels(type("Bare", (), {"callbacks": []})(), scene()) is None
    assert ranging.cast(None, None) is None
    assert ranging.horizontal(None, np.zeros(3)) is None


def main() -> int:
    for check in (check_position_and_facing, check_direction, check_view_ray, check_cast,
                  check_turning_does_not_move_the_target, check_no_voxels):
        check()
        print(f"  {check.__name__} ok")
    print("\nALL RANGING TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
