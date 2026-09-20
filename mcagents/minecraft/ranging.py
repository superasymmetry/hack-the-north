"""Where the agent is, where its target is, and how far apart they are -- in world blocks.

Apparent size is not distance. A target's box fills the frame when the agent is close to it,
but also when the agent has turned away and the lock has slid onto something nearer, and it
shrinks again the moment the agent walks past. So an arrival test built on pixels alone
answers "does this look big" when the question was "am I there".

The sim already knows the answer. `info['player_pos']` is absolute, and `VoxelsCallback`
puts the block grid around the player in `info['voxels']`. Cast the ray the pointed pixel
stands for through that grid, and the first solid block it meets is the target's address in
the world -- fixed, while the view is not. From there arrival is arithmetic:

    anchor = cast(voxels(sim, info), *view_ray(info, point, frame.shape))
    ...
    horizontal(position(info), anchor) <= 3.0

The catch is the grid's radius (7 blocks by default): a building down the street is out of
range and casts to None. That is why the caller re-casts as it walks -- the anchor arrives
by itself once the target comes inside the grid -- and keeps the width test as the fallback
for everything that never does.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

#: Eye height above the feet, which is what `player_pos` reports. Minecraft's own constant.
EYE = 1.62

#: Block names a ray goes straight through. Everything else counts as the target.
CLEAR = {"air", "cave_air", "void_air", "water", "flowing_water", "", "none"}


def position(info: Dict[str, Any]) -> Optional[np.ndarray]:
    """The player's feet as [x, y, z], or None in a sim not reporting a position."""
    pos = info.get("player_pos") or info.get("location_stats")
    if not pos:
        return None
    try:  # player_pos is x/y/z, location_stats is xpos/ypos/zpos -- the same numbers
        return np.array([float(pos[key]) for key in ("x", "y", "z")])
    except KeyError:
        return np.array([float(pos[f"{key}pos"]) for key in ("x", "y", "z")])


def facing(info: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """(yaw, pitch) in degrees, or None if this sim does not report them."""
    look = info.get("player_pos") or info.get("location_stats") or {}
    if "yaw" not in look or "pitch" not in look:
        return None
    return float(np.reshape(look["yaw"], ())), float(np.reshape(look["pitch"], ()))


def direction(yaw: float, pitch: float) -> np.ndarray:
    """A unit vector for a Minecraft yaw/pitch: yaw 0 faces +z, pitch 90 faces straight down."""
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    return np.array([-math.sin(yaw) * math.cos(pitch), -math.sin(pitch),
                     math.cos(yaw) * math.cos(pitch)])


def view_ray(info: Dict[str, Any], point: Sequence[float], shape: Sequence[int],
             fov: float = 70.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(eye, unit direction) for the pixel `point` on a frame of `shape`, in world coordinates.

    A pinhole camera with `fov` as the *vertical* field of view, which is how Minecraft
    states it and how the lock already converts camera degrees into pixels.
    """
    eye, look = position(info), facing(info)
    if eye is None or look is None:
        return None
    height, width = shape[:2]
    focal = (height / 2) / math.tan(math.radians(fov) / 2)
    dx, dy = float(point[0]) - width / 2, float(point[1]) - height / 2
    yaw = look[0] + math.degrees(math.atan2(dx, focal))
    pitch = look[1] + math.degrees(math.atan2(dy, math.hypot(dx, focal)))
    return eye + np.array([0.0, EYE, 0.0]), direction(yaw, pitch)


@dataclass
class Voxels:
    """The block grid around the player, in absolute block coordinates.

    MineStudio hands the same observation over in two shapes depending on the build -- a
    3-d array of names under `block_name`, or a flat list of `{type, x, y, z}` cells -- so
    this normalizes both to one question: is the block at these world coordinates solid?
    """
    origin: np.ndarray                        # world coordinates of the grid's [0, 0, 0] corner
    shape: Tuple[int, int, int]
    names: Optional[np.ndarray] = None        # [x][y][z], the array shape
    solids: Optional[frozenset] = None        # absolute coordinates, the list shape

    def contains(self, block: np.ndarray) -> bool:
        index = block - self.origin
        return bool(np.all(index >= 0) and np.all(index < np.array(self.shape)))

    def solid(self, block: np.ndarray) -> bool:
        if self.solids is not None:
            return tuple(int(v) for v in block) in self.solids
        i, j, k = (block - self.origin).astype(int)
        return str(self.names[i, j, k]).split(":")[-1].lower() not in CLEAR


def extent(sim) -> Optional[Sequence[int]]:
    """The [x0, x1, y0, y1, z0, z1] a VoxelsCallback on `sim` is asking for, if one is."""
    for callback in getattr(sim, "callbacks", []):
        ins = getattr(callback, "voxels_ins", None)
        if ins is not None:
            return list(ins)
    return None


def voxels(sim, info: Dict[str, Any]) -> Optional[Voxels]:
    """`info['voxels']` as a `Voxels`, or None when the sim was built without the callback."""
    raw, eye = info.get("voxels"), position(info)
    ins = extent(sim)
    if raw is None or eye is None or ins is None:
        return None
    corner = np.floor(eye).astype(int) + np.array([ins[0], ins[2], ins[4]])
    shape = (ins[1] - ins[0] + 1, ins[3] - ins[2] + 1, ins[5] - ins[4] + 1)
    names = raw.get("block_name") if isinstance(raw, dict) else None
    if names is not None:
        # Indexed [x][y][z], the order the instruction states its own bounds in.
        names = np.asarray(names)
        if names.ndim != 3:
            return None
        if names.shape != shape:  # the array is what can be indexed, so it wins; assume it
            corner = np.floor(eye).astype(int) - np.array(names.shape) // 2   # is centred
        return Voxels(corner, names.shape, names=names)
    if not isinstance(raw, (list, tuple)):
        return None
    # The list shape does not say whether its coordinates are absolute or player-relative.
    # They cannot be both: relative ones live inside the requested box and absolute ones,
    # anywhere but the world origin, do not.
    cells = [(str(cell.get("type", cell.get("name", ""))).split(":")[-1].lower(),
              np.array([int(cell["x"]), int(cell["y"]), int(cell["z"])])) for cell in raw]
    relative = all(np.all(np.abs(xyz) <= np.abs(ins).max()) for _, xyz in cells)
    shift = np.floor(eye).astype(int) if relative else np.zeros(3, int)
    return Voxels(corner, shape,
                  solids=frozenset(tuple(xyz + shift) for name, xyz in cells
                                   if name not in CLEAR))


def cast(grid: Optional[Voxels], ray: Optional[Tuple[np.ndarray, np.ndarray]],
         step: float = 0.25) -> Optional[np.ndarray]:
    """The centre of the first solid block along `ray`, or None if it leaves the grid first.

    A fixed-step march rather than a DDA: the grid is ~15 blocks across, so this is at most
    a hundred-odd lookups, and it runs once per goal rather than once per frame.
    """
    if grid is None or ray is None:
        return None
    eye, heading = ray
    limit = int(max(grid.shape) * math.sqrt(3) / step) + 1
    for count in range(1, limit + 1):
        block = np.floor(eye + heading * (count * step)).astype(int)
        if not grid.contains(block):
            return None
        if grid.solid(block):
            return block + 0.5
    return None


def horizontal(here: Optional[np.ndarray], there: Optional[np.ndarray]) -> Optional[float]:
    """Distance in blocks, ignoring height -- the one that means "am I standing next to it".

    Height is dropped on purpose: anchoring on the top of a tower would otherwise leave the
    agent a tower's height away from arriving while standing at its door.
    """
    if here is None or there is None:
        return None
    return float(math.hypot(there[0] - here[0], there[2] - here[2]))
