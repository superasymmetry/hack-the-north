"""Reading and writing the block data of a 1.16.5 save, in Anvil's own format.

`mcagents.minecraft.bake` needs to set several hundred thousand blocks in a world that no
process has open. That means going at the region files directly, and this is the layer that
knows their shape:

- A `.mca` region holds 32x32 chunks. Its first 4KB is a location table, one big-endian
  int per chunk: three bytes of *sector* offset (a sector is 4KB) and one byte of length in
  sectors. A zero entry means the chunk was never generated -- and a chunk missing from the
  file is regenerated from the seed at load time, so an edit to one silently disappears.
  `Region.chunk` returns None for those rather than inventing them.
- The second 4KB is a timestamp table, which nothing here reads and everything preserves.
- Each chunk is a 4-byte length, a compression byte (1 gzip, 2 zlib), then NBT.

The block storage is the part with a version-specific trap in it. A section's 4096 blocks
are indices into its own `Palette`, packed into a `BlockStates` long array at
`max(4, bits needed for the palette)` bits each. Before 1.16 an index could straddle two
longs; from 1.16 (DataVersion 2529) it cannot, and the leftover high bits of each long are
simply wasted. This module implements the 1.16 layout and `Chunk.__init__` refuses a save
older than that rather than writing plausible-looking rubbish.

Everything is decoded to a flat list of 4096 palette indices on first touch and repacked
once at save, because a `/fill` that walks a section block by block would otherwise repack
the whole array per block.
"""
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mcagents.minecraft import nbt

SECTOR = 4096
#: 20w17a. The first version where a block index may not span two longs.
FIRST_UNSPANNED = 2529

AIR = "minecraft:air"


@dataclass(frozen=True)
class Block:
    """A block state: a name and the properties that distinguish it from its siblings.

    Properties are strings on both sides because that is how the palette stores them --
    `open=false` is the five-character string, not a boolean.
    """

    name: str
    properties: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def parse(cls, spec: str) -> "Block":
        """`minecraft:oak_door[facing=west,half=lower]` -> Block.

        Properties are sorted, so two specs listing the same states in a different order
        land on the same palette entry instead of two identical ones.
        """
        spec = spec.strip()
        if "[" not in spec:
            return cls(spec)
        name, _, rest = spec.partition("[")
        pairs = []
        for item in rest.rstrip("]").split(","):
            if item.strip():
                key, _, value = item.partition("=")
                pairs.append((key.strip(), value.strip()))
        return cls(name.strip(), tuple(sorted(pairs)))

    def to_nbt(self) -> Dict:
        entry = {"Name": self.name}
        if self.properties:
            entry["Properties"] = {key: value for key, value in self.properties}
        return entry

    @classmethod
    def from_nbt(cls, entry: Dict) -> "Block":
        properties = entry.get("Properties") or {}
        return cls(entry["Name"], tuple(sorted((k, str(v)) for k, v in properties.items())))


AIR_BLOCK = Block(AIR)


# ---------------------------------------------------------------- bit packing

def unpack_states(longs: List[int], bits: int, count: int = 4096) -> List[int]:
    """Palette indices out of a 1.16 BlockStates array.

    The longs arrive signed (NBT has no unsigned types) and are masked back to 64 bits
    before shifting, or every index in the top half of a long comes out negative.
    """
    per_long = 64 // bits
    mask = (1 << bits) - 1
    out = [0] * count
    for index in range(count):
        word = longs[index // per_long] & 0xFFFFFFFFFFFFFFFF
        out[index] = (word >> (index % per_long) * bits) & mask
    return out


def pack_states(indices: List[int], bits: int) -> List[int]:
    """The inverse. Returns signed longs, which is how they have to go back into NBT."""
    per_long = 64 // bits
    total = (len(indices) + per_long - 1) // per_long
    words = [0] * total
    for index, value in enumerate(indices):
        words[index // per_long] |= value << (index % per_long) * bits
    return [word - (1 << 64) if word >= 1 << 63 else word for word in words]


def bits_for(palette_size: int) -> int:
    """Bits per index for a palette of this size. Never fewer than four."""
    return max(4, (palette_size - 1).bit_length())


# ---------------------------------------------------------------- chunks

class Section:
    """One 16x16x16 slice, decoded to palette + 4096 indices and repacked on the way out."""

    def __init__(self, y: int, tag: Optional[Dict] = None):
        self.y = y
        self.tag = tag if tag is not None else {"Y": nbt.Byte(y)}
        raw_palette = self.tag.get("Palette")
        if raw_palette is None:
            # No palette means the section is empty air -- 1.16 omits both tags for those.
            self.palette: List[Block] = [AIR_BLOCK]
            self.indices: List[int] = [0] * 4096
        else:
            self.palette = [Block.from_nbt(entry) for entry in raw_palette]
            self.indices = unpack_states(self.tag["BlockStates"], bits_for(len(self.palette)))
        self.lookup = {block: i for i, block in enumerate(self.palette)}
        self.dirty = False

    def index_of(self, block: Block) -> int:
        existing = self.lookup.get(block)
        if existing is None:
            existing = self.lookup[block] = len(self.palette)
            self.palette.append(block)
        return existing

    def set(self, x: int, y: int, z: int, block: Block) -> None:
        """Set one block by its coordinates *within* the section (0..15 on each axis)."""
        self.indices[y << 8 | z << 4 | x] = self.index_of(block)
        self.dirty = True

    def get(self, x: int, y: int, z: int) -> Block:
        return self.palette[self.indices[y << 8 | z << 4 | x]]

    def flush(self) -> Dict:
        """Repack into NBT, dropping palette entries nothing points at any more.

        A `/fill air` over a whole plot leaves the old grass and logs in the palette,
        which widens every index and grows the file for blocks that are no longer there.
        """
        used = sorted(set(self.indices))
        remap = {old: new for new, old in enumerate(used)}
        self.palette = [self.palette[old] for old in used]
        self.indices = [remap[i] for i in self.indices]
        self.lookup = {block: i for i, block in enumerate(self.palette)}

        bits = bits_for(len(self.palette))
        self.tag["Y"] = nbt.Byte(self.y)
        self.tag["Palette"] = nbt.TagList(nbt.COMPOUND, [b.to_nbt() for b in self.palette])
        self.tag["BlockStates"] = nbt.LongArray(pack_states(self.indices, bits))
        return self.tag


class Chunk:
    """One chunk's NBT, with its sections decoded lazily as blocks are written into them."""

    def __init__(self, root: Dict):
        self.root = root
        version = int(root.get("DataVersion", 0))
        if version < FIRST_UNSPANNED:
            raise ValueError(f"DataVersion {version} packs block states differently; "
                             f"this module only writes the 1.16+ layout")
        self.level = root["Level"]
        self.sections: Dict[int, Section] = {}
        self._tags = {int(tag["Y"]): tag for tag in self.level.get("Sections", [])}
        self.dirty = False

    def section(self, y_index: int, create: bool) -> Optional[Section]:
        """The section at this section-Y, decoding or creating it on demand.

        `create=False` is what keeps a plot-clearing `/fill air` cheap: the sky above a
        plains world is hundreds of thousands of blocks of nothing, and filling air into a
        section that does not exist would materialise it just to store zeros.
        """
        section = self.sections.get(y_index)
        if section is None:
            tag = self._tags.get(y_index)
            if tag is None and not create:
                return None
            section = self.sections[y_index] = Section(y_index, tag)
        return section

    def set_block(self, x: int, y: int, z: int, block: Block) -> bool:
        """Set one block by world coordinates. Returns whether anything changed."""
        if not 0 <= y < 256:
            return False
        section = self.section(y >> 4, create=block != AIR_BLOCK)
        if section is None:
            return False
        section.set(x & 15, y & 15, z & 15, block)
        self.dirty = True
        return True

    def get_block(self, x: int, y: int, z: int) -> Block:
        section = self.section(y >> 4, create=False)
        return AIR_BLOCK if section is None else section.get(x & 15, y & 15, z & 15)

    def flush(self) -> Dict:
        """Fold the decoded sections back in and hand the light and height maps to the game.

        Two deletions do real work here. `isLightOn=false` is how a chunk says its stored
        lighting is stale, which makes the server relight it on load -- without it a new
        building interior is lit as though the sky still reached the ground. And dropping
        `Heightmaps` makes 1.16's chunk reader prime them itself, which is both less code
        and more correct than recomputing them here.
        """
        if not self.dirty:
            return self.root
        tags = dict(self._tags)
        for y_index, section in self.sections.items():
            if section.dirty:
                tags[y_index] = section.flush()
        self.level["Sections"] = nbt.TagList(
            nbt.COMPOUND, [tags[y] for y in sorted(tags)])
        self.level["isLightOn"] = nbt.Byte(0)
        self.level.pop("Heightmaps", None)
        return self.root


# ---------------------------------------------------------------- regions

@dataclass
class Region:
    """One `.mca` file: the chunks it holds, and the ability to write them back."""

    path: Path
    chunks: Dict[Tuple[int, int], Chunk] = field(default_factory=dict)
    _raw: Dict[Tuple[int, int], bytes] = field(default_factory=dict)
    _timestamps: bytes = b""

    @classmethod
    def load(cls, path) -> "Region":
        path = Path(path)
        data = path.read_bytes()
        # A region Minecraft touched but never wrote a chunk into is a zero-byte file, and
        # there are four of them in every world this repo caches. It holds no chunks, which
        # is a fine thing for a region to hold; it is not a corrupt file.
        if len(data) < SECTOR * 2:
            return cls(path)
        region = cls(path, _timestamps=data[SECTOR:SECTOR * 2])
        for slot in range(1024):
            offset, sectors = struct.unpack(">I", data[slot * 4:slot * 4 + 4])[0] >> 8, data[slot * 4 + 3]
            if not offset:
                continue
            start = offset * SECTOR
            length = struct.unpack(">I", data[start:start + 4])[0]
            region._raw[(slot % 32, slot // 32)] = data[start + 4:start + 4 + length]
        return region

    def chunk(self, local_x: int, local_z: int) -> Optional[Chunk]:
        """The chunk at these region-local coordinates, or None if it was never generated."""
        key = (local_x, local_z)
        chunk = self.chunks.get(key)
        if chunk is None:
            raw = self._raw.get(key)
            if raw is None:
                return None
            chunk = self.chunks[key] = Chunk(nbt.loads(nbt.decompress(raw[1:])))
        return chunk

    def save(self) -> None:
        """Rewrite the file, re-laying every chunk out from sector 2 in slot order.

        Rewriting the whole file rather than editing it in place is what makes the sector
        arithmetic tractable: an edited chunk usually changes size, and growing one in place
        means finding or making a run of free sectors.
        """
        for key, chunk in self.chunks.items():
            if chunk.dirty:
                self._raw[key] = b"\x02" + zlib.compress(nbt.dumps(chunk.flush()))

        locations = bytearray(SECTOR)
        body = bytearray()
        next_sector = 2
        for key, raw in sorted(self._raw.items(), key=lambda item: (item[0][1], item[0][0])):
            payload = struct.pack(">I", len(raw)) + raw
            padded = payload + b"\x00" * (-len(payload) % SECTOR)
            sectors = len(padded) // SECTOR
            if sectors > 255:
                raise ValueError(f"chunk {key} needs {sectors} sectors; the header holds one byte")
            slot = key[0] + key[1] * 32
            struct.pack_into(">I", locations, slot * 4, next_sector << 8 | sectors)
            body += padded
            next_sector += sectors

        timestamps = self._timestamps or bytes(SECTOR)
        self.path.write_bytes(bytes(locations) + timestamps + bytes(body))


class World:
    """A save directory, addressed in world block coordinates.

    Regions are opened on demand and held until `save()`, because the city plot spans one
    region but the code has no business assuming that.
    """

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.regions: Dict[Tuple[int, int], Optional[Region]] = {}
        self.missing_chunks = set()
        # A `/fill` walks a row of x through one chunk sixteen blocks at a time, so the
        # chunk it wants is nearly always the one it wanted last. Remembering one turns
        # two dict lookups per block into a tuple compare, over ~600k blocks.
        self._recent: Tuple[Tuple[int, int], Optional[Chunk]] = ((1 << 30, 0), None)

    def region(self, region_x: int, region_z: int) -> Optional[Region]:
        key = (region_x, region_z)
        if key not in self.regions:
            path = self.save_dir / "region" / f"r.{region_x}.{region_z}.mca"
            self.regions[key] = Region.load(path) if path.exists() else None
        return self.regions[key]

    def chunk_at(self, x: int, z: int) -> Optional[Chunk]:
        chunk_x, chunk_z = x >> 4, z >> 4
        key, cached = self._recent
        if key == (chunk_x, chunk_z):
            return cached
        chunk = self._chunk_at(chunk_x, chunk_z)
        self._recent = ((chunk_x, chunk_z), chunk)
        return chunk

    def _chunk_at(self, chunk_x: int, chunk_z: int) -> Optional[Chunk]:
        region = self.region(chunk_x >> 5, chunk_z >> 5)
        if region is None:
            self.missing_chunks.add((chunk_x, chunk_z))
            return None
        chunk = region.chunk(chunk_x & 31, chunk_z & 31)
        if chunk is None:
            self.missing_chunks.add((chunk_x, chunk_z))
        return chunk

    def set_block(self, x: int, y: int, z: int, block: Block) -> bool:
        chunk = self.chunk_at(x, z)
        return False if chunk is None else chunk.set_block(x, y, z, block)

    def get_block(self, x: int, y: int, z: int) -> Block:
        chunk = self.chunk_at(x, z)
        return AIR_BLOCK if chunk is None else chunk.get_block(x, y, z)

    def save(self) -> None:
        for region in self.regions.values():
            if region is not None:
                region.save()
