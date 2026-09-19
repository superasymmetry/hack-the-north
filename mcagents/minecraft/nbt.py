"""A minimal NBT codec -- enough to read and rewrite a 1.16.5 save, and no more.

Nothing in the environment stack can write a modified world back to disk (see
`mcagents.minecraft.city`), so `mcagents.minecraft.bake` edits the save files directly. That
needs NBT, and the installed environment has no nbt library, so here is one.

It is a *round-tripping* codec, which is the only interesting requirement. A chunk read and
written straight back out must be byte-identical, or the parts of the chunk this repo does
not understand -- entities, structure references, tick lists -- quietly rot. NBT does not
carry its own numeric widths in Python's type system, so the integer tags get one class
each: reading a TAG_Short yields a `Short`, and writing it back emits a TAG_Short rather
than guessing from the value. `TagList` likewise remembers its element type, because an
empty list still has one and inferring from `items[0]` would crash on it.

    root = read_compressed(path)             # gzip or zlib, sniffed
    root["Data"]["SpawnX"] = Int(3394)
    write_compressed(path, root, gzip_format=True)
"""
import gzip
import struct
import zlib
from typing import Any, BinaryIO, Dict, List, Tuple

END, BYTE, SHORT, INT, LONG, FLOAT, DOUBLE = 0, 1, 2, 3, 4, 5, 6
BYTE_ARRAY, STRING, LIST, COMPOUND, INT_ARRAY, LONG_ARRAY = 7, 8, 9, 10, 11, 12


# ---------------------------------------------------------------- tag types

class Byte(int):
    tag_id = BYTE


class Short(int):
    tag_id = SHORT


class Int(int):
    tag_id = INT


class Long(int):
    tag_id = LONG


class Float(float):
    tag_id = FLOAT


class Double(float):
    tag_id = DOUBLE


class ByteArray(list):
    tag_id = BYTE_ARRAY


class IntArray(list):
    tag_id = INT_ARRAY


class LongArray(list):
    tag_id = LONG_ARRAY


class TagList(list):
    """A TAG_List, which is a list plus the type of the things in it.

    The element type is carried explicitly rather than read off `self[0]` because an empty
    list still declares one, and the game does care: an empty Sections list written as
    TAG_End where it was TAG_Compound is a corrupt chunk.
    """

    tag_id = LIST

    def __init__(self, element_type: int = END, items=()):
        super().__init__(items)
        self.element_type = element_type


#: Python types that map onto a tag without a wrapper class. `bool` must precede `int`:
#: it is a subclass of it, and NBT has no boolean, so True has to land on TAG_Byte.
_PLAIN = [(bool, BYTE), (int, INT), (float, DOUBLE), (str, STRING), (dict, COMPOUND)]


def tag_id_of(value: Any) -> int:
    """The tag type `value` will be written as."""
    explicit = getattr(value, "tag_id", None)
    if explicit is not None:
        return explicit
    for python_type, tag in _PLAIN:
        if isinstance(value, python_type):
            return tag
    raise TypeError(f"no NBT tag for {type(value).__name__}")


# ---------------------------------------------------------------- reading

class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, count: int) -> bytes:
        chunk = self.data[self.pos:self.pos + count]
        if len(chunk) != count:
            raise ValueError(f"truncated NBT: wanted {count} bytes at {self.pos}")
        self.pos += count
        return chunk

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack_from(fmt, self.data, self._advance(size))[0]

    def _advance(self, size: int) -> int:
        start = self.pos
        self.pos += size
        if self.pos > len(self.data):
            raise ValueError("truncated NBT")
        return start

    def string(self) -> str:
        # NBT strings are "modified UTF-8"; for the ASCII a Minecraft save actually contains
        # that is plain UTF-8, and surrogateescape keeps anything exotic round-tripping.
        return self.take(self.unpack(">H")).decode("utf-8", "surrogateescape")

    def value(self, tag: int) -> Any:
        if tag == BYTE:
            return Byte(self.unpack(">b"))
        if tag == SHORT:
            return Short(self.unpack(">h"))
        if tag == INT:
            return Int(self.unpack(">i"))
        if tag == LONG:
            return Long(self.unpack(">q"))
        if tag == FLOAT:
            return Float(self.unpack(">f"))
        if tag == DOUBLE:
            return Double(self.unpack(">d"))
        if tag == BYTE_ARRAY:
            count = self.unpack(">i")
            return ByteArray(struct.unpack(f">{count}b", self.take(count)))
        if tag == STRING:
            return self.string()
        if tag == LIST:
            element = self.unpack(">B")
            count = self.unpack(">i")
            return TagList(element, [self.value(element) for _ in range(count)])
        if tag == COMPOUND:
            out: Dict[str, Any] = {}
            while True:
                child = self.unpack(">B")
                if child == END:
                    return out
                # Name first, into a local: `out[self.string()] = self.value(child)` reads
                # the value before the key, because Python evaluates an assignment's right
                # side first -- which swaps two reads that are not commutative on a stream.
                name = self.string()
                out[name] = self.value(child)
        if tag == INT_ARRAY:
            count = self.unpack(">i")
            return IntArray(struct.unpack(f">{count}i", self.take(count * 4)))
        if tag == LONG_ARRAY:
            count = self.unpack(">i")
            return LongArray(struct.unpack(f">{count}q", self.take(count * 8)))
        raise ValueError(f"unknown NBT tag {tag} at {self.pos}")


def loads(data: bytes) -> Dict[str, Any]:
    """Parse uncompressed NBT. The outer tag is a named compound; the name is always ""."""
    reader = _Reader(data)
    tag = reader.unpack(">B")
    if tag != COMPOUND:
        raise ValueError(f"NBT root is tag {tag}, not a compound")
    reader.string()
    return reader.value(COMPOUND)


# ---------------------------------------------------------------- writing

def _write_string(out: bytearray, text: str) -> None:
    encoded = text.encode("utf-8", "surrogateescape")
    out += struct.pack(">H", len(encoded)) + encoded


def _write_value(out: bytearray, tag: int, value: Any) -> None:
    if tag == BYTE:
        out += struct.pack(">b", int(value))
    elif tag == SHORT:
        out += struct.pack(">h", int(value))
    elif tag == INT:
        out += struct.pack(">i", int(value))
    elif tag == LONG:
        out += struct.pack(">q", int(value))
    elif tag == FLOAT:
        out += struct.pack(">f", float(value))
    elif tag == DOUBLE:
        out += struct.pack(">d", float(value))
    elif tag == BYTE_ARRAY:
        out += struct.pack(f">i{len(value)}b", len(value), *value)
    elif tag == STRING:
        _write_string(out, value)
    elif tag == LIST:
        element = getattr(value, "element_type", END)
        if value and element == END:
            element = tag_id_of(value[0])
        out += struct.pack(">Bi", element, len(value))
        for item in value:
            _write_value(out, element, item)
    elif tag == COMPOUND:
        for name, item in value.items():
            child = tag_id_of(item)
            out += struct.pack(">B", child)
            _write_string(out, name)
            _write_value(out, child, item)
        out += struct.pack(">B", END)
    elif tag == INT_ARRAY:
        out += struct.pack(f">i{len(value)}i", len(value), *value)
    elif tag == LONG_ARRAY:
        out += struct.pack(f">i{len(value)}q", len(value), *value)
    else:
        raise ValueError(f"cannot write NBT tag {tag}")


def dumps(root: Dict[str, Any]) -> bytes:
    """Serialise a compound as a complete NBT document."""
    out = bytearray(struct.pack(">B", COMPOUND))
    _write_string(out, "")
    _write_value(out, COMPOUND, root)
    return bytes(out)


# ---------------------------------------------------------------- files

def read_compressed(path) -> Dict[str, Any]:
    """Read level.dat and friends, sniffing gzip vs zlib vs neither from the first bytes."""
    return loads(decompress(open(path, "rb").read()))


def decompress(raw: bytes) -> bytes:
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    if raw[:1] == b"\x78":
        return zlib.decompress(raw)
    return raw


def write_compressed(path, root: Dict[str, Any], gzip_format: bool = True) -> None:
    """Write NBT back. `mtime=0` keeps the output reproducible across runs."""
    body = dumps(root)
    packed = gzip.compress(body, mtime=0) if gzip_format else zlib.compress(body)
    open(path, "wb").write(packed)
