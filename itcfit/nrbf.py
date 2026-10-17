"""Minimal reader for the .NET Remoting Binary Format (MS-NRBF).

TA Instruments ITCRun stores ``.nitc`` files as a gzip-compressed
``BinaryFormatter`` stream of a ``System.Data.DataSet``.  This module decodes
the NRBF record stream into plain Python objects:

* classes      -> :class:`NetObject` (``.class_name`` + ``.members`` dict)
* arrays       -> ``list`` (primitive arrays of doubles -> ``numpy.ndarray``)
* strings      -> ``str``
* primitives   -> ``bool`` / ``int`` / ``float`` / ``str``

Only reading is supported; the spec is at
https://learn.microsoft.com/openspecs/windows_protocols/ms-nrbf
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# PrimitiveTypeEnumeration -> (struct format, size)
_PRIM = {
    1: ("<?", 1),   # Boolean
    2: ("<B", 1),   # Byte
    6: ("<d", 8),   # Double
    7: ("<h", 2),   # Int16
    8: ("<i", 4),   # Int32
    9: ("<q", 8),   # Int64
    10: ("<b", 1),  # SByte
    11: ("<f", 4),  # Single
    12: ("<q", 8),  # TimeSpan (ticks)
    13: ("<Q", 8),  # DateTime (ticks + kind bits)
    14: ("<H", 2),  # UInt16
    15: ("<I", 4),  # UInt32
    16: ("<Q", 8),  # UInt64
}
_NUMPY_PRIM = {2: "<u1", 6: "<f8", 7: "<i2", 8: "<i4", 9: "<i8", 10: "<i1",
               11: "<f4", 14: "<u2", 15: "<u4", 16: "<u8", 1: "?"}


@dataclass
class NetObject:
    class_name: str
    members: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.members[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.members.get(key, default)

    def __repr__(self) -> str:
        return f"NetObject({self.class_name!r}, {len(self.members)} members)"


class _Ref:
    __slots__ = ("id",)

    def __init__(self, ref_id: int):
        self.id = ref_id


class _Nulls:
    __slots__ = ("count",)

    def __init__(self, count: int):
        self.count = count


@dataclass
class _ClassMeta:
    name: str
    member_names: list[str]
    binary_types: list[int] | None
    additional: list[Any] | None


class NRBFReader:
    def __init__(self, data: bytes):
        self.buf = memoryview(data)
        self.pos = 0
        self.objects: dict[int, Any] = {}
        self.metas: dict[int, _ClassMeta] = {}
        self.root_id: int | None = None

    # -- low level ---------------------------------------------------------
    def _read(self, n: int) -> bytes:
        b = self.buf[self.pos:self.pos + n]
        if len(b) != n:
            raise EOFError("unexpected end of NRBF stream")
        self.pos += n
        return bytes(b)

    def _u8(self) -> int:
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def _i32(self) -> int:
        return struct.unpack("<i", self._read(4))[0]

    def _lps(self) -> str:
        length, shift = 0, 0
        while True:
            b = self._u8()
            length |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        return self._read(length).decode("utf-8", errors="replace")

    def _primitive(self, ptype: int) -> Any:
        if ptype == 18:   # String
            return self._lps()
        if ptype == 5:    # Decimal (as string)
            return float(self._lps())
        if ptype == 3:    # Char (UTF-8, 1-4 bytes)
            first = self.buf[self.pos]
            n = 1 if first < 0x80 else 2 if first < 0xE0 else 3 if first < 0xF0 else 4
            return self._read(n).decode("utf-8", errors="replace")
        if ptype == 17:   # Null
            return None
        fmt, size = _PRIM[ptype]
        return struct.unpack(fmt, self._read(size))[0]

    # -- metadata ----------------------------------------------------------
    def _class_info(self) -> tuple[int, str, list[str]]:
        obj_id = self._i32()
        name = self._lps()
        count = self._i32()
        return obj_id, name, [self._lps() for _ in range(count)]

    def _member_type_info(self, count: int) -> tuple[list[int], list[Any]]:
        btypes = [self._u8() for _ in range(count)]
        extra: list[Any] = []
        for bt in btypes:
            if bt in (0, 7):        # Primitive / PrimitiveArray
                extra.append(self._u8())
            elif bt == 3:           # SystemClass
                extra.append(self._lps())
            elif bt == 4:           # Class
                extra.append((self._lps(), self._i32()))
            else:                   # String, Object, ObjectArray, StringArray
                extra.append(None)
        return btypes, extra

    # -- records -----------------------------------------------------------
    def _class_values(self, obj_id: int, meta: _ClassMeta) -> NetObject:
        obj = NetObject(meta.name)
        self.objects[obj_id] = obj
        for i, name in enumerate(meta.member_names):
            if meta.binary_types is not None and meta.binary_types[i] == 0:
                obj.members[name] = self._primitive(meta.additional[i])
            else:
                obj.members[name] = self._record()
        return obj

    def _elements(self, n: int, ptype: int | None = None) -> list[Any]:
        out: list[Any] = []
        while len(out) < n:
            if ptype is not None:
                out.append(self._primitive(ptype))
                continue
            v = self._record()
            if isinstance(v, _Nulls):
                out.extend([None] * v.count)
            else:
                out.append(v)
        return out

    def _primitive_array(self, n: int, ptype: int) -> Any:
        if ptype in _NUMPY_PRIM:
            dt = np.dtype(_NUMPY_PRIM[ptype])
            arr = np.frombuffer(self._read(n * dt.itemsize), dtype=dt).copy()
            if ptype == 2:
                return arr.tobytes()   # byte[] -> bytes (often nested streams)
            return arr
        return [self._primitive(ptype) for _ in range(n)]

    def _record(self) -> Any:
        rt = self._u8()
        if rt == 0:     # SerializedStreamHeader
            self.root_id = self._i32()
            self._read(12)
            return self._record()
        if rt == 1:     # ClassWithId
            obj_id = self._i32()
            meta = self.metas[self._i32()]
            return self._class_values(obj_id, meta)
        if rt in (2, 3):  # (System)ClassWithMembers (no type info)
            obj_id, name, names = self._class_info()
            if rt == 3:
                self._i32()
            meta = _ClassMeta(name, names, None, None)
            self.metas[obj_id] = meta
            return self._class_values(obj_id, meta)
        if rt in (4, 5):  # (System)ClassWithMembersAndTypes
            obj_id, name, names = self._class_info()
            bt, extra = self._member_type_info(len(names))
            if rt == 5:
                self._i32()
            meta = _ClassMeta(name, names, bt, extra)
            self.metas[obj_id] = meta
            return self._class_values(obj_id, meta)
        if rt == 6:     # BinaryObjectString
            obj_id = self._i32()
            s = self._lps()
            self.objects[obj_id] = s
            return s
        if rt == 7:     # BinaryArray
            obj_id = self._i32()
            arr_type = self._u8()
            rank = self._i32()
            lengths = [self._i32() for _ in range(rank)]
            if arr_type in (3, 4, 5):
                [self._i32() for _ in range(rank)]
            bt, extra = self._member_type_info(1)
            n = int(np.prod(lengths)) if lengths else 0
            if bt[0] == 0:
                val = self._primitive_array(n, extra[0])
            else:
                val = self._elements(n)
            self.objects[obj_id] = val
            return val
        if rt == 8:     # MemberPrimitiveTyped
            return self._primitive(self._u8())
        if rt == 9:     # MemberReference
            return _Ref(self._i32())
        if rt == 10:    # ObjectNull
            return None
        if rt == 11:    # MessageEnd
            raise StopIteration
        if rt == 12:    # BinaryLibrary
            self._i32()
            self._lps()
            return self._record()
        if rt == 13:    # ObjectNullMultiple256
            return _Nulls(self._u8())
        if rt == 14:    # ObjectNullMultiple
            return _Nulls(self._i32())
        if rt == 15:    # ArraySinglePrimitive
            obj_id = self._i32()
            n = self._i32()
            val = self._primitive_array(n, self._u8())
            self.objects[obj_id] = val
            return val
        if rt in (16, 17):  # ArraySingleObject / ArraySingleString
            obj_id = self._i32()
            val = self._elements(self._i32())
            self.objects[obj_id] = val
            return val
        raise ValueError(f"unsupported NRBF record type {rt} at offset {self.pos - 1}")

    # -- reference resolution ---------------------------------------------
    def _resolve(self, v: Any, seen: set[int]) -> Any:
        if isinstance(v, _Ref):
            target = self.objects.get(v.id)
            return self._resolve(target, seen) if not isinstance(target, _Ref) else None
        if isinstance(v, NetObject):
            if id(v) in seen:
                return v
            seen.add(id(v))
            for k, m in v.members.items():
                v.members[k] = self._resolve(m, seen)
        elif isinstance(v, list):
            if id(v) in seen:
                return v
            seen.add(id(v))
            for i, m in enumerate(v):
                v[i] = self._resolve(m, seen)
        return v

    def parse(self) -> Any:
        try:
            while True:
                self._record()
        except StopIteration:
            pass
        seen: set[int] = set()
        for k in list(self.objects):
            self.objects[k] = self._resolve(self.objects[k], seen)
        return self.objects.get(self.root_id)


def loads(data: bytes) -> Any:
    """Deserialize an NRBF byte stream and return its root object."""
    return NRBFReader(data).parse()
