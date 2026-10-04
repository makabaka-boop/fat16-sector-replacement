"""Minimal independent FAT16 image reader used by the test harness."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

SECTOR = 512
EOC_MIN = 0xFFF8
BAD_CLUSTER = 0xFFF7
LFN_ATTR = 0x0F
VOLUME_ATTR = 0x08
DIRECTORY_ATTR = 0x10


def u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


@dataclass(frozen=True)
class Entry:
    name: str
    attr: int
    first_cluster: int
    size: int
    directory: bool
    sector: int
    offset: int
    chain: tuple[int, ...]


@dataclass(frozen=True)
class Layout:
    reserved: int
    fat_size: int
    root_entries: int
    root_sectors: int
    data_lba: int
    sectors_per_cluster: int
    max_cluster: int


class IndependentFat16:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.bytes = self.path.read_bytes()
        boot = self.bytes[:SECTOR]
        if boot[510:512] != b"\x55\xaa":
            raise AssertionError("boot signature missing")
        self.sectors_per_cluster = boot[13]
        self.reserved = u16(boot, 14)
        self.fat_count = boot[16]
        assert self.fat_count == 2
        self.root_entries = u16(boot, 17)
        total_sectors = u16(boot, 19) or u32(boot, 32)
        self.fat_size = u16(boot, 22)
        self.root_sectors = (self.root_entries * 32 + SECTOR - 1) // SECTOR
        self.data_lba = self.reserved + 2 * self.fat_size + self.root_sectors
        data_sectors = total_sectors - self.data_lba
        self.max_cluster = data_sectors // self.sectors_per_cluster + 1
        self.layout = Layout(
            self.reserved,
            self.fat_size,
            self.root_entries,
            self.root_sectors,
            self.data_lba,
            self.sectors_per_cluster,
            self.max_cluster,
        )

    @property
    def fats(self) -> tuple[bytes, bytes]:
        start0 = self.reserved * SECTOR
        start1 = start0 + self.fat_size * SECTOR
        return (
            self.bytes[start0 : start0 + self.fat_size * SECTOR],
            self.bytes[start1 : start1 + self.fat_size * SECTOR],
        )

    def fat_entry(self, which: int, cluster: int) -> int:
        start = (self.reserved + which * self.fat_size) * SECTOR
        return u16(self.bytes, start + cluster * 2)

    def sector(self, number: int) -> bytes:
        return self.bytes[number * SECTOR : (number + 1) * SECTOR]

    def cluster_sector(self, cluster: int, offset: int = 0) -> int:
        return self.data_lba + (cluster - 2) * self.sectors_per_cluster + offset

    def entries(self, region: bytes, sectors: list[int], path: tuple[str, ...]):
        for entry_offset in range(0, len(region), 32):
            record = region[entry_offset : entry_offset + 32]
            if record[0] == 0:
                break
            if record[0] == 0xE5 or record[11] == LFN_ATTR:
                continue
            attr = record[11]
            if attr & VOLUME_ATTR:
                continue
            raw_name = record[:11]
            base = raw_name[:8].rstrip(b" ").decode("cp437")
            extension = raw_name[8:11].rstrip(b" ").decode("cp437")
            name = base + ("." + extension if extension else "")
            if name in (".", ".."):
                continue
            first_cluster = u16(record, 26)
            size = u32(record, 28)
            yield Entry(
                name,
                attr,
                first_cluster,
                size,
                bool(attr & DIRECTORY_ATTR),
                sectors[entry_offset // SECTOR],
                entry_offset % SECTOR,
                (),
            ), path + (name,)

    def walk(self):
        root_start = (self.reserved + 2 * self.fat_size) * SECTOR
        root_data = self.bytes[
            root_start : root_start + self.root_sectors * SECTOR
        ]
        root_sectors = list(
            range(
                self.reserved + 2 * self.fat_size,
                self.reserved + 2 * self.fat_size + self.root_sectors,
            )
        )
        queue = [(root_data, root_sectors, ())]
        while queue:
            data, sectors, path = queue.pop(0)
            for entry, entry_path in self.entries(data, sectors, path):
                if entry.directory:
                    dir_chain = tuple(self.chain(entry.first_cluster))
                    dir_entry = Entry(
                        entry.name,
                        entry.attr,
                        entry.first_cluster,
                        entry.size,
                        True,
                        entry.sector,
                        entry.offset,
                        dir_chain,
                    )
                    directory_data = bytearray()
                    directory_sectors = []
                    for cluster in dir_chain:
                        for off in range(self.sectors_per_cluster):
                            sector = self.cluster_sector(cluster, off)
                            directory_data.extend(self.sector(sector))
                            directory_sectors.append(sector)
                    queue.append((bytes(directory_data), directory_sectors, entry_path))
                    yield dir_entry, entry_path
                else:
                    chain = tuple(self.chain(entry.first_cluster)) if entry.size else ()
                    yield Entry(
                        entry.name,
                        entry.attr,
                        entry.first_cluster,
                        entry.size,
                        False,
                        entry.sector,
                        entry.offset,
                        chain,
                    ), entry_path

    def chain(self, start: int) -> list[int]:
        if start == 0:
            return []
        result = []
        seen = set()
        current = start
        while True:
            assert 2 <= current <= self.max_cluster, f"bad cluster {current}"
            assert current not in seen, f"cycle at {current}"
            seen.add(current)
            result.append(current)
            nxt = self.fat_entry(0, current)
            if EOC_MIN <= nxt <= 0xFFFF:
                return result
            current = nxt

    def find(self, path: str) -> tuple[Entry, tuple[str, ...]]:
        wanted = tuple(part.upper() for part in path.strip("/").split("/"))
        for entry, entry_path in self.walk():
            if entry_path == wanted:
                return entry, entry_path
        raise KeyError(path)

    def file_bytes(self, entry: Entry) -> bytes:
        result = bytearray()
        for cluster in entry.chain:
            for off in range(self.sectors_per_cluster):
                result.extend(self.sector(self.cluster_sector(cluster, off)))
        return bytes(result[: entry.size])

    def allocation_owners(self) -> dict[int, tuple[Entry, tuple[str, ...]]]:
        owners: dict[int, tuple[Entry, tuple[str, ...]]] = {}
        for item in self.walk():
            entry, path = item
            for cluster in entry.chain:
                assert cluster not in owners, f"cross-link at {cluster}"
                owners[cluster] = item
        return owners
