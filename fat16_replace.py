#!/usr/bin/env python3
"""In-place FAT16 file replacement with a sector write-ahead journal.

The tool operates on a whole-disk FAT16 image (no partition table), 512-byte
sectors and two FATs.  Only a short 8.3 path is used for addressing; existing
LFN directory records are skipped and preserved byte for byte.

The journal is a separate side-car file (``<image>.journal``) made of 512-byte
records.  It stores both the old and new image sector bytes.  No image sector is
written until the complete journal has a checksummed commit record.  Recovery
therefore either redoes all new sectors or restores/keeps the old image.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import zlib
from dataclasses import dataclass

SECTOR_SIZE = 512
SECTOR_MASK = SECTOR_SIZE - 1
EOC_MIN = 0xFFF8
BAD_CLUSTER = 0xFFF7
LFN_ATTR = 0x0F
VOLUME_ATTR = 0x08
DIRECTORY_ATTR = 0x10
VALID_ATTR_MASK = 0x3F

JOURNAL_MAGIC = b"FAT16JNL"
RECORD_MAGIC = b"FAT16REC"
JOURNAL_VERSION = 1

KIND_DATA = 1
KIND_FAT0 = 2
KIND_FAT1 = 4
KIND_DIRECTORY = 8
KNOWN_KINDS = {KIND_DATA, KIND_FAT0, KIND_FAT1, KIND_DIRECTORY}
KIND_NAMES = {
    KIND_DATA: "data",
    KIND_FAT0: "fat0",
    KIND_FAT1: "fat1",
    KIND_DIRECTORY: "directory",
}


class FatError(Exception):
    exit_code = 1


class InsufficientSpace(FatError):
    exit_code = 2


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def pread_exact(fd: int, offset: int, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = os.pread(fd, remaining, offset)
        if not chunk:
            raise FatError(f"short read at sector offset {offset}")
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def pread_sector_best_effort(fd: int, sector: int) -> bytes:
    data = os.pread(fd, SECTOR_SIZE, sector * SECTOR_SIZE)
    if len(data) < SECTOR_SIZE:
        data = data + b"\0" * (SECTOR_SIZE - len(data))
    return data


def pwrite_all(fd: int, offset: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        if written <= 0:
            raise FatError(f"short write at sector offset {offset}")
        offset += written
        view = view[written:]


def fsync_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    parent_fd = os.open(parent, flags)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


class ImageReader:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        self.size = os.fstat(self.f.fileno()).st_size

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> "ImageReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def read(self, offset: int, length: int) -> bytes:
        self.f.seek(offset)
        data = self.f.read(length)
        if len(data) != length:
            raise FatError(
                f"image is too small: wanted {length} bytes at offset {offset}"
            )
        return data

    def read_sector(self, sector: int) -> bytes:
        return self.read(sector * SECTOR_SIZE, SECTOR_SIZE)


@dataclass(frozen=True)
class Layout:
    bytes_per_sector: int
    sectors_per_cluster: int
    reserved_sectors: int
    fat_count: int
    fat_size_sectors: int
    root_entries: int
    root_dir_sectors: int
    media: int
    total_sectors: int
    cluster_count: int

    @property
    def max_cluster(self) -> int:
        return self.cluster_count + 1

    @property
    def fat0_lba(self) -> int:
        return self.reserved_sectors

    @property
    def fat1_lba(self) -> int:
        return self.reserved_sectors + self.fat_size_sectors

    @property
    def root_lba(self) -> int:
        return self.reserved_sectors + self.fat_count * self.fat_size_sectors

    @property
    def data_lba(self) -> int:
        return self.root_lba + self.root_dir_sectors

    @property
    def bytes_per_cluster(self) -> int:
        return self.sectors_per_cluster * SECTOR_SIZE

    def cluster_lba(self, cluster: int) -> int:
        return self.data_lba + (cluster - 2) * self.sectors_per_cluster


def parse_and_validate_layout(image: ImageReader) -> tuple[Layout, bytes, bytes]:
    if image.size < SECTOR_SIZE:
        raise FatError("image is smaller than one sector")
    boot = image.read_sector(0)
    if boot[510:512] != b"\x55\xaa":
        raise FatError("missing boot sector signature")
    if boot[446:510] != b"\0" * 64:
        raise FatError("partition-table entries are present; whole-disk images only")

    bytes_per_sector = u16(boot, 11)
    sectors_per_cluster = boot[13]
    reserved = u16(boot, 14)
    fat_count = boot[16]
    root_entries = u16(boot, 17)
    total_small = u16(boot, 19)
    media = boot[21]
    fat_size = u16(boot, 22)
    total_large = u32(boot, 32)
    total_sectors = total_small or total_large

    if bytes_per_sector != SECTOR_SIZE:
        raise FatError("only 512-byte sectors are supported")
    if sectors_per_cluster == 0 or sectors_per_cluster & (sectors_per_cluster - 1):
        raise FatError("sectors per cluster must be a non-zero power of two")
    if sectors_per_cluster > 128:
        raise FatError("unsupported sectors-per-cluster value")
    if reserved < 1:
        raise FatError("reserved sector count must be non-zero")
    if fat_count != 2:
        raise FatError("this tool requires exactly two FATs")
    if root_entries == 0 or root_entries % 16:
        raise FatError("root entry count must be a non-zero multiple of 16")
    if fat_size == 0:
        raise FatError("FAT size is zero")
    if media not in {0xF0, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF}:
        raise FatError(f"unsupported media descriptor 0x{media:02x}")
    if total_sectors == 0:
        raise FatError("total sector count is zero")

    root_dir_sectors = (root_entries * 32 + SECTOR_MASK) // SECTOR_SIZE
    data_lba = reserved + fat_count * fat_size + root_dir_sectors
    if data_lba >= total_sectors:
        raise FatError("FAT/root regions extend beyond the volume")

    data_sectors = total_sectors - data_lba
    if data_sectors % sectors_per_cluster:
        raise FatError("data region does not end on a cluster boundary")
    cluster_count = data_sectors // sectors_per_cluster
    if not 4085 <= cluster_count <= 65524:
        raise FatError(
            f"volume has {cluster_count} clusters; it is not a FAT16 volume"
        )

    expected_size = total_sectors * SECTOR_SIZE
    if image.size != expected_size:
        raise FatError(
            f"image size {image.size} does not equal volume size {expected_size}"
        )

    fat_word_count = fat_size * (SECTOR_SIZE // 2)
    needed_words = cluster_count + 2
    if fat_word_count < needed_words:
        raise FatError("FAT sectors are too small for cluster count")

    layout = Layout(
        bytes_per_sector=SECTOR_SIZE,
        sectors_per_cluster=sectors_per_cluster,
        reserved_sectors=reserved,
        fat_count=fat_count,
        fat_size_sectors=fat_size,
        root_entries=root_entries,
        root_dir_sectors=root_dir_sectors,
        media=media,
        total_sectors=total_sectors,
        cluster_count=cluster_count,
    )

    fat0 = image.read(layout.fat0_lba * SECTOR_SIZE, fat_size * SECTOR_SIZE)
    fat1 = image.read(layout.fat1_lba * SECTOR_SIZE, fat_size * SECTOR_SIZE)
    if fat0 != fat1:
        raise FatError("FAT copies are not identical")

    words0 = list(struct.unpack(f"<{fat_word_count}H", fat0))
    if words0[0] != 0xFF00 | media:
        raise FatError(
            f"FAT[0] is 0x{words0[0]:04x}, expected 0x{0xFF00 | media:04x}"
        )
    if words0[1] < EOC_MIN:
        raise FatError(f"FAT[1] is not an end-of-chain marker: 0x{words0[1]:04x}")
    for cluster in range(layout.max_cluster + 1, fat_word_count):
        if words0[cluster] != 0:
            raise FatError(f"unused FAT entry {cluster} is non-zero")

    return layout, fat0, fat1


def normalize_fat_path(path_text: str) -> list[str]:
    text = path_text.replace("\\", "/")
    while text.startswith("/"):
        text = text[1:]
    while text.endswith("/") and len(text) > 1:
        text = text[:-1]
    if not text or "//" in text:
        raise FatError("empty path component")

    result = []
    for component in text.split("/"):
        if component in (".", "..") or not component:
            raise FatError(f"invalid path component {component!r}")
        pieces = component.split(".")
        if len(pieces) > 2 or len(pieces[0]) > 8:
            raise FatError(f"{component!r} is not an 8.3 short file name")
        if len(pieces) == 2 and (len(pieces[1]) > 3 or not pieces[1]):
            raise FatError(f"{component!r} is not an 8.3 short file name")
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in component):
            raise FatError(f"invalid character in path component {component!r}")
        result.append(component.upper())
    return result


def decode_short_name(raw11: bytes) -> str:
    base = raw11[:8].rstrip(b" \x00")
    extension = raw11[8:11].rstrip(b" \x00")
    if not base:
        raise FatError("directory entry has an empty short name")
    if any(0x00 < ch < 0x20 for ch in raw11):
        raise FatError("directory entry name contains a control character")
    if extension:
        return (base + b"." + extension).decode("cp437")
    return base.decode("cp437")


@dataclass
class Entry:
    path: list[str]
    name: str
    attr: int
    first_cluster: int
    size: int
    sector: int
    offset: int
    chain: list[int]


def claim_chain(
    start: int,
    fat: list[int],
    max_cluster: int,
    owners: dict[int, str],
    where: str,
) -> list[int]:
    if not 2 <= start <= max_cluster:
        raise FatError(f"{where}: first cluster {start} is out of bounds")

    chain = []
    current = start
    while True:
        if current in owners:
            raise FatError(
                f"cluster {current} is cross-linked or part of a cycle "
                f"({owners[current]} and {where})"
            )
        owners[current] = where
        chain.append(current)

        nxt = fat[current]
        if EOC_MIN <= nxt <= 0xFFFF:
            return chain
        if nxt == 0:
            raise FatError(f"{where}: chain ends in a free FAT entry")
        if nxt == 1:
            raise FatError(f"{where}: chain contains reserved FAT entry 1")
        if nxt == BAD_CLUSTER:
            raise FatError(f"{where}: chain contains bad cluster {current}")
        if not 2 <= nxt <= max_cluster:
            raise FatError(
                f"{where}: cluster {current} points outside the data region "
                f"(0x{nxt:04x})"
            )
        current = nxt


def scan_directory_region(
    data: bytes,
    sector_map: list[int],
    parent_path: list[str],
    layout: Layout,
    fat: list[int],
    target_parts: list[str],
    owners: dict[int, str],
    directory_queue: list[Entry],
) -> Entry | None:
    target: Entry | None = None
    seen_names: set[str] = set()

    for entry_offset in range(0, len(data), 32):
        record = data[entry_offset : entry_offset + 32]
        if record[0] == 0x00:
            break
        if record[0] == 0xE5:
            continue

        attr = record[11]
        if attr == LFN_ATTR:
            # LFN records are not used for addressing and need no interpretation.
            continue
        if attr & ~VALID_ATTR_MASK:
            raise FatError(
                f"directory entry at sector {sector_map[entry_offset // 512]} "
                "has reserved attribute bits"
            )

        if attr & VOLUME_ATTR:
            if attr != VOLUME_ATTR:
                raise FatError("volume label attribute is combined with other bits")
            if u16(record, 20) or u16(record, 26) or u32(record, 28):
                raise FatError("volume label entry contains cluster or size")
            continue

        name = decode_short_name(record[:11])
        upper_name = name.upper()

        # FAT directory clusters begin with self/parent aliases.  They are not
        # independent entries or traversal edges.
        if upper_name in (".", ".."):
            continue

        if upper_name in seen_names:
            raise FatError(f"duplicate short name {name!r} in one directory")
        seen_names.add(upper_name)

        entry_path = parent_path + [name]
        where = "/" + "/".join(entry_path)
        high_cluster = u16(record, 20)
        if high_cluster:
            raise FatError(f"{where}: FAT16 entry contains a high cluster number")
        first_cluster = u16(record, 26)
        size = u32(record, 28)
        sector_index = entry_offset // 512
        sector = sector_map[sector_index]
        offset_in_sector = entry_offset % 512

        is_directory = bool(attr & DIRECTORY_ATTR)
        if is_directory:
            if size != 0:
                raise FatError(f"{where}: directory has non-zero size {size}")
            if first_cluster == 0:
                raise FatError(f"{where}: subdirectory has no cluster")
            chain = claim_chain(
                first_cluster, fat, layout.max_cluster, owners, where
            )
            entry = Entry(
                entry_path, name, attr, first_cluster, size,
                sector, offset_in_sector, chain,
            )
            directory_queue.append(entry)
            if [p.upper() for p in entry_path] == target_parts:
                target = entry
            continue

        if size == 0:
            if first_cluster != 0:
                raise FatError(f"{where}: zero-length file owns a cluster")
            chain: list[int] = []
        else:
            if first_cluster == 0:
                raise FatError(f"{where}: non-empty file has no first cluster")
            chain = claim_chain(
                first_cluster, fat, layout.max_cluster, owners, where
            )
            capacity = len(chain) * layout.bytes_per_cluster
            if size > capacity or size <= capacity - layout.bytes_per_cluster:
                raise FatError(f"{where}: file size does not match its chain")

        entry = Entry(
            entry_path, name, attr, first_cluster, size,
            sector, offset_in_sector, chain,
        )
        if [p.upper() for p in entry_path] == target_parts:
            target = entry

    return target


def find_file_and_validate_chains(
    image: ImageReader,
    layout: Layout,
    fat0: bytes,
    target_parts: list[str],
) -> Entry:
    fat = list(struct.unpack(f"<{layout.fat_size_sectors * 256}H", fat0))
    owners: dict[int, str] = {}
    directory_queue: list[Entry] = []

    root_data = image.read(
        layout.root_lba * SECTOR_SIZE,
        layout.root_dir_sectors * SECTOR_SIZE,
    )
    root_sectors = list(
        range(layout.root_lba, layout.root_lba + layout.root_dir_sectors)
    )
    target = scan_directory_region(
        root_data, root_sectors, [], layout, fat, target_parts,
        owners, directory_queue,
    )

    while directory_queue:
        directory = directory_queue.pop(0)
        directory_data = bytearray()
        directory_sectors: list[int] = []
        for cluster in directory.chain:
            for sector_offset in range(layout.sectors_per_cluster):
                sector = layout.cluster_lba(cluster) + sector_offset
                directory_data.extend(image.read_sector(sector))
                directory_sectors.append(sector)
        found = scan_directory_region(
            bytes(directory_data), directory_sectors, directory.path,
            layout, fat, target_parts, owners, directory_queue,
        )
        if found is not None:
            target = found

    for cluster in range(2, layout.max_cluster + 1):
        value = fat[cluster]
        if value == 0 or value == BAD_CLUSTER:
            continue
        if cluster not in owners:
            raise FatError(f"cluster {cluster} is allocated but not reachable")

    if target is None:
        raise FatError("/" + "/".join(target_parts) + " was not found")
    if target.attr & DIRECTORY_ATTR:
        raise FatError("/" + "/".join(target_parts) + " is a directory")
    return target


@dataclass(frozen=True)
class SectorChange:
    sector: int
    old: bytes
    new: bytes
    kind: int


@dataclass(frozen=True)
class ReplacementPlan:
    target_path: str
    size_before: int
    size_after: int
    chain_before: tuple[int, ...]
    chain_after: tuple[int, ...]
    changes: tuple[SectorChange, ...]


def build_replacement_plan(
    image: ImageReader,
    layout: Layout,
    fat0: bytes,
    target: Entry,
    new_content: bytes,
) -> ReplacementPlan:
    if len(new_content) > 0xFFFFFFFF:
        raise FatError("FAT16 files cannot be larger than 4 GiB-1")

    fat = list(struct.unpack(f"<{layout.fat_size_sectors * 256}H", fat0))
    old_chain = target.chain
    cluster_bytes = layout.bytes_per_cluster
    clusters_needed = (
        (len(new_content) + cluster_bytes - 1) // cluster_bytes
        if new_content
        else 0
    )

    if clusters_needed > layout.cluster_count:
        raise InsufficientSpace("new content is larger than the whole data region")

    if clusters_needed <= len(old_chain):
        new_chain = old_chain[:clusters_needed]
    else:
        old_set = set(old_chain)
        free_clusters = [
            cluster
            for cluster in range(2, layout.max_cluster + 1)
            if fat[cluster] == 0 and cluster not in old_set
        ]
        additional = clusters_needed - len(old_chain)
        if len(free_clusters) < additional:
            raise InsufficientSpace(
                f"need {additional} clusters, but only {len(free_clusters)} are free"
            )
        new_chain = old_chain + free_clusters[:additional]

    changes: list[SectorChange] = []

    # Data clusters.  The final cluster is padded to zeroes through its end.
    padded_content = new_content + b"\0" * (
        len(new_chain) * cluster_bytes - len(new_content)
    )
    for logical_cluster, cluster in enumerate(new_chain):
        for sector_offset in range(layout.sectors_per_cluster):
            sector = layout.cluster_lba(cluster) + sector_offset
            buffer_offset = (
                logical_cluster * layout.sectors_per_cluster + sector_offset
            ) * SECTOR_SIZE
            new_sector = padded_content[
                buffer_offset : buffer_offset + SECTOR_SIZE
            ].ljust(SECTOR_SIZE, b"\0")
            old_sector = image.read_sector(sector)
            if old_sector != new_sector:
                changes.append(SectorChange(sector, old_sector, new_sector, KIND_DATA))

    # FAT metadata, including links, EOC and freed tail clusters.
    new_fat = bytearray(fat0)

    def put_fat(cluster: int, value: int) -> None:
        struct.pack_into("<H", new_fat, cluster * 2, value)

    for cluster in old_chain[len(new_chain) :]:
        put_fat(cluster, 0)
    for index, cluster in enumerate(new_chain):
        if index + 1 < len(new_chain):
            put_fat(cluster, new_chain[index + 1])
        else:
            put_fat(cluster, 0xFFFF)

    for fat_sector in range(layout.fat_size_sectors):
        start = fat_sector * SECTOR_SIZE
        end = start + SECTOR_SIZE
        old_fat_sector = fat0[start:end]
        new_fat_sector = bytes(new_fat[start:end])
        if old_fat_sector != new_fat_sector:
            changes.append(
                SectorChange(
                    layout.fat0_lba + fat_sector,
                    old_fat_sector,
                    new_fat_sector,
                    KIND_FAT0,
                )
            )
            changes.append(
                SectorChange(
                    layout.fat1_lba + fat_sector,
                    old_fat_sector,  # FAT copies were proved identical.
                    new_fat_sector,
                    KIND_FAT1,
                )
            )

    # Directory entry: only first cluster and size can change.
    directory_sector_number = target.sector
    directory_sector = bytearray(image.read_sector(directory_sector_number))
    entry_offset = target.offset
    struct.pack_into(
        "<H",
        directory_sector,
        entry_offset + 26,
        new_chain[0] if new_chain else 0,
    )
    struct.pack_into("<I", directory_sector, entry_offset + 28, len(new_content))
    old_directory_sector = image.read_sector(directory_sector_number)
    if bytes(directory_sector) != old_directory_sector:
        changes.append(
            SectorChange(
                directory_sector_number,
                old_directory_sector,
                bytes(directory_sector),
                KIND_DIRECTORY,
            )
        )

    changes.sort(key=lambda change: change.sector)
    return ReplacementPlan(
        target_path="/" + "/".join(target.path),
        size_before=target.size,
        size_after=len(new_content),
        chain_before=tuple(old_chain),
        chain_after=tuple(new_chain),
        changes=tuple(changes),
    )


def make_superblock(record_count: int, state: int, image_size: int) -> bytes:
    sector = bytearray(SECTOR_SIZE)
    struct.pack_into(
        "<8sHHIIQ",
        sector,
        0,
        JOURNAL_MAGIC,
        JOURNAL_VERSION,
        SECTOR_SIZE,
        record_count,
        state,
        image_size,
    )
    struct.pack_into("<I", sector, 508, crc32(sector[:508]))
    return bytes(sector)


def parse_superblock(sector: bytes) -> dict[str, int] | None:
    if len(sector) != SECTOR_SIZE or sector[:8] != JOURNAL_MAGIC:
        return None
    if crc32(sector[:508]) != u32(sector, 508):
        return None
    _magic, version, sector_size, record_count, state, image_size = struct.unpack_from(
        "<8sHHIIQ", sector, 0
    )
    if (
        version != JOURNAL_VERSION
        or sector_size != SECTOR_SIZE
        or state not in (0, 1)
    ):
        return None
    return {
        "record_count": record_count,
        "state": state,
        "image_size": image_size,
    }


def make_descriptor(index: int, change: SectorChange) -> bytes:
    sector = bytearray(SECTOR_SIZE)
    struct.pack_into(
        "<8sIIQII",
        sector,
        0,
        RECORD_MAGIC,
        index,
        change.kind,
        change.sector,
        crc32(change.old),
        crc32(change.new),
    )
    struct.pack_into("<I", sector, 508, crc32(sector[:508]))
    return bytes(sector)


def parse_descriptor(sector: bytes) -> dict[str, int] | None:
    if len(sector) != SECTOR_SIZE or sector[:8] != RECORD_MAGIC:
        return None
    if crc32(sector[:508]) != u32(sector, 508):
        return None
    _magic, index, kind, image_sector, old_crc, new_crc = struct.unpack_from(
        "<8sIIQII", sector, 0
    )
    if kind not in KNOWN_KINDS:
        return None
    return {
        "index": index,
        "kind": kind,
        "sector": image_sector,
        "old_crc": old_crc,
        "new_crc": new_crc,
    }


def durable_sector_write(
    fd: int,
    sector: int,
    payload: bytes,
    counter: list[int],
) -> None:
    if len(payload) != SECTOR_SIZE:
        raise FatError("journal/image writes must be exactly one sector")

    counter[0] += 1
    fault_at = int(os.environ.get("FAT16_FAULT_AFTER", "0") or "0")
    if fault_at and counter[0] == fault_at:
        # Simulate a physically torn 512-byte sector write: the first half is
        # made durable, then the process is killed before the remainder arrives.
        if os.environ.get("FAT16_TORN") == "1":
            pwrite_all(fd, sector * SECTOR_SIZE, payload[: SECTOR_SIZE // 2])
            os.fsync(fd)
        os._exit(3)

    pwrite_all(fd, sector * SECTOR_SIZE, payload)
    os.fsync(fd)


def execute_plan(image_path: str, plan: ReplacementPlan) -> dict[str, object]:
    image_size = os.path.getsize(image_path)
    journal_path = image_path + ".journal"
    record_count = len(plan.changes)
    counter = [0]

    image_fd = os.open(image_path, os.O_RDWR)
    journal_fd = os.open(
        journal_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        durable_sector_write(
            journal_fd, 0, make_superblock(record_count, 0, image_size), counter
        )
        for index, change in enumerate(plan.changes):
            base = 1 + index * 3
            durable_sector_write(journal_fd, base, make_descriptor(index, change), counter)
            durable_sector_write(journal_fd, base + 1, change.old, counter)
            durable_sector_write(journal_fd, base + 2, change.new, counter)

        durable_sector_write(
            journal_fd, 0, make_superblock(record_count, 1, image_size), counter
        )

        # Only after this durable commit is it legal to touch the image.
        for change in plan.changes:
            durable_sector_write(image_fd, change.sector, change.new, counter)
        os.fsync(image_fd)
    finally:
        os.close(journal_fd)
        os.close(image_fd)

    os.unlink(journal_path)
    fsync_parent(journal_path)
    return plan_result(plan, counter[0], replaced=True)


def parse_committed_records(
    journal_fd: int,
    record_count: int,
    image_size: int,
) -> list[dict[str, object]]:
    expected_file_sectors = 1 + record_count * 3
    if os.fstat(journal_fd).st_size != expected_file_sectors * SECTOR_SIZE:
        raise FatError("committed journal has an unexpected size")

    records = []
    seen_sectors: set[int] = set()
    for index in range(record_count):
        base = 1 + index * 3
        descriptor_sector = pread_exact(journal_fd, base * SECTOR_SIZE, SECTOR_SIZE)
        old_sector = pread_exact(
            journal_fd, (base + 1) * SECTOR_SIZE, SECTOR_SIZE
        )
        new_sector = pread_exact(
            journal_fd, (base + 2) * SECTOR_SIZE, SECTOR_SIZE
        )
        descriptor = parse_descriptor(descriptor_sector)
        if descriptor is None:
            raise FatError(f"journal record {index} has a corrupt descriptor")
        if descriptor["index"] != index:
            raise FatError(f"journal record {index} has wrong sequence number")
        if descriptor["sector"] in seen_sectors:
            raise FatError("journal contains duplicate image sector")
        seen_sectors.add(descriptor["sector"])
        if descriptor["sector"] * SECTOR_SIZE + SECTOR_SIZE > image_size:
            raise FatError(f"journal record {index} refers outside the image")
        if crc32(old_sector) != descriptor["old_crc"]:
            raise FatError(f"journal record {index} has corrupt old payload")
        if crc32(new_sector) != descriptor["new_crc"]:
            raise FatError(f"journal record {index} has corrupt new payload")
        records.append(
            {
                "sector": descriptor["sector"],
                "kind": descriptor["kind"],
                "old": old_sector,
                "new": new_sector,
            }
        )
    return records


def parse_undo_records(journal_fd: int) -> list[dict[str, object]]:
    """Read sequentially complete undo information from an uncommitted journal."""
    journal_size = os.fstat(journal_fd).st_size
    block_count = max(0, (journal_size // SECTOR_SIZE - 1) // 3)
    records = []
    for index in range(block_count):
        base = 1 + index * 3
        descriptor_sector = pread_sector_best_effort(journal_fd, base)
        descriptor = parse_descriptor(descriptor_sector)
        if descriptor is None or descriptor["index"] != index:
            break
        old_sector = pread_sector_best_effort(journal_fd, base + 1)
        if crc32(old_sector) != descriptor["old_crc"]:
            break
        records.append(
            {
                "sector": descriptor["sector"],
                "kind": descriptor["kind"],
                "old": old_sector,
            }
        )
    return records


def remove_journal(journal_path: str) -> None:
    os.unlink(journal_path)
    fsync_parent(journal_path)


def recover_journal(image_path: str) -> dict[str, object]:
    journal_path = image_path + ".journal"
    if not os.path.exists(journal_path):
        raise FatError("no recovery journal is present")

    image_size = os.path.getsize(image_path)
    with open(journal_path, "rb") as journal_file:
        journal_fd = journal_file.fileno()
        superblock_sector = pread_sector_best_effort(journal_fd, 0)
        superblock = parse_superblock(superblock_sector)

        if superblock is not None and superblock["state"] == 1:
            if superblock["image_size"] != image_size:
                raise FatError("image size does not match committed journal")
            records = parse_committed_records(
                journal_fd, superblock["record_count"], image_size
            )
            committed = True
        else:
            if superblock is not None and superblock["image_size"] != image_size:
                raise FatError("image size does not match aborted journal")
            records = parse_undo_records(journal_fd)
            committed = False

    image_fd = os.open(image_path, os.O_RDWR)
    counter = [0]
    try:
        if committed:
            for record in records:
                durable_sector_write(
                    image_fd, int(record["sector"]), record["new"], counter
                )
            for record in records:
                current = pread_exact(
                    image_fd, int(record["sector"]) * SECTOR_SIZE, SECTOR_SIZE
                )
                if current != record["new"]:
                    raise FatError("redo verification failed")
        else:
            # An uncommitted transaction normally never touched the image.
            # If a sector was nevertheless modified, restore its old copy.
            for record in records:
                sector = int(record["sector"])
                current = pread_exact(image_fd, sector * SECTOR_SIZE, SECTOR_SIZE)
                if current != record["old"]:
                    durable_sector_write(image_fd, sector, record["old"], counter)
                    current = pread_exact(image_fd, sector * SECTOR_SIZE, SECTOR_SIZE)
                    if current != record["old"]:
                        raise FatError("undo verification failed")
        os.fsync(image_fd)
    finally:
        os.close(image_fd)

    remove_journal(journal_path)
    return {
        "recovered_version": "new" if committed else "old",
        "needs_rerun": not committed,
        "journal_records": len(records),
        "sectors": sorted(int(record["sector"]) for record in records),
        "sector_kinds": sectors_by_kind(
            [
                (int(record["sector"]), int(record["kind"]))
                for record in records
            ]
        ),
    }


def sectors_by_kind(kind_pairs: list[tuple[int, int]]) -> dict[str, list[int]]:
    result = {name: [] for name in KIND_NAMES.values()}
    for sector, kind in kind_pairs:
        result[KIND_NAMES[kind]].append(sector)
    for values in result.values():
        values.sort()
    return result


def plan_result(
    plan: ReplacementPlan,
    fault_boundaries: int,
    replaced: bool,
) -> dict[str, object]:
    kind_pairs = [(change.sector, change.kind) for change in plan.changes]
    old_len = len(plan.chain_before)
    new_len = len(plan.chain_after)
    return {
        "visible_version": "new",
        "replaced": replaced,
        "target": plan.target_path,
        "size_before": plan.size_before,
        "size_after": plan.size_after,
        "chain_before": list(plan.chain_before),
        "chain_after": list(plan.chain_after),
        "allocated_clusters": list(plan.chain_after[old_len:]),
        "freed_clusters": list(plan.chain_before[new_len:]),
        "journal_records": len(plan.changes),
        "changed_sectors": sorted(change.sector for change in plan.changes),
        "sector_kinds": sectors_by_kind(kind_pairs),
        "fault_boundaries": fault_boundaries,
    }


def read_input(path: str | None) -> bytes:
    if path is None:
        raise FatError("--input is required")
    if path == "-":
        return sys.stdin.buffer.read()
    with open(path, "rb") as source:
        return source.read()


def command_main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace one existing file in a whole-disk FAT16 image."
    )
    parser.add_argument("--recover", action="store_true", help="recover a journal")
    parser.add_argument("--plan", action="store_true", help="print plan and do not write")
    parser.add_argument("image", help="whole-disk FAT16 image")
    parser.add_argument("fat_path", nargs="?", help="short 8.3 path, e.g. DIR/FILE.TXT")
    parser.add_argument(
        "-i", "--input", help="new file content, or '-' for standard input"
    )
    args = parser.parse_args()

    try:
        if args.recover:
            if args.fat_path is not None or args.input is not None:
                raise FatError("--recover takes only the image path")
            print(json.dumps(recover_journal(args.image), indent=2))
            return 0

        if args.fat_path is None:
            raise FatError("a short file path is required")
        if os.path.exists(args.image + ".journal"):
            # Finish the previously interrupted transaction before accepting a
            # new one. An aborted recovery leaves the old file and asks to rerun.
            print(json.dumps(recover_journal(args.image), indent=2))
            return 0

        new_content = read_input(args.input)
        target_parts = normalize_fat_path(args.fat_path)

        with ImageReader(args.image) as image:
            layout, fat0, _fat1 = parse_and_validate_layout(image)
            target = find_file_and_validate_chains(
                image, layout, fat0, target_parts
            )
            plan = build_replacement_plan(
                image, layout, fat0, target, new_content
            )

        if args.plan:
            print(json.dumps(plan_result(plan, 2 + 4 * len(plan.changes), False), indent=2))
            return 0

        if not plan.changes:
            result = plan_result(plan, 0, False)
            result["replaced"] = False
            print(json.dumps(result, indent=2))
            return 0

        result = execute_plan(args.image, plan)
        print(json.dumps(result, indent=2))
        return 0

    except InsufficientSpace as exc:
        print(json.dumps({"error": str(exc), "visible_version": "old"}, indent=2))
        return exc.exit_code
    except (FatError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return getattr(exc, "exit_code", 1)


if __name__ == "__main__":
    sys.exit(command_main())
