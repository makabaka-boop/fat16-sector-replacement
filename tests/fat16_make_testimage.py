"""Create deterministic FAT16 images used by the crash-recovery tests.

This fixture writer intentionally lives beside the tests instead of importing
the production tool.  It gives the independent checker known raw layouts.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

SECTOR = 512
SECTORS_PER_CLUSTER = 1
RESERVED = 1
FAT_COUNT = 2
FAT_SIZE = 64
ROOT_ENTRIES = 512
ROOT_SECTORS = 32
DATA_LBA = RESERVED + FAT_COUNT * FAT_SIZE + ROOT_SECTORS
CLUSTERS = 16000
TOTAL_SECTORS = DATA_LBA + CLUSTERS
BAD_CLUSTER = 200


def lfn_checksum(short_name: bytes) -> int:
    value = 0
    for byte in short_name:
        value = ((value & 1) << 7) + (value >> 1) + byte
        value &= 0xFF
    return value


def pack_name(name: str) -> bytes:
    field = bytearray(b" " * 11)
    if name in (".", ".."):
        field[: len(name)] = name.encode("ascii")
        return bytes(field)
    base, dot, extension = name.partition(".")
    field[: len(base)] = base.encode("ascii")
    if dot:
        field[8 : 8 + len(extension)] = extension.encode("ascii")
    return bytes(field)


def short_entry(
    name: str,
    attr: int,
    first_cluster: int,
    size: int,
    *,
    create_time: int = 0x1234,
    modify_time: int = 0x5678,
    modify_date: int = 0x5A29,
) -> bytes:
    entry = bytearray(32)
    entry[:11] = pack_name(name)
    entry[11] = attr
    entry[13] = 0
    struct.pack_into("<H", entry, 14, create_time)
    struct.pack_into("<H", entry, 16, modify_date)
    struct.pack_into("<H", entry, 18, 0)
    struct.pack_into("<H", entry, 22, modify_time)
    struct.pack_into("<H", entry, 24, modify_date)
    struct.pack_into("<H", entry, 20, 0)
    struct.pack_into("<H", entry, 26, first_cluster)
    struct.pack_into("<I", entry, 28, size)
    return bytes(entry)


def lfn_entry(sequence: int, short_name11: bytes, parts: list[str | None]) -> bytes:
    entry = bytearray(32)
    entry[0] = sequence
    chars: list[int] = []
    for text in parts:
        if text is None:
            chars.extend([0, 0])
        else:
            chars.extend(ord(ch) for ch in text)
    chars.extend([0, 0, 0xFF, 0xFF, 0xFF, 0xFF])
    struct.pack_into("<H", entry, 1, chars[0])
    struct.pack_into("<H", entry, 3, chars[1])
    struct.pack_into("<H", entry, 5, chars[2])
    struct.pack_into("<H", entry, 7, chars[3])
    struct.pack_into("<H", entry, 9, chars[4])
    struct.pack_into("<H", entry, 14, chars[5])
    struct.pack_into("<H", entry, 16, chars[6])
    struct.pack_into("<H", entry, 18, chars[7])
    struct.pack_into("<H", entry, 20, chars[8])
    struct.pack_into("<H", entry, 22, chars[9])
    struct.pack_into("<H", entry, 24, chars[10])
    struct.pack_into("<H", entry, 26, chars[11])
    struct.pack_into("<H", entry, 28, chars[12])
    entry[11] = 0x0F
    entry[12] = 0
    entry[13] = lfn_checksum(short_name11)
    return bytes(entry)


def chain(fat: bytearray, clusters: list[int], eoc: int = 0xFFFF) -> None:
    for current, nxt in zip(clusters, clusters[1:] + [eoc]):
        struct.pack_into("<H", fat, current * 2, nxt)


def boot_sector() -> bytes:
    boot = bytearray(SECTOR)
    boot[:3] = b"\xEB\x3C\x90"
    boot[3:11] = b"TESTFAT "
    struct.pack_into("<H", boot, 11, SECTOR)
    boot[13] = SECTORS_PER_CLUSTER
    struct.pack_into("<H", boot, 14, RESERVED)
    boot[16] = FAT_COUNT
    struct.pack_into("<H", boot, 17, ROOT_ENTRIES)
    struct.pack_into("<H", boot, 19, TOTAL_SECTORS)
    boot[21] = 0xF8
    struct.pack_into("<H", boot, 22, FAT_SIZE)
    struct.pack_into("<H", boot, 24, 63)
    struct.pack_into("<H", boot, 26, 16)
    struct.pack_into("<I", boot, 28, 0)
    struct.pack_into("<I", boot, 32, 0)
    boot[36] = 0x80
    boot[38] = 0x29
    struct.pack_into("<I", boot, 39, 0x12345678)
    boot[43:54] = b"TEST VOLUME"
    boot[54:62] = b"FAT16   "
    boot[510:512] = b"\x55\xaa"
    return bytes(boot)


def repeated_cluster(marker: bytes, clusters: int) -> bytes:
    one_cluster = (marker * ((SECTOR // len(marker)) + 1))[:SECTOR]
    return one_cluster * clusters


def make_normal() -> bytes:
    image = bytearray(TOTAL_SECTORS * SECTOR)
    image[:SECTOR] = boot_sector()

    fat = bytearray(FAT_SIZE * SECTOR)
    struct.pack_into("<H", fat, 0, 0xFFF8)
    struct.pack_into("<H", fat, 2, 0xFFFF)
    target_clusters = list(range(2, 7))  # 2..6
    other_clusters = list(range(7, 11))  # 7..10
    subdir_cluster = 11
    chain(fat, target_clusters)
    chain(fat, other_clusters)
    chain(fat, [subdir_cluster])
    struct.pack_into("<H", fat, BAD_CLUSTER * 2, 0xFFF7)
    fat_bytes = bytes(fat)
    fat_start = RESERVED * SECTOR
    image[fat_start : fat_start + len(fat)] = fat
    image[fat_start + FAT_SIZE * SECTOR : fat_start + 2 * FAT_SIZE * SECTOR] = fat

    short_myfile = pack_name("MYFILE.TXT")
    root = bytearray(ROOT_ENTRIES * 32)
    root[0:32] = lfn_entry(
        0x41,
        short_myfile,
        ["L", "o", "n", "g", "N", "a", "m", "e", ".", "t"],
    )
    root[32:64] = short_entry("MYFILE.TXT", 0x20, 2, 5 * SECTOR)
    root[64:96] = short_entry("OTHER.TXT", 0x20, 7, 2048)
    root[96:128] = short_entry("SUBDIR", 0x10, 11, 0)
    root[128:160] = short_entry("EMPTY.TXT", 0x20, 0, 0)
    root_start = (RESERVED + FAT_COUNT * FAT_SIZE) * SECTOR
    image[root_start : root_start + len(root)] = root

    subdir = bytearray(SECTOR)
    subdir[0:32] = short_entry(".", 0x10, 11, 0)
    subdir[32:64] = short_entry("..", 0x10, 0, 0)
    image[(DATA_LBA + 9) * SECTOR : (DATA_LBA + 10) * SECTOR] = subdir

    old_content = (b"OLD" * 1000)[:5 * SECTOR]
    image[DATA_LBA * SECTOR : DATA_LBA * SECTOR + len(old_content)] = old_content
    other_content = b"OTHER FILE CONTENT\n"
    for cluster in other_clusters:
        start = (DATA_LBA + cluster - 2) * SECTOR
        image[start : start + SECTOR] = (other_content * 30)[:SECTOR]

    # Fill unused clusters with a deterministic non-zero background.  The bad
    # cluster must remain untouched by allocation and recovery tests.
    for cluster in range(2, CLUSTERS + 2):
        if 2 <= cluster <= 11 or cluster == BAD_CLUSTER:
            continue
        start = (DATA_LBA + cluster - 2) * SECTOR
        image[start : start + SECTOR] = bytes(
            [(cluster * 17 + cluster // 257) & 0xFF]
        ) * SECTOR
    bad_start = (DATA_LBA + BAD_CLUSTER - 2) * SECTOR
    image[bad_start : bad_start + SECTOR] = b"BAD" + b"\xCC" * 509
    return bytes(image)


def make_full() -> bytes:
    image = bytearray(TOTAL_SECTORS * SECTOR)
    image[:SECTOR] = boot_sector()

    fat = bytearray(FAT_SIZE * SECTOR)
    struct.pack_into("<H", fat, 0, 0xFFF8)
    struct.pack_into("<H", fat, 2, 0xFFFF)
    target_clusters = list(range(2, 7))
    chain(fat, target_clusters)
    struct.pack_into("<H", fat, BAD_CLUSTER * 2, 0xFFF7)

    free_available = [
        cluster
        for cluster in range(7, CLUSTERS + 2)
        if cluster != BAD_CLUSTER
    ]
    filler = []
    number = 1
    while free_available:
        take = free_available[:1000]
        del free_available[:1000]
        filler.append((number, take))
        chain(fat, take)
        number += 1

    fat_bytes = bytes(fat)
    fat_start = RESERVED * SECTOR
    image[fat_start : fat_start + len(fat)] = fat_bytes
    image[fat_start + FAT_SIZE * SECTOR : fat_start + 2 * FAT_SIZE * SECTOR] = fat_bytes

    root = bytearray(ROOT_ENTRIES * 32)
    root[0:32] = short_entry("MYFILE.TXT", 0x20, 2, 5 * SECTOR)
    offset = 32
    for number, take in filler:
        name = f"F{number:07X}"
        root[offset : offset + 32] = short_entry(
            name, 0x20, take[0], len(take) * SECTOR
        )
        offset += 32
    root_start = (RESERVED + FAT_COUNT * FAT_SIZE) * SECTOR
    image[root_start : root_start + len(root)] = root

    old_content = b"OLD" * 1000
    image[DATA_LBA * SECTOR : DATA_LBA * SECTOR + len(old_content)] = old_content
    return bytes(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--fixture", choices=("normal", "full"), default="normal")
    args = parser.parse_args()
    data = make_normal() if args.fixture == "normal" else make_full()
    Path(args.path).write_bytes(data)


if __name__ == "__main__":
    main()
