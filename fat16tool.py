#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fat16tool -- crash-safe, in-place replacement of a single file inside a
FAT16 disk image.

Image assumptions (validated before any modification):
  * no partition table: the image starts directly with the FAT boot sector
  * 512 bytes per sector, exactly two FATs, FAT16 cluster count
  * files are addressed by 8.3 short-name paths such as /DIR/FILE.TXT;
    long-file-name (LFN) directory entries are preserved byte-for-byte but
    are never used for addressing

Commands:
    check    IMAGE                      validate layout, FAT copies, cluster chains
    replace  IMAGE PATH (--file F | --stdin | --text T) [--journal J]
    recover  IMAGE [--journal J]

Before modifying anything, `replace` validates the BPB layout, requires both
FAT copies to be identical, and walks every active cluster chain; loops,
out-of-range links or clusters owned by two files are rejected and the image
is left untouched.  Free clusters are allocated lowest-number first; if there
is not enough room the image is left unchanged.

Crash safety
------------
All sector writes of one replace operation (file data, both FAT copies and
the directory entry carrying the new file length) are first written to a
sidecar journal (default: IMAGE.journal), then the journal is committed with
a single header sector, and only then the sectors are replayed into the image
in place.  The image itself is never copied or rewritten wholesale.

Journal layout (512-byte sectors):
    sector 0        commit header (written LAST, after fsync of the body):
                    magic "FAT16JRN", version, state, entry count,
                    table-sector count, image sector count,
                    crc32(body), crc32(header[0:32])
    sector 1..T     table: one little-endian u32 image sector number per entry
    sector T+1..    payload: the new 512-byte content of each tabled sector

Recovery (`recover`, also run automatically at the start of `replace`):
  * no journal                -> nothing to do
  * journal fails validation  -> incomplete: discard it; the image still holds
                                 the complete old version (it was never
                                 touched before the commit point)
  * journal valid+committed   -> replay every sector (idempotent), fsync,
                                 then delete the journal: the complete new
                                 version becomes visible

Crash model: storage may tear (half-write) only the most recently issued
sector write; sectors acknowledged by fsync are durable.  Replay is
idempotent, so a crash during recovery is handled by simply recovering again.
After any crash and recovery the image shows either the complete old file or
the complete new file, both FATs stay identical and no cluster is leaked.

Test hook: if the environment variable FAT16_CRASH_AT names a persistence
boundary ("journal-create", "journal-sector:<k>", "journal-data",
"journal-commit", "image:<k>", "image", "journal-clear"), the process kills
itself with SIGKILL immediately after that boundary has been fsynced.
"""

import argparse
import os
import signal
import struct
import sys
import zlib

SECTOR = 512
EOC_MIN = 0xFFF8          # FAT entries >= this mark the end of a chain
BAD_CLUSTER = 0xFFF7
FREE_CLUSTER = 0x0000
EOC_VALUE = 0xFFFF

ATTR_LFN = 0x0F
ATTR_DIRECTORY = 0x10
ATTR_VOLUME_ID = 0x08

JOURNAL_MAGIC = b'FAT16JRN'
JOURNAL_VERSION = 1
JOURNAL_COMMITTED = 1

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CORRUPT = 2
EXIT_NO_SPACE = 3
EXIT_NOT_FOUND = 4


class ImageError(Exception):
    """The image failed validation; refuse to modify it."""


class NoSpaceError(Exception):
    """Not enough free clusters; the image is left unchanged."""


class NotFoundError(Exception):
    """Target path not found or not addressable as an 8.3 short name."""


def u16(buf, off):
    return struct.unpack_from('<H', buf, off)[0]


def u32(buf, off):
    return struct.unpack_from('<I', buf, off)[0]


# --------------------------------------------------------------------------
# crash-injection hook (used by the test suite)

class CrashPoints(object):
    def __init__(self):
        self.at = os.environ.get('FAT16_CRASH_AT', '')

    def point(self, name):
        if self.at and self.at == name:
            sys.stderr.write('crash-point: %s\n' % name)
            sys.stderr.flush()
            os.kill(os.getpid(), signal.SIGKILL)


# --------------------------------------------------------------------------
# image layout

class Layout(object):
    """Parsed and validated BPB geometry."""

    def __init__(self, boot, image_sectors):
        if boot[510:512] != b'\x55\xaa':
            raise ImageError('missing 0x55AA boot signature')
        bps = u16(boot, 11)
        if bps != SECTOR:
            raise ImageError('bytes/sector is %d, expected 512' % bps)
        spc = boot[13]
        if spc == 0 or (spc & (spc - 1)) != 0:
            raise ImageError('sectors/cluster %d is not a power of two' % spc)
        reserved = u16(boot, 14)
        if reserved < 1:
            raise ImageError('reserved sector count is 0')
        num_fats = boot[16]
        if num_fats != 2:
            raise ImageError('FAT count is %d, expected 2' % num_fats)
        root_entries = u16(boot, 17)
        if root_entries == 0 or (root_entries * 32) % SECTOR != 0:
            raise ImageError('bad root entry count %d' % root_entries)
        total = u16(boot, 19) or u32(boot, 32)
        if total == 0:
            raise ImageError('total sector count is 0')
        if total != image_sectors:
            raise ImageError('BPB total sectors %d != image size %d sectors'
                             % (total, image_sectors))
        fat_size = u16(boot, 22)
        if fat_size == 0:
            raise ImageError('FAT size is 0 (FAT32 layout not supported)')

        self.media = boot[21]
        self.spc = spc
        self.reserved = reserved
        self.root_entries = root_entries
        self.total_sectors = total
        self.fat_size = fat_size
        self.root_sectors = root_entries * 32 // SECTOR
        self.cluster_size = spc * SECTOR

        self.fat1_off = reserved * SECTOR
        self.fat2_off = self.fat1_off + fat_size * SECTOR
        self.root_off = self.fat2_off + fat_size * SECTOR
        self.data_off = self.root_off + self.root_sectors * SECTOR

        data_sectors = total - self.data_off // SECTOR
        if data_sectors <= 0:
            raise ImageError('no data area')
        self.cluster_count = data_sectors // spc
        self.cluster_end = 2 + self.cluster_count   # first invalid cluster no.
        if not 4085 <= self.cluster_count < 65525:
            raise ImageError('cluster count %d is not FAT16' % self.cluster_count)
        if self.cluster_end * 2 > fat_size * SECTOR:
            raise ImageError('FAT too small for the cluster count')


class Fat(object):
    def __init__(self, raw):
        self.raw = raw                      # bytes, fat_size * SECTOR

    def get(self, cluster):
        return struct.unpack_from('<H', self.raw, cluster * 2)[0]


def read_sector(img, n):
    img.seek(n * SECTOR)
    data = img.read(SECTOR)
    if len(data) != SECTOR:
        raise ImageError('short read at sector %d' % n)
    return data


def read_region(img, off, length):
    img.seek(off)
    data = img.read(length)
    if len(data) != length:
        raise ImageError('short read at offset %d' % off)
    return data


def load_and_validate(img):
    """Validate geometry and FAT-pair consistency.  Returns (Layout, Fat)."""
    img.seek(0, os.SEEK_END)
    size = img.tell()
    if size == 0 or size % SECTOR != 0:
        raise ImageError('image size %d is not a multiple of 512' % size)
    lay = Layout(read_sector(img, 0), size // SECTOR)
    fat1 = read_region(img, lay.fat1_off, lay.fat_size * SECTOR)
    fat2 = read_region(img, lay.fat2_off, lay.fat_size * SECTOR)
    if fat1 != fat2:
        raise ImageError('the two FAT copies differ')
    fat = Fat(fat1)
    if fat.get(0) & 0xFF != lay.media:
        raise ImageError('FAT[0] media byte 0x%02X does not match BPB media 0x%02X'
                         % (fat.get(0) & 0xFF, lay.media))
    return lay, fat


# --------------------------------------------------------------------------
# directory walk

class FileEntry(object):
    __slots__ = ('path', 'name11', 'attr', 'first_cluster', 'size',
                 'sector', 'offset', 'chain')

    def __init__(self, path, name11, attr, first_cluster, size,
                 sector, offset, chain):
        self.path = path
        self.name11 = name11
        self.attr = attr
        self.first_cluster = first_cluster
        self.size = size
        self.sector = sector          # image sector holding the dir entry
        self.offset = offset          # offset of the 32-byte entry in it
        self.chain = chain


def short_to_str(name11):
    name = name11[0:8].decode('ascii', 'replace').rstrip(' ')
    ext = name11[8:11].decode('ascii', 'replace').rstrip(' ')
    return name + ('.' + ext if ext else '')


def collect_files(img, lay, fat):
    """Walk the root directory and every subdirectory cluster chain.

    Returns (files, owner).  Every active cluster chain is validated:
    loops, out-of-range links and clusters claimed by two different chains
    raise ImageError.  LFN entries are skipped (kept as raw bytes only).
    """
    owner = {}
    files = []

    def walk_chain(start, what):
        chain = []
        seen = set()
        c = start
        while True:
            if not 2 <= c < lay.cluster_end:
                raise ImageError('%s: cluster 0x%04X is outside the data area'
                                 % (what, c))
            if c in seen:
                raise ImageError('%s: cluster chain loops at cluster %d'
                                 % (what, c))
            if c in owner:
                raise ImageError('%s: cluster %d is also used by %s'
                                 % (what, c, owner[c]))
            seen.add(c)
            owner[c] = what
            chain.append(c)
            nxt = fat.get(c)
            if nxt >= EOC_MIN:
                return chain
            if nxt == FREE_CLUSTER:
                raise ImageError('%s: chain runs into a free cluster' % what)
            if nxt == BAD_CLUSTER:
                raise ImageError('%s: chain runs into a bad cluster' % what)
            c = nxt

    def cluster_sectors(c):
        base = lay.data_off // SECTOR + (c - 2) * lay.spc
        return range(base, base + lay.spc)

    def scan(sectors, prefix):
        for sec in sectors:
            block = read_sector(img, sec)
            for off in range(0, SECTOR, 32):
                ent = block[off:off + 32]
                tag = ent[0]
                if tag == 0x00:
                    return                       # rest of directory unused
                if tag == 0xE5:
                    continue                     # deleted
                attr = ent[11]
                if attr == ATTR_LFN:
                    continue                     # long name: raw bytes only
                if attr & ATTR_VOLUME_ID:
                    continue
                if tag == 0x2E:
                    continue                     # '.' or '..'
                name11 = ent[0:11]
                path = prefix + short_to_str(name11)
                first = u16(ent, 26)
                size = u32(ent, 28)
                if attr & ATTR_DIRECTORY:
                    if first == 0:
                        raise ImageError('%s: directory has no cluster chain' % path)
                    chain = walk_chain(first, path)
                    secs = [s for c in chain for s in cluster_sectors(c)]
                    scan(secs, path + '/')
                else:
                    if first == 0:
                        if size != 0:
                            raise ImageError('%s: size %d but no cluster chain'
                                             % (path, size))
                        chain = []
                    else:
                        chain = walk_chain(first, path)
                        if size > len(chain) * lay.cluster_size:
                            raise ImageError('%s: size %d exceeds its cluster chain'
                                             % (path, size))
                    files.append(FileEntry(path, name11, attr, first, size,
                                           sec, off, chain))

    root_secs = range(lay.root_off // SECTOR,
                      lay.root_off // SECTOR + lay.root_sectors)
    scan(root_secs, '/')
    return files, owner


# --------------------------------------------------------------------------
# 8.3 short-name path handling (no long-name addressing)

_SHORT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$%'-_@~`!(){}^#&")


def to_short_name(comp):
    """Validate one 8.3 path component; return its 11-byte directory form."""
    upper = comp.upper()
    if upper in ('.', '..'):
        raise NotFoundError('invalid path component %r' % comp)
    if upper.count('.') > 1:
        raise NotFoundError('not an 8.3 name: %r' % comp)
    if '.' in upper:
        base, ext = upper.split('.')
    else:
        base, ext = upper, ''
    if not base or len(base) > 8 or len(ext) > 3:
        raise NotFoundError('not an 8.3 name: %r' % comp)
    if any(ch not in _SHORT_CHARS for ch in base + ext):
        raise NotFoundError('invalid character in 8.3 name: %r' % comp)
    return base.encode('ascii').ljust(8, b' ') + ext.encode('ascii').ljust(3, b' ')


def normalize_path(path):
    parts = [p for p in path.split('/') if p not in ('', '.')]
    if not parts:
        raise NotFoundError('empty path')
    return '/' + '/'.join(short_to_str(to_short_name(p)) for p in parts)


def find_target(files, path):
    want = normalize_path(path)
    for f in files:
        if f.path == want:
            return f
    raise NotFoundError('no such file: %s (short-name addressing only)' % want)


# --------------------------------------------------------------------------
# planning: compute the exact sector writes, touch nothing yet

def plan_replace(img, lay, fat, target, new_data):
    """Return (writes, new_chain, freed, allocated).

    writes maps image sector number -> new 512-byte content.  The target's
    existing chain is reused as a prefix; extra clusters come from the free
    pool, lowest cluster number first.
    """
    csz = lay.cluster_size
    old_chain = list(target.chain)
    n_new = -(-len(new_data) // csz) if new_data else 0

    if n_new <= len(old_chain):
        new_chain = old_chain[:n_new]
        freed = old_chain[n_new:]
        alloc = []
    else:
        need = n_new - len(old_chain)
        free_list = [c for c in range(2, lay.cluster_end)
                     if fat.get(c) == FREE_CLUSTER]
        if len(free_list) < need:
            raise NoSpaceError('insufficient space: need %d more cluster(s), '
                               'only %d free' % (need, len(free_list)))
        alloc = free_list[:need]
        new_chain = old_chain + alloc
        freed = []

    fat_changes = {}
    for i, c in enumerate(new_chain):
        want = new_chain[i + 1] if i + 1 < len(new_chain) else EOC_VALUE
        if fat.get(c) != want:
            fat_changes[c] = want
    for c in freed:
        if fat.get(c) != FREE_CLUSTER:
            fat_changes[c] = FREE_CLUSTER

    writes = {}

    # file data (last cluster zero-padded)
    for i, c in enumerate(new_chain):
        chunk = new_data[i * csz:(i + 1) * csz]
        chunk = chunk + b'\x00' * (csz - len(chunk))
        base = lay.data_off // SECTOR + (c - 2) * lay.spc
        for j in range(lay.spc):
            writes[base + j] = chunk[j * SECTOR:(j + 1) * SECTOR]

    # FAT sectors, both copies, read-modify-write
    fat_new = bytearray(fat.raw)
    for c, v in fat_changes.items():
        struct.pack_into('<H', fat_new, c * 2, v)
    for s in range(lay.fat_size):
        old_sec = bytes(fat.raw[s * SECTOR:(s + 1) * SECTOR])
        new_sec = bytes(fat_new[s * SECTOR:(s + 1) * SECTOR])
        if new_sec != old_sec:
            writes[lay.fat1_off // SECTOR + s] = new_sec
            writes[lay.fat2_off // SECTOR + s] = new_sec

    # directory entry: only start cluster and file length change; name,
    # attributes, timestamps, position and any LFN entries stay untouched
    dir_sec = read_sector(img, target.sector)
    patched = bytearray(dir_sec)
    o = target.offset
    struct.pack_into('<H', patched, o + 26, new_chain[0] if new_chain else 0)
    struct.pack_into('<I', patched, o + 28, len(new_data))
    if bytes(patched) != dir_sec:
        writes[target.sector] = bytes(patched)

    # drop writes that would rewrite identical content
    final = {}
    for sec in sorted(writes):
        if read_sector(img, sec) != writes[sec]:
            final[sec] = writes[sec]
    return final, new_chain, freed, alloc


# --------------------------------------------------------------------------
# journal

def dir_fsync(path):
    try:
        fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_journal(jpath, writes, image_sectors, crash):
    """Write body sectors first, then the commit header as the last sector."""
    sectors = sorted(writes)
    table = b''.join(struct.pack('<I', s) for s in sectors)
    table += b'\x00' * (-len(table) % SECTOR)
    table_sectors = len(table) // SECTOR
    payload = b''.join(writes[s] for s in sectors)
    body = table + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF

    fd = os.open(jpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        crash.point('journal-create')
        for i in range(0, len(body), SECTOR):
            os.pwrite(fd, body[i:i + SECTOR], SECTOR + i)   # body after header slot
            os.fsync(fd)
            crash.point('journal-sector:%d' % ((SECTOR + i) // SECTOR))
        crash.point('journal-data')
        dir_fsync(jpath)                                    # name must survive
        header = bytearray(SECTOR)
        header[0:8] = JOURNAL_MAGIC
        struct.pack_into('<IIIIII', header, 8, JOURNAL_VERSION, JOURNAL_COMMITTED,
                         len(sectors), table_sectors, image_sectors, crc)
        struct.pack_into('<I', header, 32,
                         zlib.crc32(bytes(header[0:32])) & 0xFFFFFFFF)
        os.pwrite(fd, bytes(header), 0)                     # atomic commit point
        os.fsync(fd)
        crash.point('journal-commit')
    finally:
        os.close(fd)


def load_journal(jpath, image_sectors):
    """Return ('none'|'invalid'|'committed', [(sector, payload), ...])."""
    if not os.path.exists(jpath):
        return 'none', []
    with open(jpath, 'rb') as f:
        data = f.read()
    if len(data) < 2 * SECTOR:
        return 'invalid', []
    hdr = data[:SECTOR]
    if (hdr[0:8] != JOURNAL_MAGIC
            or u32(hdr, 8) != JOURNAL_VERSION
            or u32(hdr, 12) != JOURNAL_COMMITTED
            or (zlib.crc32(hdr[:32]) & 0xFFFFFFFF) != u32(hdr, 32)):
        return 'invalid', []
    count = u32(hdr, 16)
    table_sectors = u32(hdr, 20)
    if u32(hdr, 24) != image_sectors:
        return 'invalid', []
    if count == 0 or table_sectors == 0 or count * 4 > table_sectors * SECTOR:
        return 'invalid', []
    need = SECTOR + table_sectors * SECTOR + count * SECTOR
    if len(data) < need:
        return 'invalid', []
    body = data[SECTOR:need]
    if (zlib.crc32(body) & 0xFFFFFFFF) != u32(hdr, 28):
        return 'invalid', []
    entries = []
    payload_off = table_sectors * SECTOR
    for i in range(count):
        sec = struct.unpack_from('<I', body, i * 4)[0]
        if sec >= image_sectors:
            return 'invalid', []
        entries.append((sec, body[payload_off + i * SECTOR:
                                  payload_off + (i + 1) * SECTOR]))
    return 'committed', entries


def replay_writes(img, entries, crash):
    for i, (sec, payload) in enumerate(entries):
        img.seek(sec * SECTOR)
        img.write(payload)
        img.flush()
        os.fsync(img.fileno())
        crash.point('image:%d' % (i + 1))
    crash.point('image')


def do_recover(image_path, jpath, crash):
    """Bring the image to a consistent state.  Returns (state, sectors)."""
    image_sectors = os.path.getsize(image_path) // SECTOR
    state, entries = load_journal(jpath, image_sectors)
    if state == 'none':
        return 'none', []
    if state == 'invalid':
        os.remove(jpath)
        dir_fsync(jpath)
        return 'discarded', []
    with open(image_path, 'r+b') as img:
        replay_writes(img, entries, crash)
    os.remove(jpath)
    dir_fsync(jpath)
    crash.point('journal-clear')
    return 'replayed', [s for s, _ in entries]


# --------------------------------------------------------------------------
# commands

def read_new_content(args):
    if args.file is not None:
        with open(args.file, 'rb') as f:
            return f.read()
    if args.stdin:
        return sys.stdin.buffer.read()
    return args.text.encode('utf-8')


def cmd_check(args):
    try:
        with open(args.image, 'rb') as img:
            lay, fat = load_and_validate(img)
            files, owner = collect_files(img, lay, fat)
    except ImageError as exc:
        print('check: %s' % exc, file=sys.stderr)
        return EXIT_CORRUPT
    except OSError as exc:
        print('check: %s' % exc, file=sys.stderr)
        return EXIT_ERROR
    print('layout: %d sector(s) x 512 bytes, %d sector(s)/cluster, '
          '2 FATs x %d sector(s)' % (lay.total_sectors, lay.spc, lay.fat_size))
    print('        %d root entries, %d data cluster(s), cluster size %d byte(s)'
          % (lay.root_entries, lay.cluster_count, lay.cluster_size))
    print('FAT copies: identical')
    free = sum(1 for c in range(2, lay.cluster_end)
               if fat.get(c) == FREE_CLUSTER)
    leaked = [c for c in range(2, lay.cluster_end)
              if fat.get(c) not in (FREE_CLUSTER, BAD_CLUSTER) and c not in owner]
    print('files: %d' % len(files))
    for f in files:
        print('  %-36s %8d byte(s)  chain=%s' % (f.path, f.size, f.chain))
    print('free clusters: %d' % free)
    if leaked:
        print('warning: %d allocated cluster(s) not reachable from any file: %s'
              % (len(leaked), leaked[:10]))
    print('check: OK')
    return EXIT_OK


def cmd_replace(args):
    crash = CrashPoints()
    image_path = args.image
    jpath = args.journal or (image_path + '.journal')
    try:
        state, secs = do_recover(image_path, jpath, crash)
        if state == 'replayed':
            print('replace: finished interrupted update first '
                  '(replayed %d sector(s))' % len(secs))
        elif state == 'discarded':
            print('replace: discarded incomplete journal from interrupted update')
        new_data = read_new_content(args)
        with open(image_path, 'r+b') as img:
            lay, fat = load_and_validate(img)
            files, _owner = collect_files(img, lay, fat)
            target = find_target(files, args.path)
            writes, _new_chain, freed, alloc = plan_replace(
                img, lay, fat, target, new_data)
        if not writes:
            print('replace: %s already has the requested content; nothing to do'
                  % target.path)
            return EXIT_OK
        write_journal(jpath, writes, lay.total_sectors, crash)
        state, entries = load_journal(jpath, lay.total_sectors)
        if state != 'committed':
            raise ImageError('internal error: journal not committed after write')
        with open(image_path, 'r+b') as img:
            replay_writes(img, entries, crash)
        os.remove(jpath)
        dir_fsync(jpath)
        crash.point('journal-clear')
        print('replace: %s: %d -> %d byte(s)' % (target.path, target.size,
                                                 len(new_data)))
        if alloc:
            print('replace: allocated cluster(s): %s'
                  % ' '.join(map(str, alloc)))
        if freed:
            print('replace: freed cluster(s): %s' % ' '.join(map(str, freed)))
        print('replace: journaled %d image sector(s): %s'
              % (len(entries), ' '.join(str(s) for s, _ in entries)))
        print('replace: done; journal cleared')
        return EXIT_OK
    except NotFoundError as exc:
        print('replace: %s' % exc, file=sys.stderr)
        return EXIT_NOT_FOUND
    except NoSpaceError as exc:
        print('replace: %s; image left unchanged' % exc, file=sys.stderr)
        return EXIT_NO_SPACE
    except ImageError as exc:
        print('replace: refusing to modify image: %s' % exc, file=sys.stderr)
        return EXIT_CORRUPT
    except OSError as exc:
        print('replace: %s' % exc, file=sys.stderr)
        return EXIT_ERROR


def cmd_recover(args):
    crash = CrashPoints()
    jpath = args.journal or (args.image + '.journal')
    try:
        state, secs = do_recover(args.image, jpath, crash)
    except OSError as exc:
        print('recover: %s' % exc, file=sys.stderr)
        return EXIT_ERROR
    if state == 'none':
        print('recover: no journal present; image untouched')
    elif state == 'discarded':
        print('recover: discarded incomplete journal; previous version intact')
    else:
        print('recover: replayed %d image sector(s): %s'
              % (len(secs), ' '.join(map(str, secs))))
        print('recover: journal cleared; new version complete')
    return EXIT_OK


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='fat16tool',
        description='Crash-safe in-place replacement of one file in a '
                    'FAT16 image (512-byte sectors, two FATs, no partition '
                    'table, 8.3 short-name paths).')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('check', help='validate layout, FAT copies, cluster chains')
    p.add_argument('image')

    p = sub.add_parser('replace', help='replace one file in place (journaled)')
    p.add_argument('image')
    p.add_argument('path', help='8.3 short-name path, e.g. /DIR/FILE.TXT')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--file', help='read new content from this file')
    src.add_argument('--stdin', action='store_true',
                     help='read new content from stdin')
    src.add_argument('--text', help='use this UTF-8 text as the new content')
    p.add_argument('--journal', help='journal path (default: IMAGE.journal)')

    p = sub.add_parser('recover', help='finish or roll back an interrupted replace')
    p.add_argument('image')
    p.add_argument('--journal', help='journal path (default: IMAGE.journal)')

    args = parser.parse_args(argv)
    if args.command == 'check':
        return cmd_check(args)
    if args.command == 'replace':
        return cmd_replace(args)
    return cmd_recover(args)


if __name__ == '__main__':
    sys.exit(main())
