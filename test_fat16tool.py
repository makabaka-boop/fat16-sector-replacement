#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for fat16tool.

Contains an independent FAT16 image builder and reader (written directly
from the FAT16 spec, sharing no code with the tool) plus:

  * functional tests (grow/shrink/empty/same-size, subdirectories, LFN
    preservation, allocation policy, insufficient space, corrupt images)
  * a crash-recovery matrix: the replace process is killed with SIGKILL at
    every journal/image persistence boundary, a half-written (torn) sector
    is injected, `recover` is run, and the independent reader verifies file
    bytes, cluster ownership, FAT-pair consistency and untouched regions.
    Each scenario prints the recovered version and the sectors involved.
"""

import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, 'fat16tool.py')

SECTOR = 512


# ---------------------------------------------------------------------------
# independent image builder

def short11(name):
    name = name.upper()
    if '.' in name:
        base, ext = name.rsplit('.', 1)
    else:
        base, ext = name, ''
    return base.encode('ascii').ljust(8, b' ') + ext.encode('ascii').ljust(3, b' ')


def lfn_entries(short_name11, longname):
    """Build LFN directory entries (UCS-2, checksummed, reversed order)."""
    csum = 0
    for b in short_name11:
        csum = (((csum & 1) << 7) + (csum >> 1) + b) & 0xFF
    enc = longname.encode('utf-16-le') + b'\x00\x00'
    enc += b'\xff' * ((-len(enc)) % 26)
    chunks = [enc[i:i + 26] for i in range(0, len(enc), 26)]
    n = len(chunks)
    out = []
    for i in range(n):
        seq = i + 1
        if i == n - 1:
            seq |= 0x40
        e = bytearray(32)
        e[0] = seq
        e[1:11] = chunks[i][0:10]
        e[11] = 0x0F
        e[13] = csum
        e[14:26] = chunks[i][10:22]
        e[28:32] = chunks[i][22:26]
        out.append(bytes(e))
    out.reverse()
    return out


def make_image(spec, total_sectors=4200, spc=1, root_entries=256,
               fat_size=17, media=0xF8):
    """Build a FAT16 image (no partition table, 2 FATs) from a spec.

    spec: list of (path, content, lfn) -- content bytes for a file, None
    for a directory; lfn is an optional long name stored as LFN entries.
    """
    reserved = 1
    root_sectors = root_entries * 32 // SECTOR
    data_start = reserved + 2 * fat_size + root_sectors
    n_clusters = (total_sectors - data_start) // spc
    csz = spc * SECTOR

    fat = bytearray(fat_size * SECTOR)
    struct.pack_into('<HH', fat, 0, 0xFF00 | media, 0xFFFF)
    data = bytearray(n_clusters * csz)
    next_cluster = [2]

    def alloc(n):
        chain = list(range(next_cluster[0], next_cluster[0] + n))
        next_cluster[0] += n
        for i, c in enumerate(chain):
            struct.pack_into('<H', fat, c * 2,
                             chain[i + 1] if i + 1 < len(chain) else 0xFFFF)
        return chain

    children = {'': []}

    def ensure_dir(d):
        if d == '' or d in children:
            return
        parent = d.rsplit('/', 1)[0] if '/' in d else ''
        ensure_dir(parent)
        children[d] = []
        children[parent].append({'name': d.rsplit('/', 1)[-1],
                                 'is_dir': True, 'lfn': None, 'path': d})

    for item in spec:
        path, content = item[0].strip('/'), item[1]
        lfn = item[2] if len(item) > 2 else None
        parent = path.rsplit('/', 1)[0] if '/' in path else ''
        ensure_dir(parent)
        if content is None:
            ensure_dir(path)
        else:
            children[parent].append({'name': path.rsplit('/', 1)[-1],
                                     'is_dir': False, 'content': content,
                                     'lfn': lfn, 'path': path})

    date = (40 << 9) | (1 << 5) | 1        # 2020-01-01

    def short_entry(name11, attr, cluster, size):
        e = bytearray(32)
        e[0:11] = name11
        e[11] = attr
        struct.pack_into('<H', e, 22, date)
        struct.pack_into('<H', e, 26, cluster)
        struct.pack_into('<I', e, 28, size)
        return bytes(e)

    dir_cluster = {'': 0}
    for d in children:
        if d:
            dir_cluster[d] = alloc(1)[0]

    dir_tables = {}
    for d, ents in children.items():
        table = bytearray()
        if d:
            parent = d.rsplit('/', 1)[0] if '/' in d else ''
            table += short_entry(b'.' + b' ' * 10, 0x10, dir_cluster[d], 0)
            table += short_entry(b'..' + b' ' * 9, 0x10, dir_cluster[parent], 0)
        for ent in ents:
            n11 = short11(ent['name'])
            if ent.get('lfn'):
                for le in lfn_entries(n11, ent['lfn']):
                    table += le
            if ent['is_dir']:
                table += short_entry(n11, 0x10, dir_cluster[ent['path']], 0)
            else:
                content = ent['content']
                n = -(-len(content) // csz) if content else 0
                chain = alloc(n) if n else []
                for i, c in enumerate(chain):
                    chunk = content[i * csz:(i + 1) * csz]
                    off = (c - 2) * csz
                    data[off:off + len(chunk)] = chunk
                table += short_entry(n11, 0x20, chain[0] if chain else 0,
                                     len(content))
        limit = (root_sectors * SECTOR) if not d else csz
        assert len(table) <= limit, 'directory %r does not fit' % d
        dir_tables[d] = table

    boot = bytearray(SECTOR)
    boot[0:3] = b'\xeb\x3c\x90'
    boot[3:11] = b'PYFAT16 '
    struct.pack_into('<H', boot, 11, SECTOR)
    boot[13] = spc
    struct.pack_into('<H', boot, 14, reserved)
    boot[16] = 2
    struct.pack_into('<H', boot, 17, root_entries)
    struct.pack_into('<H', boot, 19, total_sectors)
    boot[21] = media
    struct.pack_into('<H', boot, 22, fat_size)
    struct.pack_into('<H', boot, 24, 1)
    struct.pack_into('<H', boot, 26, 1)
    boot[36] = 0x80
    boot[38] = 0x29
    struct.pack_into('<I', boot, 39, 0x12345678)
    boot[43:54] = b'NO NAME    '
    boot[54:62] = b'FAT16   '
    boot[510:512] = b'\x55\xaa'

    img = bytearray(total_sectors * SECTOR)
    img[0:SECTOR] = boot
    fo = reserved * SECTOR
    img[fo:fo + fat_size * SECTOR] = fat
    img[fo + fat_size * SECTOR:fo + 2 * fat_size * SECTOR] = fat
    ro = fo + 2 * fat_size * SECTOR
    root = bytearray(root_sectors * SECTOR)
    root[:len(dir_tables[''])] = dir_tables['']
    img[ro:ro + root_sectors * SECTOR] = root
    do = ro + root_sectors * SECTOR
    for d, table in dir_tables.items():
        if d:
            off = (dir_cluster[d] - 2) * csz
            data[off:off + len(table)] = table
    img[do:do + len(data)] = data
    return bytes(img)


# ---------------------------------------------------------------------------
# independent image reader

class Reader(object):
    """Standalone FAT16 reader used to verify the tool's results."""

    def __init__(self, data):
        self.data = data
        self.spc = data[13]
        reserved = struct.unpack_from('<H', data, 14)[0]
        root_entries = struct.unpack_from('<H', data, 17)[0]
        self.fat_size = struct.unpack_from('<H', data, 22)[0]
        self.fat1_off = reserved * SECTOR
        self.fat2_off = self.fat1_off + self.fat_size * SECTOR
        self.root_off = self.fat2_off + self.fat_size * SECTOR
        self.root_sectors = root_entries * 32 // SECTOR
        self.data_off = self.root_off + self.root_sectors * SECTOR
        total = struct.unpack_from('<H', data, 19)[0] or \
            struct.unpack_from('<I', data, 32)[0]
        self.n_clusters = (total - self.data_off // SECTOR) // self.spc
        self.csz = self.spc * SECTOR

    def fat_entry(self, c, which=1):
        off = (self.fat1_off if which == 1 else self.fat2_off) + c * 2
        return struct.unpack_from('<H', self.data, off)[0]

    def chain_of(self, start):
        chain = []
        seen = set()
        c = start
        while True:
            if not 2 <= c < 2 + self.n_clusters:
                raise ValueError('cluster 0x%04X out of range' % c)
            if c in seen:
                raise ValueError('loop at cluster %d' % c)
            seen.add(c)
            chain.append(c)
            nxt = self.fat_entry(c)
            if nxt >= 0xFFF8:
                return chain
            if nxt == 0:
                raise ValueError('chain runs into free cluster')
            c = nxt

    def files(self):
        result = {}
        owner = {}
        problems = []
        seen_dirs = set()

        def scan(sectors, prefix):
            for sec in sectors:
                block = self.data[sec * SECTOR:(sec + 1) * SECTOR]
                for off in range(0, SECTOR, 32):
                    e = block[off:off + 32]
                    if e[0] == 0x00:
                        return
                    if e[0] == 0xE5 or e[11] == 0x0F or e[11] & 0x08 \
                            or e[0] == 0x2E:
                        continue
                    base = e[0:8].decode('ascii', 'replace').rstrip(' ')
                    ext = e[8:11].decode('ascii', 'replace').rstrip(' ')
                    name = base + ('.' + ext if ext else '')
                    path = prefix + name
                    fc = struct.unpack_from('<H', e, 26)[0]
                    size = struct.unpack_from('<I', e, 28)[0]
                    chain = self.chain_of(fc) if fc else []
                    for c in chain:
                        if c in owner:
                            problems.append('cluster %d cross-linked: %s and %s'
                                            % (c, owner[c], path))
                        owner.setdefault(c, path)
                    if e[11] & 0x10:
                        if fc in seen_dirs:
                            problems.append('directory cluster %d reused' % fc)
                            continue
                        seen_dirs.add(fc)
                        secs = [self.data_off // SECTOR + (c - 2) * self.spc + j
                                for c in chain for j in range(self.spc)]
                        scan(secs, path + '/')
                    else:
                        result[path] = {'cluster': fc, 'size': size,
                                        'chain': chain, 'sector': sec,
                                        'offset': off}

        root_secs = [self.root_off // SECTOR + i
                     for i in range(self.root_sectors)]
        scan(root_secs, '')
        return result, owner, problems

    def read_file(self, path):
        files, _, _ = self.files()
        ent = files[path]
        out = bytearray()
        for c in ent['chain']:
            off = self.data_off + (c - 2) * self.csz
            out += self.data[off:off + self.csz]
        return bytes(out[:ent['size']])

    def consistency(self):
        """[] if healthy: identical FATs, no chain errors, no cross-links,
        no leaked (allocated but unreachable) clusters."""
        probs = []
        if self.data[self.fat1_off:self.fat1_off + self.fat_size * SECTOR] != \
                self.data[self.fat2_off:self.fat2_off + self.fat_size * SECTOR]:
            probs.append('FAT copies differ')
        try:
            _files, owner, walk_probs = self.files()
            probs.extend(walk_probs)
        except ValueError as exc:
            probs.append('chain error: %s' % exc)
            return probs
        allocated = {c for c in range(2, 2 + self.n_clusters)
                     if self.fat_entry(c) != 0}
        bad = {c for c in allocated if self.fat_entry(c) == 0xFFF7}
        leaked = sorted(allocated - set(owner) - bad)
        if leaked:
            probs.append('leaked clusters: %s' % leaked[:10])
        return probs


# ---------------------------------------------------------------------------
# helpers

def run_tool(*args, env_extra=None, input_data=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, TOOL] + list(args),
                          capture_output=True, env=env, input=input_data)


SPEC = [
    ('README.TXT', b'hello world\n' * 30, None),                  # 360 B
    ('SUB', None, None),
    ('SUB/TARGET.BIN', bytes(range(256)) * 6, None),              # 1536 B
    ('SUB/NOTES.TXT', b'notes' * 100, None),                      # 500 B
    ('LONGFI~1.TXT', b'LFN content\n' * 40, 'Long File Name.txt'),
    ('EMPTY.DAT', b'', None),
]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='fat16test-')
        self.addCleanup(shutil.rmtree, self.tmp)
        self.orig = make_image(SPEC)
        self.img = os.path.join(self.tmp, 'test.img')
        self._write(self.img, self.orig)

    @staticmethod
    def _write(path, data):
        with open(path, 'wb') as f:
            f.write(data)

    def read_img(self):
        with open(self.img, 'rb') as f:
            return f.read()

    def replace(self, path, content, expect=0, **kw):
        cf = os.path.join(self.tmp, 'content.bin')
        self._write(cf, content)
        cp = run_tool('replace', self.img, path, '--file', cf, **kw)
        self.assertEqual(cp.returncode, expect,
                         'rc=%d stderr=%s' % (cp.returncode, cp.stderr))
        return cp


# ---------------------------------------------------------------------------
# functional tests

class CheckTest(Base):
    def test_check_ok(self):
        cp = run_tool('check', self.img)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertIn(b'FAT copies: identical', cp.stdout)
        self.assertIn(b'check: OK', cp.stdout)
        self.assertIn(b'/SUB/TARGET.BIN', cp.stdout)


class ReplaceTest(Base):
    def verify_result(self, orig_bytes, new_bytes, target_path, new_content):
        """Independent verification of one replace run."""
        r_old, r_new = Reader(orig_bytes), Reader(new_bytes)
        self.assertEqual(r_new.consistency(), [])
        files_old, _, _ = r_old.files()
        files_new, _, _ = r_new.files()

        # file bytes
        self.assertEqual(r_new.read_file(target_path), new_content)

        # cluster ownership: old chain reused as prefix, free clusters
        # taken lowest-number first
        old_chain = files_old[target_path]['chain']
        n_new = -(-len(new_content) // r_old.csz) if new_content else 0
        free_old = [c for c in range(2, 2 + r_old.n_clusters)
                    if r_old.fat_entry(c) == 0]
        if n_new <= len(old_chain):
            expected = old_chain[:n_new]
            freed = old_chain[n_new:]
        else:
            expected = old_chain + free_old[:n_new - len(old_chain)]
            freed = []
        self.assertEqual(files_new[target_path]['chain'], expected)
        for c in freed:
            self.assertEqual(r_new.fat_entry(c), 0,
                             'freed cluster %d still allocated' % c)

        # other files untouched
        for p in files_old:
            if p != target_path:
                self.assertEqual(r_new.read_file(p), r_old.read_file(p), p)
                self.assertEqual(files_new[p]['chain'], files_old[p]['chain'], p)

        # changed sectors stay inside allowed regions
        allowed = set()
        for c in set(old_chain) | set(expected):
            base = r_old.data_off // SECTOR + (c - 2) * r_old.spc
            allowed.update(range(base, base + r_old.spc))
        allowed.update(range(r_old.fat1_off // SECTOR,
                             r_old.fat1_off // SECTOR + r_old.fat_size))
        allowed.update(range(r_old.fat2_off // SECTOR,
                             r_old.fat2_off // SECTOR + r_old.fat_size))
        allowed.add(files_old[target_path]['sector'])
        changed = {i for i in range(len(orig_bytes) // SECTOR)
                   if orig_bytes[i * SECTOR:(i + 1) * SECTOR]
                   != new_bytes[i * SECTOR:(i + 1) * SECTOR]}
        self.assertFalse(changed - allowed,
                         'unexpected sectors changed: %s'
                         % sorted(changed - allowed))

        # directory entry: only start-cluster and size fields may differ;
        # name, attributes, timestamps and position are preserved
        sec = files_old[target_path]['sector']
        off = files_old[target_path]['offset']
        old_sec = orig_bytes[sec * SECTOR:(sec + 1) * SECTOR]
        new_sec = new_bytes[sec * SECTOR:(sec + 1) * SECTOR]
        for i in range(SECTOR):
            if old_sec[i] != new_sec[i]:
                self.assertTrue(off + 26 <= i < off + 32,
                                'dir entry byte %d changed' % i)
        self.assertEqual(struct.unpack_from('<I', new_sec, off + 28)[0],
                         len(new_content))
        self.assertEqual(struct.unpack_from('<H', new_sec, off + 26)[0],
                         expected[0] if expected else 0)
        return changed

    def test_grow(self):
        content = bytes((i * 7 + 3) & 0xFF for i in range(3000))
        self.replace('SUB/TARGET.BIN', content)
        self.verify_result(self.orig, self.read_img(), 'SUB/TARGET.BIN', content)
        self.assertFalse(os.path.exists(self.img + '.journal'))

    def test_shrink(self):
        content = b'small\n' * 20                       # 120 B, 1 cluster
        self.replace('SUB/TARGET.BIN', content)
        self.verify_result(self.orig, self.read_img(), 'SUB/TARGET.BIN', content)

    def test_to_empty(self):
        self.replace('SUB/TARGET.BIN', b'')
        self.verify_result(self.orig, self.read_img(), 'SUB/TARGET.BIN', b'')

    def test_from_empty(self):
        content = bytes(range(256)) * 5                 # 1280 B, 3 clusters
        self.replace('EMPTY.DAT', content)
        self.verify_result(self.orig, self.read_img(), 'EMPTY.DAT', content)

    def test_same_size(self):
        content = bytes((i * 13 + 1) & 0xFF for i in range(1536))
        self.replace('SUB/TARGET.BIN', content)
        changed = self.verify_result(self.orig, self.read_img(),
                                     'SUB/TARGET.BIN', content)
        r = Reader(self.orig)
        files, _, _ = r.files()
        chain = files['SUB/TARGET.BIN']['chain']
        expected = {r.data_off // SECTOR + (c - 2) for c in chain}
        self.assertEqual(changed, expected)             # data sectors only

    def test_root_file(self):
        content = b'root replacement\n' * 40
        self.replace('/README.TXT', content)
        self.verify_result(self.orig, self.read_img(), 'README.TXT', content)

    def test_noop_when_identical(self):
        files, _, _ = Reader(self.orig).files()
        content = Reader(self.orig).read_file('README.TXT')
        cp = self.replace('README.TXT', content)
        self.assertIn(b'nothing to do', cp.stdout)
        self.assertEqual(self.read_img(), self.orig)

    def test_text_and_stdin(self):
        cp = run_tool('replace', self.img, 'README.TXT', '--text', 'abc')
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(Reader(self.read_img()).read_file('README.TXT'), b'abc')
        cp = run_tool('replace', self.img, 'README.TXT', '--stdin',
                      input_data=b'xyz123')
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(Reader(self.read_img()).read_file('README.TXT'),
                         b'xyz123')

    def test_lfn_entries_preserved(self):
        # replace the LFN file itself, addressed by its short name
        content = b'new long file content\n' * 30
        self.replace('LONGFI~1.TXT', content)
        new = self.read_img()
        self.verify_result(self.orig, new, 'LONGFI~1.TXT', content)
        r = Reader(self.orig)
        root = [self.orig[r.root_off + i * SECTOR:
                          r.root_off + (i + 1) * SECTOR]
                for i in range(r.root_sectors)]
        root_new = [new[r.root_off + i * SECTOR:
                        r.root_off + (i + 1) * SECTOR]
                    for i in range(r.root_sectors)]
        lfn_old = [e for sec in root for e in
                   (sec[i:i + 32] for i in range(0, SECTOR, 32)) if e[11] == 0x0F]
        lfn_new = [e for sec in root_new for e in
                   (sec[i:i + 32] for i in range(0, SECTOR, 32)) if e[11] == 0x0F]
        self.assertTrue(lfn_old)
        self.assertEqual(lfn_old, lfn_new)              # raw bytes preserved

    def test_lfn_not_addressable(self):
        cp = run_tool('replace', self.img, 'Long File Name.txt',
                      '--text', 'x')
        self.assertEqual(cp.returncode, 4)
        self.assertEqual(self.read_img(), self.orig)

    def test_not_found(self):
        cp = run_tool('replace', self.img, 'NOPE.TXT', '--text', 'x')
        self.assertEqual(cp.returncode, 4)
        cp = run_tool('replace', self.img, 'SUB', '--text', 'x')
        self.assertEqual(cp.returncode, 4)
        self.assertEqual(self.read_img(), self.orig)

    def test_spc4_image(self):
        orig = make_image(SPEC, total_sectors=17000, spc=4)
        self._write(self.img, orig)
        cp = run_tool('check', self.img)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        content = bytes((i * 5 + 1) & 0xFF for i in range(9000))
        self.replace('SUB/TARGET.BIN', content)
        self.verify_result(orig, self.read_img(), 'SUB/TARGET.BIN', content)

    def test_insufficient_space(self):
        r = Reader(self.orig)
        free = sum(1 for c in range(2, 2 + r.n_clusters)
                   if r.fat_entry(c) == 0)
        leave = 3
        filler = bytes((free - leave) * SECTOR)
        spec = SPEC + [('FILLER.BIN', filler, None)]
        orig = make_image(spec)
        self._write(self.img, orig)
        content = bytes(5 * SECTOR)                     # needs 4 extra > 3 free
        cf = os.path.join(self.tmp, 'big.bin')
        self._write(cf, content)
        cp = run_tool('replace', self.img, 'README.TXT', '--file', cf)
        self.assertEqual(cp.returncode, 3, cp.stderr)
        self.assertIn(b'insufficient space', cp.stderr)
        self.assertEqual(self.read_img(), orig)         # image unchanged
        self.assertFalse(os.path.exists(self.img + '.journal'))


class CorruptTest(Base):
    def corrupt_variants(self):
        r = Reader(self.orig)
        files, _, _ = r.files()
        chain = files['SUB/TARGET.BIN']['chain']

        m = bytearray(self.orig)
        m[r.fat2_off + 20] ^= 0xFF
        yield 'fat-mismatch', bytes(m)

        m = bytearray(self.orig)
        for base in (r.fat1_off, r.fat2_off):
            struct.pack_into('<H', m, base + chain[-1] * 2, chain[0])
        yield 'loop', bytes(m)

        m = bytearray(self.orig)
        for base in (r.fat1_off, r.fat2_off):
            struct.pack_into('<H', m, base + chain[-1] * 2, 5000)  # beyond end
        yield 'out-of-bounds', bytes(m)

        m = bytearray(self.orig)
        notes = files['SUB/NOTES.TXT']
        struct.pack_into('<H', m, notes['sector'] * SECTOR + notes['offset'] + 26,
                         chain[1])                     # NOTES starts inside TARGET
        yield 'cross-link', bytes(m)

    def test_corrupt_rejected(self):
        for name, bad in self.corrupt_variants():
            self._write(self.img, bad)
            cp = run_tool('check', self.img)
            self.assertEqual(cp.returncode, 2, '%s: %s' % (name, cp.stderr))
            cp = run_tool('replace', self.img, 'README.TXT', '--text', 'x')
            self.assertEqual(cp.returncode, 2, '%s: %s' % (name, cp.stderr))
            self.assertEqual(self.read_img(), bad, '%s: image modified' % name)
            print('corrupt %-14s -> rejected: %s'
                  % (name, cp.stderr.decode().strip()))


# ---------------------------------------------------------------------------
# crash / recovery matrix

class CrashRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='fat16crash-')
        self.addCleanup(shutil.rmtree, self.tmp)
        self.orig = make_image(SPEC)
        self.new_content = bytes((i * 31 + 7) & 0xFF for i in range(2500))
        self.content_file = os.path.join(self.tmp, 'new.bin')
        with open(self.content_file, 'wb') as f:
            f.write(self.new_content)
        # reference: one clean run produces the expected "new" image
        ref = os.path.join(self.tmp, 'ref.img')
        with open(ref, 'wb') as f:
            f.write(self.orig)
        cp = run_tool('replace', ref, 'SUB/TARGET.BIN',
                      '--file', self.content_file)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        with open(ref, 'rb') as f:
            self.ref_new = f.read()
        os.remove(ref)
        m = re.search(rb'journaled (\d+) image sector', cp.stdout)
        self.n_img = int(m.group(1))
        self.assertLessEqual(self.n_img, 128)
        n_body = 1 + self.n_img                 # 1 table sector + payload
        self.boundaries = (['journal-create']
                           + ['journal-sector:%d' % k
                              for k in range(1, n_body + 1)]
                           + ['journal-data', 'journal-commit']
                           + ['image:%d' % k
                              for k in range(1, self.n_img + 1)]
                           + ['image', 'journal-clear'])

    def tears_for(self, boundary):
        if boundary in ('journal-create', 'journal-clear'):
            return [None]
        if boundary.startswith('journal-sector') \
                or boundary in ('journal-data', 'journal-commit'):
            return [None, 'zero-first-half', 'zero-second-half']
        return [None, 'img-first-half-old', 'img-second-half-old']

    def expected(self, boundary, tear):
        if boundary == 'journal-commit':
            # tearing the first half of the commit sector destroys the
            # commit; the second half holds no critical fields
            return 'OLD' if tear == 'zero-first-half' else 'NEW'
        if boundary.startswith('image') or boundary == 'journal-clear':
            return 'NEW'
        return 'OLD'

    def apply_tear(self, img, jrn, boundary, tear):
        """Inject a half-written sector; return a label for the report."""
        if tear in ('zero-first-half', 'zero-second-half'):
            if not os.path.exists(jrn) or os.path.getsize(jrn) < SECTOR:
                return None
            size = os.path.getsize(jrn)
            secno = 0 if boundary == 'journal-commit' else size // SECTOR - 1
            with open(jrn, 'r+b') as f:
                f.seek(secno * SECTOR)
                cur = f.read(SECTOR)
                torn = (b'\x00' * 256 + cur[256:]) if tear == 'zero-first-half' \
                    else (cur[:256] + b'\x00' * 256)
                f.seek(secno * SECTOR)
                f.write(torn)
                f.flush()
                os.fsync(f.fileno())
            return 'journal-sector-%d' % secno
        # image tear: newest sector written so far, half old / half new
        with open(img, 'rb') as f:
            cur = f.read()
        diff = [i for i in range(len(cur) // SECTOR)
                if cur[i * SECTOR:(i + 1) * SECTOR]
                != self.orig[i * SECTOR:(i + 1) * SECTOR]]
        if not diff:
            return None
        secno = diff[-1]
        old_sec = self.orig[secno * SECTOR:(secno + 1) * SECTOR]
        new_sec = bytearray(cur[secno * SECTOR:(secno + 1) * SECTOR])
        if tear == 'img-first-half-old':
            new_sec[:256] = old_sec[:256]
        else:
            new_sec[256:] = old_sec[256:]
        with open(img, 'r+b') as f:
            f.seek(secno * SECTOR)
            f.write(bytes(new_sec))
            f.flush()
            os.fsync(f.fileno())
        return 'image-sector-%d' % secno

    def scenario(self, boundary, tear):
        img = os.path.join(self.tmp, 'crash.img')
        jrn = img + '.journal'
        with open(img, 'wb') as f:
            f.write(self.orig)
        cp = run_tool('replace', img, 'SUB/TARGET.BIN',
                      '--file', self.content_file,
                      env_extra={'FAT16_CRASH_AT': boundary})
        self.assertEqual(cp.returncode, -signal.SIGKILL,
                         'boundary %s: rc=%d' % (boundary, cp.returncode))
        torn = self.apply_tear(img, jrn, boundary, tear) if tear else None
        cp2 = run_tool('recover', img)
        self.assertEqual(cp2.returncode, 0, cp2.stderr)
        self.assertFalse(os.path.exists(jrn), 'journal left behind')
        with open(img, 'rb') as f:
            final = f.read()
        self.assertEqual(Reader(final).consistency(), [],
                         'boundary %s tear %s' % (boundary, tear))
        if final == self.orig:
            version = 'OLD'
        elif final == self.ref_new:
            version = 'NEW'
        else:
            self.fail('boundary %s tear %s: image is neither the complete '
                      'old nor the complete new version' % (boundary, tear))
        changed = [i for i in range(len(final) // SECTOR)
                   if final[i * SECTOR:(i + 1) * SECTOR]
                   != self.orig[i * SECTOR:(i + 1) * SECTOR]]
        print('boundary=%-18s tear=%-26s -> recovered=%s '
              'changed_sectors=%s recover="%s"'
              % (boundary, ('%s@%s' % (tear, torn)) if tear else '-',
                 version, changed,
                 cp2.stdout.decode().strip().splitlines()[0]))
        return version

    def test_crash_matrix(self):
        counts = {'OLD': 0, 'NEW': 0}
        for boundary in self.boundaries:
            for tear in self.tears_for(boundary):
                version = self.scenario(boundary, tear)
                self.assertEqual(version, self.expected(boundary, tear),
                                 '%s / %s' % (boundary, tear))
                counts[version] += 1
        print('crash matrix: %d scenarios, recovered OLD=%d NEW=%d'
              % (sum(counts.values()), counts['OLD'], counts['NEW']))

    def test_recovery_itself_interrupted(self):
        img = os.path.join(self.tmp, 'rec.img')
        with open(img, 'wb') as f:
            f.write(self.orig)
        cp = run_tool('replace', img, 'SUB/TARGET.BIN',
                      '--file', self.content_file,
                      env_extra={'FAT16_CRASH_AT': 'journal-commit'})
        self.assertEqual(cp.returncode, -signal.SIGKILL)
        # kill the recovery in the middle of the replay
        cp = run_tool('recover', img,
                      env_extra={'FAT16_CRASH_AT': 'image:1'})
        self.assertEqual(cp.returncode, -signal.SIGKILL)
        torn = self.apply_tear(img, img + '.journal', 'image:1',
                               'img-second-half-old')
        # and again after the last replayed sector, before journal removal
        cp = run_tool('recover', img, env_extra={'FAT16_CRASH_AT': 'image'})
        self.assertEqual(cp.returncode, -signal.SIGKILL)
        cp = run_tool('recover', img)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        with open(img, 'rb') as f:
            final = f.read()
        self.assertEqual(final, self.ref_new)
        self.assertEqual(Reader(final).consistency(), [])
        print('interrupted recovery (torn=%s) -> recovered=NEW '
              'changed_sectors=%s'
              % (torn, [i for i in range(len(final) // SECTOR)
                        if final[i * SECTOR:(i + 1) * SECTOR]
                        != self.orig[i * SECTOR:(i + 1) * SECTOR]]))

    def test_replace_after_crash(self):
        img = os.path.join(self.tmp, 'again.img')
        with open(img, 'wb') as f:
            f.write(self.orig)
        cp = run_tool('replace', img, 'SUB/TARGET.BIN',
                      '--file', self.content_file,
                      env_extra={'FAT16_CRASH_AT': 'journal-commit'})
        self.assertEqual(cp.returncode, -signal.SIGKILL)
        # a new replace first finishes the interrupted one, then proceeds
        content2 = b'second generation\n' * 10
        cf = os.path.join(self.tmp, 'second.bin')
        with open(cf, 'wb') as f:
            f.write(content2)
        cp = run_tool('replace', img, 'SUB/TARGET.BIN', '--file', cf)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertIn(b'finished interrupted update', cp.stdout)
        with open(img, 'rb') as f:
            final = f.read()
        r = Reader(final)
        self.assertEqual(r.consistency(), [])
        self.assertEqual(r.read_file('SUB/TARGET.BIN'), content2)
        print('replace-after-crash -> final version = second content, '
              'image consistent')

    def test_recover_without_journal(self):
        img = os.path.join(self.tmp, 'clean.img')
        with open(img, 'wb') as f:
            f.write(self.orig)
        cp = run_tool('recover', img)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertIn(b'no journal present', cp.stdout)
        with open(img, 'rb') as f:
            self.assertEqual(f.read(), self.orig)


if __name__ == '__main__':
    unittest.main(verbosity=1)
