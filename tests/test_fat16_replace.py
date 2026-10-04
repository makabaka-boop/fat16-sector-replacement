from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tests.fat16_make_testimage import make_normal, make_full  # noqa: E402
from tests.fat16_reader import IndependentFat16  # noqa: E402

TOOL = ROOT / "fat16_replace.py"
BAD_CLUSTER = 200


def run_tool(image: Path, *args: str, env: dict[str, str] | None = None):
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("FAT16_")
    }
    if env:
        clean_env.update(env)
    proc = subprocess.run(
        [sys.executable, str(TOOL), str(image), *args],
        text=True,
        capture_output=True,
        env=clean_env,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise AssertionError(
            f"tool did not emit JSON: rc={proc.returncode}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return proc.returncode, payload


def replace(image: Path, path: str, input_path: Path, **env: str):
    return run_tool(image, path, "-i", str(input_path), env=env or None)


def faulted_replace(
    image: Path, path: str, input_path: Path, boundary: int
) -> subprocess.CompletedProcess[str]:
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("FAT16_")
    }
    clean_env["FAT16_FAULT_AFTER"] = str(boundary)
    clean_env["FAT16_TORN"] = "1"
    return subprocess.run(
        [sys.executable, str(TOOL), str(image), path, "-i", str(input_path)],
        text=True,
        capture_output=True,
        env=clean_env,
    )


def recover(image: Path):
    return run_tool(image, "--recover")


def changed_sectors(before: bytes, after: bytes) -> list[int]:
    assert len(before) == len(after)
    return [
        sector
        for sector in range(len(before) // 512)
        if before[sector * 512 : (sector + 1) * 512]
        != after[sector * 512 : (sector + 1) * 512]
    ]


def mutate_fat_pair(image: bytearray, layout: IndependentFat16, cluster: int, value: int):
    for which in (0, 1):
        start = (layout.reserved + which * layout.fat_size) * 512
        struct.pack_into("<H", image, start + cluster * 2, value)


def assert_directory_entry_preserved(
    baseline: IndependentFat16,
    current: IndependentFat16,
    entry,
    expected_cluster: int,
    expected_size: int,
):
    old_sector = baseline.sector(entry.sector)
    new_sector = current.sector(entry.sector)
    old_record = old_sector[entry.offset : entry.offset + 32]
    new_record = new_sector[entry.offset : entry.offset + 32]

    assert old_record[:26] == new_record[:26], "name/attribute/metadata changed"
    assert struct.unpack_from("<H", new_record, 26)[0] == expected_cluster
    assert struct.unpack_from("<I", new_record, 28)[0] == expected_size
    if entry.name == "MYFILE.TXT":
        assert old_sector[:32] == new_sector[:32], "LFN record changed"


def assert_clean_fat(reader: IndependentFat16):
    fat0, fat1 = reader.fats
    assert fat0 == fat1, "FAT copies differ after recovery"
    owners = reader.allocation_owners()

    for cluster in range(2, reader.max_cluster + 1):
        value = reader.fat_entry(0, cluster)
        if value == 0xFFF7:
            assert cluster == BAD_CLUSTER
            sector = reader.cluster_sector(cluster)
            assert reader.sector(sector) == b"BAD" + b"\xCC" * 509
        elif value == 0:
            assert cluster not in owners
        else:
            assert cluster in owners, f"lost/orphan allocation at cluster {cluster}"


def assert_expected_image(
    image_path: Path,
    expected_bytes: bytes,
    baseline_bytes: bytes,
    target_path: str,
    expected_content: bytes,
    expected_chain: tuple[int, ...],
    expected_sectors: list[int],
):
    actual = image_path.read_bytes()
    assert len(actual) == len(baseline_bytes)
    differences = changed_sectors(baseline_bytes, actual)
    assert differences == sorted(expected_sectors), differences
    assert actual == expected_bytes

    baseline = IndependentFat16(image_path)
    # Re-point baseline raw bytes to the pristine snapshot.
    baseline.bytes = baseline_bytes
    current = IndependentFat16(image_path)
    assert_clean_fat(current)

    target, _ = current.find(target_path)
    assert target.size == len(expected_content)
    assert target.chain == expected_chain
    assert current.file_bytes(target) == expected_content
    baseline_target, _ = baseline.find(target_path)
    assert_directory_entry_preserved(
        baseline, current, baseline_target, expected_chain[0] if expected_chain else 0,
        len(expected_content),
    )

    # All non-target files/directories keep exact content and cluster ownership.
    for entry, path in baseline.walk():
        if "/".join(path) == target_path:
            continue
        current_entry, _ = current.find("/".join(path))
        assert current_entry.chain == entry.chain
        assert current_entry.attr == entry.attr
        assert current_entry.size == entry.size
        if not entry.directory:
            assert current.file_bytes(current_entry) == baseline.file_bytes(entry)


class FAT16ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp = Path(self.temp_dir.name)
        self.normal = make_normal()
        self.full = make_full()

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_image(self, name: str, fixture: bytes = None) -> Path:
        image = self.temp / name
        image.write_bytes(self.normal if fixture is None else fixture)
        return image

    def write_input(self, name: str, content: bytes) -> Path:
        path = self.temp / name
        path.write_bytes(content)
        return path

    def clean_replacement(
        self,
        image: Path,
        target: str,
        content: bytes,
        fixture: bytes,
    ):
        input_path = self.write_input(f"{image.stem}.bin", content)
        rc, result = replace(image, target, input_path)
        self.assertEqual(rc, 0, result)
        self.assertFalse((self.temp / (image.name + ".journal")).exists())
        return result

    def test_clean_grow_shrink_empty_and_plan(self):
        cases = [
            ("grow.img", "MYFILE.TXT", b"GROW-1234!" * 300, (2, 3, 4, 5, 6, 12)),
            ("shrink.img", "MYFILE.TXT", b"SHRUNK!" * 80, (2, 3)),
            ("zero.img", "MYFILE.TXT", b"", ()),
            ("empty.img", "EMPTY.TXT", b"EMPTY-NEW!" * 100, (12, 13)),
        ]
        for filename, target, content, expected_chain in cases:
            with self.subTest(filename):
                image = self.make_image(filename)
                plan_path = self.make_image(filename + ".plan")
                rc, plan = run_tool(
                    plan_path,
                    target,
                    "-i",
                    str(self.write_input(filename + ".input", content)),
                )
                self.assertEqual(rc, 0, plan)
                expected_boundaries = 2 + 4 * plan["journal_records"]
                self.assertEqual(plan["fault_boundaries"], expected_boundaries)

                result = self.clean_replacement(image, target, content, self.normal)
                self.assertEqual(
                    result["changed_sectors"],
                    changed_sectors(self.normal, image.read_bytes()),
                )
                assert_expected_image(
                    image,
                    image.read_bytes(),
                    self.normal,
                    target,
                    content,
                    expected_chain,
                    result["changed_sectors"],
                )

    def fault_loop(self, filename: str, target: str, content: bytes, expected_chain):
        baseline_image = self.make_image(filename + ".baseline")
        canonical_image = self.make_image(filename + ".canonical")
        input_path = self.write_input(filename + ".input", content)

        rc, plan = run_tool(
            baseline_image, target, "-i", str(input_path), "--plan"
        )
        self.assertEqual(rc, 0, plan)
        total = plan["fault_boundaries"]
        self.assertGreaterEqual(total, 10)

        rc, canonical_result = replace(canonical_image, target, input_path)
        self.assertEqual(rc, 0, canonical_result)
        canonical_bytes = canonical_image.read_bytes()
        self.assertEqual(canonical_result["fault_boundaries"], total)

        observed = {"old": 0, "new": 0}
        for boundary in range(1, total + 1):
            image = self.make_image(f"{filename}.fault-{boundary}.img")
            journal = Path(str(image) + ".journal")
            partial = faulted_replace(image, target, input_path, boundary)
            self.assertEqual(partial.returncode, 3)
            self.assertEqual(partial.stdout, "")
            self.assertTrue(journal.exists())

            rc, recovery = recover(image)
            self.assertEqual(rc, 0, recovery)
            version = recovery["recovered_version"]
            observed[version] += 1
            self.assertFalse(journal.exists())
            self.assertIn("sectors", recovery)
            self.assertIn("sector_kinds", recovery)

            if version == "old":
                self.assertTrue(recovery["needs_rerun"])
                assert_expected_image(
                    image, self.normal, self.normal, target,
                    (b"OLD" * 1000)[:5 * 512]
                    if target == "MYFILE.TXT"
                    else b"",
                    (2, 3, 4, 5, 6) if target == "MYFILE.TXT" else (),
                    [],
                )
                rc, rerun = replace(image, target, input_path)
                self.assertEqual(rc, 0, rerun)
            else:
                self.assertFalse(recovery["needs_rerun"])

            if target == "MYFILE.TXT":
                expected_sectors = canonical_result["changed_sectors"]
            else:
                expected_sectors = canonical_result["changed_sectors"]
            assert_expected_image(
                image,
                canonical_bytes,
                self.normal,
                target,
                content,
                expected_chain,
                expected_sectors,
            )

        self.assertGreater(observed["old"], 0)
        self.assertGreater(observed["new"], 0)
        print(
            f"{filename}: {total} torn-sector boundaries; "
            f"old={observed['old']}, new={observed['new']}, "
            f"sectors={canonical_result['changed_sectors']}"
        )

    def test_fault_boundaries_grow(self):
        self.fault_loop(
            "grow",
            "MYFILE.TXT",
            b"GROW-1234!" * 300,
            (2, 3, 4, 5, 6, 12),
        )

    def test_fault_boundaries_shrink(self):
        self.fault_loop(
            "shrink",
            "MYFILE.TXT",
            b"SHRUNK!" * 80,
            (2, 3),
        )

    def test_fault_boundaries_zero_length(self):
        self.fault_loop("zero", "MYFILE.TXT", b"", ())

    def test_fault_boundaries_empty_to_nonempty(self):
        self.fault_loop(
            "empty",
            "EMPTY.TXT",
            b"EMPTY-NEW!" * 100,
            (12, 13),
        )

    def test_insufficient_space_preserves_image(self):
        image = self.make_image("full.img", self.full)
        before = image.read_bytes()
        input_path = self.write_input("too-big.bin", b"X" * 3000)
        rc, result = replace(image, "MYFILE.TXT", input_path)
        self.assertEqual(rc, 2)
        self.assertEqual(result["visible_version"], "old")
        self.assertEqual(image.read_bytes(), before)
        self.assertFalse(Path(str(image) + ".journal").exists())

    def test_corrupt_layouts_are_rejected_without_modification(self):
        reader_path = self.make_image("reader.img", self.normal)
        reader = IndependentFat16(reader_path)

        # FAT copy mismatch.
        mismatch = bytearray(self.normal)
        fat1_start = (reader.reserved + reader.fat_size) * 512
        struct.pack_into("<H", mismatch, fat1_start + 2 * 2, 4)

        cycle = bytearray(self.normal)
        mutate_fat_pair(cycle, reader, 6, 3)

        cross = bytearray(self.normal)
        mutate_fat_pair(cross, reader, 10, 2)

        out_of_bounds = bytearray(self.normal)
        mutate_fat_pair(out_of_bounds, reader, 6, 0x7FFF)

        orphan = bytearray(self.normal)
        mutate_fat_pair(orphan, reader, 100, 0xFFFF)

        for name, broken in [
            ("mismatch", bytes(mismatch)),
            ("cycle", bytes(cycle)),
            ("cross", bytes(cross)),
            ("oob", bytes(out_of_bounds)),
            ("orphan", bytes(orphan)),
        ]:
            with self.subTest(name):
                image = self.make_image(f"{name}.img", broken)
                input_path = self.write_input(f"{name}.bin", b"new")
                rc, result = replace(image, "MYFILE.TXT", input_path)
                self.assertEqual(rc, 1, result)
                self.assertIn("error", result)
                self.assertEqual(image.read_bytes(), broken)
                self.assertFalse(Path(str(image) + ".journal").exists())

    def test_long_name_is_not_used_for_addressing(self):
        image = self.make_image("lfn.img")
        input_path = self.write_input("lfn.bin", b"new")
        rc, result = replace(image, "MyFile.txt", input_path)
        # Case-insensitive 8.3 still works; this is not long-name addressing.
        self.assertEqual(rc, 0, result)

        image = self.make_image("lfn-missing.img")
        rc, result = replace(image, "LONGFILE.TXT", input_path)
        self.assertEqual(rc, 1)
        self.assertIn("not found", result["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
