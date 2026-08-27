from __future__ import annotations

import json
import unittest
from pathlib import Path

from foldcrack_qc.registry import default_registry_path, load_registry
from foldcrack_qc.resources import resource_path


class PackagedResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = Path(__file__).resolve().parents[1]

    def test_default_registry_is_packaged_and_matches_audited_mirror(self) -> None:
        packaged = default_registry_path()
        mirror = self.repository_root / "configs" / "datasets.json"

        self.assertTrue(packaged.is_file())
        self.assertEqual(packaged.read_bytes(), mirror.read_bytes())
        self.assertEqual(
            load_registry(), json.loads(mirror.read_text(encoding="utf-8"))
        )

    def test_public_lock_manifests_match_audited_mirrors(self) -> None:
        packaged_root = resource_path("public_data")
        mirror_root = self.repository_root / "configs" / "public_data"
        packaged_names = sorted(path.name for path in packaged_root.glob("*.json"))
        mirror_names = sorted(path.name for path in mirror_root.glob("*.json"))

        self.assertEqual(packaged_names, mirror_names)
        for name in mirror_names:
            with self.subTest(name=name):
                self.assertEqual(
                    (packaged_root / name).read_bytes(),
                    (mirror_root / name).read_bytes(),
                )

    def test_resource_paths_cannot_escape_the_package(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside the package"):
            resource_path("..", "registry.py")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
