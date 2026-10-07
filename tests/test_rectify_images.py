from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.preprocessing.document_rectification import RectificationConfig, rectify_document
from src.preprocessing.rectify_images import main, rectify_file, rectify_folder


class RectifyImagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        self.config = RectificationConfig(detect_page=False, deskew=False)
        self.image = np.arange(120, dtype=np.uint8).reshape(10, 12)

    def write_image(self, relative: str) -> Path:
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(path), self.image))
        return path

    def test_single_image_returns_model_input_without_disk_io(self) -> None:
        original = self.image.copy()
        with patch("cv2.imread", side_effect=AssertionError("Unexpected file read")), \
             patch("cv2.imwrite", side_effect=AssertionError("Unexpected file write")):
            result = rectify_document(self.image, config=self.config, clockwise_rotation=90)
        np.testing.assert_array_equal(result.image, np.rot90(original, -1))
        np.testing.assert_array_equal(self.image, original)

    def test_recursive_folder_preserves_structure_and_shared_pipeline_results(self) -> None:
        first = self.write_image("page.png")
        second = self.write_image("nested/page.BMP")
        (self.source / "notes.txt").write_text("not an image", encoding="utf-8")
        originals = {path: path.read_bytes() for path in (first, second)}
        manifest = rectify_folder(self.source, self.output, recursive=True,
                                  config=self.config, clockwise_rotation=90)
        self.assertEqual((manifest["total"], manifest["succeeded"], manifest["failed"]), (2, 2, 0))
        for source, target in ((first, self.output / "page.png"), (second, self.output / "nested/page.png")):
            expected = rectify_document(cv2.imread(str(source)), config=self.config, clockwise_rotation=90)
            np.testing.assert_array_equal(cv2.imread(str(target)), expected.image)
            metadata = json.loads(target.with_suffix(".png.json").read_text())
            np.testing.assert_allclose(metadata["transform"], expected.transform)
            self.assertEqual(metadata["clockwise_rotation"], 90)
            self.assertEqual(source.read_bytes(), originals[source])
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), manifest)

    def test_flat_folder_ignores_nested_images(self) -> None:
        self.write_image("page.png")
        self.write_image("nested/other.png")
        manifest = rectify_folder(self.source, self.output, config=self.config)
        self.assertEqual(manifest["total"], 1)
        self.assertFalse((self.output / "nested").exists())
        self.assertTrue(manifest["records"][0]["warnings"])

    def test_corrupt_image_does_not_block_remaining_images(self) -> None:
        (self.source / "a_corrupt.jpg").write_bytes(b"broken image")
        self.write_image("b_good.png")
        manifest = rectify_folder(self.source, self.output, config=self.config)
        self.assertEqual(manifest["succeeded"], 1)
        self.assertEqual(manifest["failed"], 1)
        self.assertEqual(manifest["records"][0]["status"], "error")
        self.assertIn("Cannot read image", manifest["records"][0]["error"])
        self.assertTrue((self.output / "b_good.png").is_file())

    def test_filename_collisions_are_rejected_before_writing(self) -> None:
        self.write_image("same.png")
        self.write_image("same.jpg")
        with self.assertRaisesRegex(ValueError, "collision"):
            rectify_folder(self.source, self.output, config=self.config)
        self.assertFalse(self.output.exists())

    def test_existing_outputs_require_explicit_overwrite(self) -> None:
        self.write_image("page.png")
        rectify_folder(self.source, self.output, config=self.config, clockwise_rotation=0)
        before = (self.output / "page.png").read_bytes()
        with self.assertRaises(FileExistsError):
            rectify_folder(self.source, self.output, config=self.config, clockwise_rotation=90)
        self.assertEqual((self.output / "page.png").read_bytes(), before)
        manifest = rectify_folder(self.source, self.output, config=self.config,
                                  clockwise_rotation=90, overwrite=True)
        self.assertEqual(manifest["succeeded"], 1)
        self.assertEqual(cv2.imread(str(self.output / "page.png")).shape[:2], (12, 10))

    def test_source_paths_and_overlapping_folders_are_rejected(self) -> None:
        source = self.write_image("page.png")
        for target in (self.source, self.source / "rectified", self.root):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "non-overlapping"):
                rectify_folder(self.source, target, overwrite=True)
        with self.assertRaisesRegex(ValueError, "preserve the source"):
            rectify_file(source, source, overwrite=True)

    def test_empty_folder_and_non_folder_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "No supported images"):
            rectify_folder(self.source, self.output)
        with self.assertRaises(NotADirectoryError):
            rectify_folder(self.source / "missing", self.output)

    def test_cli_handles_both_file_and_folder_and_returns_failure_status(self) -> None:
        source = self.write_image("page.png")
        flags = ["--orientation", "90", "--no-page-detection", "--no-deskew"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(source), str(self.root / "single.png"), *flags]), 0)
            self.assertEqual(main([str(self.source), str(self.output), "--recursive", *flags]), 0)
            (self.source / "corrupt.png").write_bytes(b"not an image")
            self.assertEqual(main([str(self.source), str(self.output), "--overwrite", *flags]), 1)
        self.assertTrue((self.root / "single.png.json").is_file())
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["failed"], 1)


if __name__ == "__main__":
    unittest.main()
