from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.preprocessing.document_rectification import (
    RectificationConfig,
    detect_page_quad,
    estimate_skew,
    normalize_orientation,
    rectify_document,
    rotate_expanded,
    tesseract_orientation,
)


class DocumentRectificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RectificationConfig(detect_page=False)

    def ruled_page(self) -> np.ndarray:
        page = np.full((600, 440), 255, np.uint8)
        for y in range(60, 560, 40):
            cv2.line(page, (40, y), (400, y), 0, 2)
        return page

    def test_page_detection_and_perspective_warp(self) -> None:
        image = np.full((800, 700, 3), 30, np.uint8)
        quad = np.float32([[130, 70], [580, 130], [620, 720], [60, 680]])
        cv2.fillConvexPoly(image, quad.astype(int), (245, 245, 245))
        config = RectificationConfig(deskew=False)
        detected = detect_page_quad(image, config)
        self.assertIsNotNone(detected)
        np.testing.assert_allclose(detected, quad, atol=5)
        result = rectify_document(image, config=config, clockwise_rotation=0)
        height, width = result.image.shape[:2]
        expected = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        np.testing.assert_allclose(result.map_points(result.page_quad), expected, atol=0.001)
        self.assertGreater(np.median(result.image), 240)

    def test_scanned_internal_table_does_not_get_cropped(self) -> None:
        image = np.full((700, 500), 255, np.uint8)
        cv2.rectangle(image, (40, 50), (460, 650), 0, 3)
        self.assertIsNone(detect_page_quad(image, RectificationConfig()))

    def test_blank_image_is_preserved_and_reports_uncertainty(self) -> None:
        image = np.full((100, 80, 3), 255, np.uint8)
        result = rectify_document(image)
        np.testing.assert_array_equal(result.image, image)
        np.testing.assert_array_equal(result.transform, np.eye(3))
        self.assertEqual(len(result.warnings), 3)
        self.assertIsNone(result.clockwise_rotation)

    def test_positive_and_negative_skew_are_removed(self) -> None:
        for angle in (-11, -5, 5, 11):
            with self.subTest(angle=angle):
                skewed, _ = rotate_expanded(self.ruled_page(), angle)
                estimated = estimate_skew(skewed, self.config)
                self.assertIsNotNone(estimated)
                self.assertAlmostEqual(estimated, -angle, delta=0.6)
                result = rectify_document(skewed, config=self.config, clockwise_rotation=0)
                residual = estimate_skew(result.image, self.config)
                self.assertIsNotNone(residual)
                self.assertAlmostEqual(residual, 0, delta=0.6)

    def test_vertical_lines_support_deskew_before_orientation(self) -> None:
        sideways = np.rot90(self.ruled_page())
        skewed, _ = rotate_expanded(sideways, 7)
        estimated = estimate_skew(skewed, self.config)
        self.assertIsNotNone(estimated)
        self.assertAlmostEqual(estimated, -7, delta=0.6)

    def test_tight_skew_limit_still_accepts_aligned_lines(self) -> None:
        config = RectificationConfig(detect_page=False, max_skew_degrees=0.1)
        self.assertEqual(estimate_skew(self.ruled_page(), config), 0)

    def test_text_without_table_lines_supports_deskew(self) -> None:
        page = np.full((700, 900), 255, np.uint8)
        for y in range(60, 650, 50):
            cv2.putText(page, "Document reference and delivery address", (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, 0, 2)
        skewed, _ = rotate_expanded(page, -8)
        estimated = estimate_skew(skewed, self.config)
        self.assertIsNotNone(estimated)
        self.assertAlmostEqual(estimated, 8, delta=1)

    def test_exact_quarter_turns_and_coordinate_mapping(self) -> None:
        image = np.arange(35, dtype=np.uint8).reshape(5, 7)
        for angle in (0, 90, 180, 270):
            with self.subTest(angle=angle):
                result = rectify_document(image, config=RectificationConfig(detect_page=False, deskew=False),
                                          clockwise_rotation=angle)
                np.testing.assert_array_equal(result.image, np.rot90(image, -(angle // 90)))
                points = np.array([[0, 0], [6, 4], [3, 2]])
                mapped = result.map_points(points).astype(int)
                for source, target in zip(points, mapped):
                    self.assertEqual(image[source[1], source[0]], result.image[target[1], target[0]])
                np.testing.assert_allclose(result.map_points(mapped, inverse=True), points)

    def test_combined_transform_round_trip_and_boxes(self) -> None:
        image = self.ruled_page()
        quad = np.float32([[10, 20], [420, 10], [430, 580], [20, 590]])
        result = rectify_document(image, page_quad=quad, clockwise_rotation=90)
        points = np.array([[50, 60], [300, 400], [400, 500]])
        np.testing.assert_allclose(result.map_points(result.map_points(points), inverse=True), points, atol=1e-8)
        box = np.array([[50, 60, 300, 400]])
        corners = np.array([[[50, 60], [300, 60], [300, 400], [50, 400]]])
        np.testing.assert_allclose(result.map_boxes(box), result.map_points(corners))
        np.testing.assert_allclose(result.map_points(result.map_boxes(box), inverse=True), corners, atol=1e-8)
        self.assertEqual(result.map_boxes(np.empty((0, 4))).shape, (0, 4, 2))

    def test_rotation_canvas_contains_all_original_corners(self) -> None:
        image = np.zeros((80, 120), np.uint8)
        rotated, matrix = rotate_expanded(image, 13)
        corners = np.array([[0, 0, 1], [119, 0, 1], [119, 79, 1], [0, 79, 1]]) @ matrix.T
        self.assertGreaterEqual(corners[:, :2].min(), -1e-9)
        self.assertLessEqual(corners[:, 0].max(), rotated.shape[1] - 1)
        self.assertLessEqual(corners[:, 1].max(), rotated.shape[0] - 1)

    def test_orientation_detector_receives_deskewed_page_and_override_wins(self) -> None:
        skewed, _ = rotate_expanded(self.ruled_page(), 8)
        received = []

        def detector(image: np.ndarray) -> int:
            received.append(estimate_skew(image, self.config))
            return 180

        result = rectify_document(skewed, config=self.config, orientation_detector=detector)
        self.assertEqual(result.clockwise_rotation, 180)
        self.assertAlmostEqual(received[0], 0, delta=0.6)
        rectify_document(skewed, config=self.config, orientation_detector=detector, clockwise_rotation=0)
        self.assertEqual(len(received), 1)

    def test_missing_tesseract_abstains(self) -> None:
        with patch("src.preprocessing.document_rectification.shutil.which", return_value=None):
            self.assertIsNone(tesseract_orientation(self.ruled_page()))

    def test_tesseract_confidence_and_clockwise_contract(self) -> None:
        import subprocess

        with patch("src.preprocessing.document_rectification.shutil.which", return_value="tesseract"), \
             patch("src.preprocessing.document_rectification.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, b"Rotate: 270\nOrientation confidence: 20.0\n", b"")
            self.assertEqual(tesseract_orientation(self.ruled_page()), 270)
            run.return_value = subprocess.CompletedProcess([], 0, b"Rotate: 180\nOrientation confidence: 2.0\n", b"")
            self.assertIsNone(tesseract_orientation(self.ruled_page()))

    def test_invalid_inputs_fail_clearly(self) -> None:
        with self.assertRaises(ValueError):
            rectify_document(np.ones((20, 20), dtype=float))
        with self.assertRaises(ValueError):
            rectify_document(np.ones((20, 20, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            rectify_document(self.ruled_page(), page_quad=np.zeros((4, 2)))
        with self.assertRaises(ValueError):
            rectify_document(self.ruled_page(), page_quad=np.float32([[0, 0], [1000, 0], [1000, 500], [0, 500]]))
        with self.assertRaises(ValueError):
            normalize_orientation(self.ruled_page(), 45)
        with self.assertRaises(ValueError):
            RectificationConfig(max_skew_degrees=45)


if __name__ == "__main__":
    unittest.main()
