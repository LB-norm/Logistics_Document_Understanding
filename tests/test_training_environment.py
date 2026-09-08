from __future__ import annotations

import unittest

from src.training_environment import (
    DEFAULT_PYTORCH_CUDA_ALLOC_CONF,
    configure_pytorch_cuda_allocator,
)


class TrainingEnvironmentTests(unittest.TestCase):
    def test_expandable_segments_are_enabled_by_default(self) -> None:
        environment: dict[str, str] = {}

        configured = configure_pytorch_cuda_allocator(environment)

        self.assertEqual(configured, DEFAULT_PYTORCH_CUDA_ALLOC_CONF)
        self.assertEqual(
            environment["PYTORCH_CUDA_ALLOC_CONF"],
            "expandable_segments:True",
        )

    def test_explicit_allocator_configuration_is_preserved(self) -> None:
        environment = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"}

        configured = configure_pytorch_cuda_allocator(environment)

        self.assertEqual(configured, "max_split_size_mb:256")
        self.assertEqual(
            environment["PYTORCH_CUDA_ALLOC_CONF"],
            "max_split_size_mb:256",
        )


if __name__ == "__main__":
    unittest.main()
