from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.utils.vram_tracking import PeakVramTracker, build_peak_vram_callback


class PeakVramTrackingTests(unittest.TestCase):
    def test_persists_per_device_peaks_without_resetting_between_phases(self) -> None:
        gib = 1024**3
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.device_count.return_value = 2
        cuda.get_device_name.side_effect = lambda device: f"GPU {device}"
        allocated = [3 * gib, 2 * gib]
        reserved = [4 * gib, 3 * gib]
        cuda.max_memory_allocated.side_effect = lambda device: allocated[device]
        cuda.max_memory_reserved.side_effect = lambda device: reserved[device]

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()) as output:
            tracker = PeakVramTracker(SimpleNamespace(cuda=cuda), Path(directory))
            initial = tracker.record("model_loaded", 0)
            self.assertEqual(initial["devices"][0]["peak_allocated_gib"], 3)
            allocated[0] = 7 * gib
            reserved[0] = 9 * gib
            tracker.record("training", 10)
            summary = tracker.record("completed", 20)
            persisted = json.loads((Path(directory) / "vram_usage.json").read_text())

        self.assertEqual(persisted, summary)
        self.assertEqual(summary["global_step"], 20)
        self.assertEqual(summary["devices"][0]["peak_allocated_bytes"], 7 * gib)
        self.assertEqual(summary["devices"][0]["peak_reserved_gib"], 9)
        self.assertEqual(summary["devices"][1]["peak_allocated_gib"], 2)
        self.assertEqual(cuda.reset_peak_memory_stats.call_count, 2)
        cuda.reset_peak_memory_stats.assert_any_call(0)
        cuda.reset_peak_memory_stats.assert_any_call(1)
        self.assertIn("7.000 GiB allocated, 9.000 GiB reserved", output.getvalue())

    def test_no_cuda_is_reported_as_unavailable(self) -> None:
        cuda = Mock()
        cuda.is_available.return_value = False
        with tempfile.TemporaryDirectory() as directory:
            tracker = PeakVramTracker(SimpleNamespace(cuda=cuda), Path(directory))
            summary = tracker.record("completed")
        self.assertFalse(summary["available"])
        self.assertEqual(summary["devices"], [])
        cuda.reset_peak_memory_stats.assert_not_called()
        cuda.max_memory_allocated.assert_not_called()

    def test_callback_records_logging_evaluation_and_training_end(self) -> None:
        tracker = Mock()
        callback = build_peak_vram_callback(object, tracker)
        state = SimpleNamespace(global_step=50, is_world_process_zero=True)
        callback.on_log(None, state, None, logs={"loss": 0.1})
        tracker.record.assert_called_with("training", 50)
        callback.on_evaluate(None, state, None)
        tracker.record.assert_called_with("evaluation", 50)
        callback.on_train_end(None, state, None)
        tracker.record.assert_called_with("training_finished", 50)

        tracker.reset_mock()
        state.is_world_process_zero = False
        callback.on_log(None, state, None)
        callback.on_evaluate(None, state, None)
        callback.on_train_end(None, state, None)
        tracker.record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
