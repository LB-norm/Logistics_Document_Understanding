"""Check model selection without loading inference dependencies or checkpoints."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


LAUNCHER = Path(__file__).resolve().parents[1] / 'experiments/qwen/fusion/run_testset.py'
FINAL_RUN = 'qwen35-9b-fusion-soft-f2-bf16-full-lr5e6'


class FusionTestsetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        previous = Path.cwd()
        os.chdir(self.temporary.name)
        self.addCleanup(os.chdir, previous)
        self.root = Path('runs/qwen-fusion-compact-chunked')
        for name in [f'qwen35-completed-{i}' for i in range(14)] + [FINAL_RUN]:
            (self.root / name / 'best_model').mkdir(parents=True)
        (self.root / 'qwen35-incomplete').mkdir()

    def launch(self, *args):
        with patch.object(sys, 'argv', [str(LAUNCHER), *args]), \
                patch('subprocess.run', return_value=SimpleNamespace(returncode=0)) as child, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as stopped:
            runpy.run_path(str(LAUNCHER), run_name='__main__')
        return stopped.exception.code, child

    def test_select_final_model_without_rerunning_other_completed_models(self):
        status, child = self.launch('--run', FINAL_RUN)
        self.assertEqual(status, 0)
        child.assert_called_once()
        self.assertEqual(child.call_args.args[0][-1], str(self.root / FINAL_RUN))

    def test_all_completed_models_supports_fifteen_and_skips_incomplete_run(self):
        status, child = self.launch()
        self.assertEqual(status, 0)
        self.assertEqual(child.call_count, 15)
        selected = {Path(call.args[0][-1]).name for call in child.call_args_list}
        self.assertIn(FINAL_RUN, selected)
        self.assertNotIn('qwen35-incomplete', selected)

    def test_unknown_or_incomplete_run_does_not_start_inference(self):
        for name in ('missing', 'qwen35-incomplete'):
            with self.subTest(name=name):
                status, child = self.launch('--run', name)
                self.assertEqual(status, 2)
                child.assert_not_called()


if __name__ == '__main__':
    unittest.main()
