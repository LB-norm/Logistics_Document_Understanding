"""Run completed compact-fusion models on the 30 test images.

Execute from the remote repository root using its Python environment.
Uses saved compact prompts and training precision; writes predictions and
diagnostics with GPU OCR. Override the layout model with
QWEN_FUSION_YOLO_MODEL_DIR if needed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', metavar='NAME', help='Infer only this completed run; defaults to all completed runs')
options = parser.parse_args()
runs_root = Path('runs/qwen-fusion-compact-chunked')
runs = sorted(p for p in runs_root.glob('qwen35-*') if (p / 'best_model').is_dir())
if options.run is not None:
    runs = [run for run in runs if run.name == options.run]
if not runs:
    parser.error(f'No completed best_model found for {options.run or "any run"} under {runs_root}')
worker = r'''
import json
import os
import sys
from pathlib import Path
from src.Qwen import run_inference as qwen
from src.fusion import to_llm_text
from src.fusion.compact import compact_document
from src.fusion.compact_sft import COMPACT_MARKER
from src.inference_pipeline.__main__ import main

run = Path(sys.argv[1])
cfg = json.loads((run / 'training_config.json').read_text())
data = Path(cfg['dataset_root'])
with (data / cfg['train_file']).open() as f:
    example = json.loads(next(f))
settings = json.loads((data / 'preparation_summary.json').read_text())
system = example['messages'][0]['content']
prefix, marker, _ = example['messages'][1]['content'].partition(COMPACT_MARKER)
assert marker, 'Expected compact fusion training prompts'

def compact_messages(system_prompt, user_prompt, document, template, schema):
    text = prefix + COMPACT_MARKER + to_llm_text(
        compact_document(document, settings['decimals']))
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': system}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': text}]},
    ]

qwen.build_fusion_messages = compact_messages
images = sorted(Path('data/datasets/250_CMRS_240dpi_20260707/test/images').rglob('*.png'))
assert len(images) == 30, f'Expected 30 test images, found {len(images)}'
ocr = Path('output/ocr_v6_model_cache/official_models')
args = [
    '--mode', 'parser', '--model-id', str(run / 'best_model'),
    '--fusion-mode', example['fusion_mode'],
    '--yolo-model-dir', os.environ.get('QWEN_FUSION_YOLO_MODEL_DIR',
        'runs/yolo_v11s/cmr_100epochs/best_model'),
    '--ocr-detection-model-dir', str(ocr / 'PP-OCRv6_medium_det'),
    '--ocr-recognition-model-dir', str(ocr / 'PP-OCRv6_medium_rec'),
    '--ocr-device', 'gpu:0',
    '--schema-path', str(data / 'schema.json'),
    '--template-path', str(data / 'template.json'),
    '--annotation-target-key', 'root',
    '--compute-dtype', cfg['compute_dtype'],
    '--load-in-4bit' if cfg['load_in_4bit'] else '--no-load-in-4bit',
    '--max-new-tokens', '2048', '--local-files-only',
    '--output-dir', str(Path('output/qwen-fusion-compact-chunked-test') / run.name),
]
if cfg.get('cache_dir'):
    args += ['--cache-dir', cfg['cache_dir']]
raise SystemExit(main(args + ['--images', *map(str, images)]))
'''
failed = []
for index, run in enumerate(runs, 1):
    print(f'[{index}/{len(runs)}] {run.name}', flush=True)
    result = subprocess.run([sys.executable, '-u', '-c', worker, str(run)])
    if result.returncode:
        failed.append(run.name)
print('Runs reporting errors:', failed or 'none')
raise SystemExit(bool(failed))
