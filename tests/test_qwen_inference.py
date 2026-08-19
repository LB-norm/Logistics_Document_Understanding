from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.Qwen.run_inference import (
    ImageInferenceResult,
    InferenceRuntime,
    fill_from_template,
    main,
    parse_args,
    resolve_model_id,
    resolve_processor_source,
)


class QwenInferenceTests(unittest.TestCase):
    def test_filled_prediction_keeps_template_shape(self) -> None:
        template = {
            "sender": {"name": None, "city": None},
            "items": [{"description": None, "quantity": None}],
        }
        prediction = {
            "sender": {"name": "Example GmbH", "unexpected": "discard me"},
            "items": [
                {"description": "Pallet", "quantity": 2, "unexpected": True},
                {"description": "Carton"},
            ],
            "unexpected": "discard me",
        }

        filled = fill_from_template(template, prediction)

        self.assertEqual(
            filled,
            {
                "sender": {"name": "Example GmbH", "city": None},
                "items": [
                    {"description": "Pallet", "quantity": 2},
                    {"description": "Carton", "quantity": None},
                ],
            },
        )

    def test_adapter_metadata_selects_base_model_and_saved_processor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_path = Path(temp_dir)
            (adapter_path / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B"}),
                encoding="utf-8",
            )
            (adapter_path / "processor_config.json").write_text("{}", encoding="utf-8")
            args = parse_args(["--adapter-path", str(adapter_path)])

            model_id = resolve_model_id(args)

            self.assertEqual(model_id, "Qwen/Qwen3.5-4B")
            self.assertEqual(resolve_processor_source(args, model_id), str(adapter_path))

    def test_multiple_images_load_once_and_write_one_filled_json_each(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_paths = [root / "document_1.png", root / "document_2.jpg"]
            for image_path in image_paths:
                image_path.write_bytes(b"not decoded because generation is mocked")

            template_path = root / "template.json"
            schema_path = root / "schema.json"
            output_dir = root / "predictions"
            template_path.write_text('{"value": null}', encoding="utf-8")
            schema_path.write_text(
                json.dumps(
                    {
                        "type": "object",
                        "required": ["value"],
                        "additionalProperties": False,
                        "properties": {"value": {"type": ["string", "null"]}},
                    }
                ),
                encoding="utf-8",
            )
            runtime = InferenceRuntime(
                torch=None,
                image_module=None,
                processor=None,
                model=None,
                model_id="Qwen/Qwen3.5-4B",
                processor_source="saved-processor",
            )

            def fake_generate(runtime, image_path, template, target_schema, args):
                raw_prediction = {"value": image_path.stem}
                return ImageInferenceResult(
                    image_path=image_path,
                    prediction=fill_from_template(template, raw_prediction),
                    raw_text=json.dumps(raw_prediction),
                    cleaned_text=json.dumps(raw_prediction),
                    json_candidate=json.dumps(raw_prediction),
                    raw_prediction=raw_prediction,
                    notes=[],
                    schema_errors=[],
                )

            argv = [
                "--image-paths",
                *(str(path) for path in image_paths),
                "--schema-path",
                str(schema_path),
                "--template-path",
                str(template_path),
                "--output-dir",
                str(output_dir),
            ]
            with (
                patch(
                    "src.Qwen.run_inference.load_inference_runtime",
                    return_value=runtime,
                ) as load_runtime,
                patch(
                    "src.Qwen.run_inference.generate_image_prediction",
                    side_effect=fake_generate,
                ) as generate,
            ):
                exit_code = main(argv)

            self.assertEqual(exit_code, 0)
            load_runtime.assert_called_once()
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(
                json.loads((output_dir / "document_1.json").read_text(encoding="utf-8")),
                {"value": "document_1"},
            )
            self.assertEqual(
                json.loads((output_dir / "document_2.json").read_text(encoding="utf-8")),
                {"value": "document_2"},
            )
            manifest_lines = (output_dir / "inference_manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            manifest = [json.loads(line) for line in manifest_lines]
            self.assertEqual([record["status"] for record in manifest], ["ok", "ok"])
            self.assertEqual(
                [Path(record["prediction_path"]).name for record in manifest],
                ["document_1.json", "document_2.json"],
            )

    def test_single_image_primary_output_contains_only_filled_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "document.png"
            image_path.write_bytes(b"mock image")
            template_path = root / "template.json"
            schema_path = root / "schema.json"
            output_path = root / "prediction.json"
            template_path.write_text('{"value": null}', encoding="utf-8")
            schema_path.write_text(
                '{"type":"object","required":["value"],"properties":{"value":{}}}',
                encoding="utf-8",
            )
            runtime = InferenceRuntime(None, None, None, None, "model", "processor")
            result = ImageInferenceResult(
                image_path=image_path,
                prediction={"value": "extracted"},
                raw_text='{"value":"extracted"}',
                cleaned_text='{"value":"extracted"}',
                json_candidate='{"value":"extracted"}',
                raw_prediction={"value": "extracted"},
                notes=[],
                schema_errors=[],
            )

            with (
                patch("src.Qwen.run_inference.load_inference_runtime", return_value=runtime),
                patch("src.Qwen.run_inference.generate_image_prediction", return_value=result),
            ):
                exit_code = main(
                    [
                        "--image-path",
                        str(image_path),
                        "--schema-path",
                        str(schema_path),
                        "--template-path",
                        str(template_path),
                        "--output-path",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), {"value": "extracted"})
            diagnostics = json.loads(
                output_path.with_suffix(".json.diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostics["status"], "ok")
            self.assertIn("raw_text", diagnostics)


if __name__ == "__main__":
    unittest.main()
