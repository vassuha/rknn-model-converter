"""Convert smile checkpoints or static ONNX models to RKNN."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def toolkit_version() -> str:
    try:
        return importlib.metadata.version("rknn-toolkit2")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def make_output_dir(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def compile_onnx_to_rknn(onnx_path: Path, rknn_path: Path, target: str) -> None:
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "RKNN-Toolkit2 is missing. Install the version compatible with the "
            "RKNN Lite/runtime used by the application."
        ) from exc

    compiler = RKNN(verbose=True)
    try:
        result = compiler.config(target_platform=target, float_dtype="float16")
        if result != 0:
            raise RuntimeError(f"RKNN config failed with code {result}")

        result = compiler.load_onnx(model=str(onnx_path))
        if result != 0:
            raise RuntimeError(f"Could not load ONNX model (code {result}): {onnx_path}")

        # Start with FP16. INT8 calibration is deliberately a separate decision
        # because quantization can reduce smile recall.
        result = compiler.build(do_quantization=False)
        if result != 0:
            raise RuntimeError(f"RKNN build failed with code {result}")

        result = compiler.export_rknn(str(rknn_path))
        if result != 0:
            raise RuntimeError(f"Could not export RKNN model (code {result})")
    finally:
        compiler.release()


def load_smile_checkpoint(checkpoint_path: Path) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no 'model_state_dict' field")

    model_name = checkpoint.get("model_name", "enet_b0_8_best_vgaf")
    recognizer = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device="cpu")
    model = recognizer.model

    if not hasattr(model, "classifier"):
        raise ValueError(f"Base model {model_name!r} has no classifier to replace")

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def export_static_onnx(model: Any, onnx_path: Path, img_size: int) -> None:
    import torch

    example = torch.randn(1, 3, img_size, img_size)
    try:
        torch.onnx.export(
            model,
            example,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
    except TypeError:
        # Compatibility with PyTorch releases that predate the dynamo switch.
        torch.onnx.export(
            model,
            example,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )


def validate_onnx(model: Any, onnx_path: Path, img_size: int, tests: int = 20) -> float:
    import numpy as np
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    max_diff = 0.0

    for _ in range(tests):
        sample = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            expected = torch.sigmoid(model(sample)).cpu().numpy()
        actual_logits = session.run(None, {input_name: sample.numpy()})[0]
        actual = 1.0 / (1.0 + np.exp(-np.clip(actual_logits, -80.0, 80.0)))
        max_diff = max(max_diff, float(np.max(np.abs(expected - actual))))

    if max_diff >= 1e-3:
        raise RuntimeError(f"PyTorch/ONNX check failed: max probability difference {max_diff:.6g}")
    return max_diff


def convert_smile_checkpoint(checkpoint_path: Path, output_dir: Path, target: str) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_dir = make_output_dir(output_dir)

    model, checkpoint = load_smile_checkpoint(checkpoint_path)
    img_size = int(checkpoint.get("img_size", 224))
    if img_size <= 0:
        raise ValueError(f"Invalid checkpoint image size: {img_size}")

    onnx_name = f"{checkpoint_path.stem}.onnx"
    rknn_name = f"{checkpoint_path.stem}.rknn"

    with tempfile.TemporaryDirectory(prefix="smile-rknn-") as temp_dir:
        temp = Path(temp_dir)
        onnx_tmp = temp / onnx_name
        rknn_tmp = temp / rknn_name
        export_static_onnx(model, onnx_tmp, img_size)
        max_diff = validate_onnx(model, onnx_tmp, img_size)
        compile_onnx_to_rknn(onnx_tmp, rknn_tmp, target)
        onnx_final = output_dir / onnx_name
        rknn_final = output_dir / rknn_name
        shutil.copy2(onnx_tmp, onnx_final)
        shutil.copy2(rknn_tmp, rknn_final)

    mean = [float(value) for value in checkpoint.get("mean", [0.485, 0.456, 0.406])]
    std = [float(value) for value in checkpoint.get("std", [0.229, 0.224, 0.225])]
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("Checkpoint mean/std must have 3 values; std values must be positive")

    manifest = {
        "source_checkpoint": checkpoint_path.name,
        "source_sha256": sha256(checkpoint_path),
        "target_platform": target,
        "precision": "float16",
        "rknn_toolkit2_version": toolkit_version(),
        "model_name": checkpoint.get("model_name", "enet_b0_8_best_vgaf"),
        "output_type": "binary_logit",
        "img_size": img_size,
        "onnx_probability_max_abs_difference": max_diff,
        "mean": mean,
        "std": std,
        "quality_validation": "Graph parity only; validate recall on labeled faces and the target board.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config = (
        "MODELS:\n"
        "  smile_detection_mode: 'rknn'\n"
        f"  smile_model_name: '{rknn_name}'\n"
        f"  smile_model_rknn_img_size: {img_size}\n"
        "  smile_model_rknn_output_type: 'binary_logit'\n"
        f"  smile_model_rknn_mean: {json.dumps(mean)}\n"
        f"  smile_model_rknn_std: {json.dumps(std)}\n"
    )
    (output_dir / "config-snippet.yaml").write_text(config, encoding="utf-8")
    return output_dir


def convert_onnx_file(onnx_path: Path, output_path: Path, target: str) -> Path:
    onnx_path = onnx_path.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compile_onnx_to_rknn(onnx_path, output_path, target)
    manifest = {
        "source_onnx": onnx_path.name,
        "source_sha256": sha256(onnx_path),
        "target_platform": target,
        "precision": "float16",
        "rknn_toolkit2_version": toolkit_version(),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smile = commands.add_parser("smile", help="Convert a fine-tuned binary smile .pth checkpoint")
    smile.add_argument("--checkpoint", required=True, type=Path)
    smile.add_argument("--output-dir", required=True, type=Path)
    smile.add_argument("--target-platform", default="rk3588")

    onnx = commands.add_parser("onnx", help="Convert a static ONNX model to RKNN")
    onnx.add_argument("--input", required=True, type=Path)
    onnx.add_argument("--output", required=True, type=Path)
    onnx.add_argument("--target-platform", default="rk3588")

    args = parser.parse_args()
    try:
        if args.command == "smile":
            result = convert_smile_checkpoint(args.checkpoint, args.output_dir, args.target_platform)
            print(f"Release package created: {result}")
        else:
            result = convert_onnx_file(args.input, args.output, args.target_platform)
            print(f"RKNN model created: {result}")
    except Exception as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")


if __name__ == "__main__":
    main()
