"""ONNX conversion utilities."""

import torch
import onnx
from pathlib import Path

from brain_stroke_segmentation.lightning_module import StrokeSegmentationModule


def convert_to_onnx(
    model_path: Path | str,
    output_path: Path | str,
    img_height: int = 256,
    img_width: int = 256,
    encoder_name: str = "efficientnet-b4",
    opset_version: int = 13,
) -> None:
    """
    Convert PyTorch Lightning model to ONNX format.

    Args:
        model_path: Path to Lightning checkpoint or PyTorch model weights
        output_path: Path to save ONNX model
        img_height: Input image height
        img_width: Input image width
        encoder_name: Encoder name used in training
        opset_version: ONNX opset version
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    try:
        model = StrokeSegmentationModule.load_from_checkpoint(
            str(model_path), encoder_name=encoder_name
        )
        model = model.model
        model.eval()
    except Exception:
        from brain_stroke_segmentation.model import build_model

        model = build_model(encoder_name=encoder_name)
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()

    dummy_input = torch.randn(1, 3, img_height, img_width)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved to {output_path}")

