"""ONNX conversion utilities."""

from pathlib import Path

import onnx
import torch

from brain_stroke_segmentation.model import build_model


def convert_to_onnx(
    model_path: Path | str,
    output_path: Path | str,
    img_height: int = 256,
    img_width: int = 256,
    encoder_name: str = "efficientnet-b4",
    opset_version: int = 18,
) -> None:
    """
    Convert PyTorch model to ONNX format.

    Args:
        model_path: Path to PyTorch model weights (.pth file)
        output_path: Path to save ONNX model
        img_height: Input image height
        img_width: Input image width
        encoder_name: Encoder name used in training
        opset_version: ONNX opset version
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    model = build_model(encoder_name=encoder_name)
    state_dict = torch.load(model_path, map_location="cpu")
    # Handle both .pth (state_dict) and legacy .ckpt (Lightning format)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        # Legacy Lightning checkpoint format
        state_dict = state_dict["state_dict"]
    # Remove 'model.' prefix if present
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {
            k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
        }
    model.load_state_dict(state_dict)
    model.eval()

    model = model.cpu()
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
