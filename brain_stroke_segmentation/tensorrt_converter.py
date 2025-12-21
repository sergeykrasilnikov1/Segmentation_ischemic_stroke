"""TensorRT conversion utilities."""

import subprocess
from pathlib import Path


def convert_to_tensorrt(
    onnx_path: Path | str,
    output_path: Path | str,
    precision: str = "fp16",
    workspace_size: int = 4096,
) -> None:
    """
    Convert ONNX model to TensorRT format.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TensorRT engine
        precision: Precision mode (fp32, fp16, int8)
        workspace_size: Workspace size in MB
    """
    onnx_path = Path(onnx_path)
    output_path = Path(output_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--{precision}",
        f"--workspace={workspace_size}",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"TensorRT conversion successful. Engine saved to {output_path}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"TensorRT conversion failed: {e}")
        print(e.stderr)
        raise
    except FileNotFoundError:
        print("trtexec not found. Please install TensorRT and ensure trtexec is in your PATH.")
        print("Alternatively, you can use the Python TensorRT API.")
        raise
