import argparse

import numpy as np
import onnxruntime
import torch
from onnx2torch import convert

from inswapper import InswapperModel

TARGET_SHAPE = (1, 3, 128, 128)
SOURCE_SHAPE = (1, 512)


def compare_onnx_and_pytorch(
    onnx_path, pytorch_model, random_seed=42, device="cuda", cached_onnx_outputs=None
):
    # Set seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Create dummy inputs
    np_input_1 = np.load("data/blob.npy").astype(np.float32)
    np_input_2 = np.load("data/latent.npy").astype(np.float32)

    if cached_onnx_outputs is not None:
        onnx_outputs = np.load(cached_onnx_outputs)
    else:
        # Load ONNX model with GPU provider
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)

        # Run ONNX model
        onnx_outputs = onnx_session.run(
            None, {"target": np_input_1, "source": np_input_2}
        )

    # Convert to PyTorch tensor and move to GPU
    torch_input_1 = torch.from_numpy(np_input_1).to(device)
    torch_input_2 = torch.from_numpy(np_input_2).to(device)

    # Run PyTorch model
    with torch.no_grad():
        pytorch_outputs = pytorch_model(torch_input_1, torch_input_2)

        if not isinstance(pytorch_outputs, tuple):
            pytorch_outputs = [pytorch_outputs]

    # Compare outputs
    print("\nComparing outputs:")
    for i, (onnx_out, pytorch_out) in enumerate(zip(onnx_outputs, pytorch_outputs)):
        # Move PyTorch output to CPU for comparison
        pytorch_np = pytorch_out.cpu().numpy()
        max_diff = np.max(np.abs(onnx_out - pytorch_np))
        mean_diff = np.mean(np.abs(onnx_out - pytorch_np))

        print(f"\nOutput {i}:")
        print(f"Shapes - ONNX: {onnx_out.shape}, PyTorch: {pytorch_np.shape}")
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"Outputs match within tolerance: {max_diff < 1e-3}")

        if max_diff >= 1e-3:
            print("\nSample differences:")
            for j in range(min(5, onnx_out.size)):
                print(
                    f"Index {j}: ONNX={onnx_out.flatten()[j]}, PyTorch={pytorch_np.flatten()[j]}"
                )

    return onnx_outputs, pytorch_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx_path", type=str, default="models/inswapper_128.onnx")
    parser.add_argument("--cached_onnx_outputs", type=str)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pytorch_model = InswapperModel(onnx_checkpoint=args.onnx_path)
    pytorch_model.to(device)
    pytorch_model.eval()

    onnx_outputs, pytorch_outputs = compare_onnx_and_pytorch(
        args.onnx_path,
        pytorch_model,
        device=device,
        cached_onnx_outputs=args.cached_onnx_outputs,
    )
