"""
Use this script to convert an ONNX checkpoint to .pt
"""

import argparse

import torch

from inswapper import InswapperModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conversion script for inswapper_128.onnx checkpoint to PyTorch."
    )

    parser.add_argument(
        "onnx_checkpoint", help="Path to the inswapper_128.onnx checkpoint to convert."
    )
    parser.add_argument(
        "save_path", help="Path to save the converted PyTorch checkpoint."
    )

    args = parser.parse_args()

    model = InswapperModel(onnx_checkpoint=args.onnx_checkpoint)
    torch.save(model, args.save_path)
