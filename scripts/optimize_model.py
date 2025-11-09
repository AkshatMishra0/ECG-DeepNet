#!/usr/bin/env python3
"""Optimize PyTorch model checkpoints.

Safe operations included:
- strip: remove optimizer/training state and keep only model state_dict and minimal metadata
- torchscript: load model, trace with a dummy input and save TorchScript artifact

Usage examples:
  python scripts/optimize_model.py --action strip --input saved_models/best_model.pth --output optimized/best_model_stripped.pth
  python scripts/optimize_model.py --action torchscript --input saved_models/best_model.pth --output optimized/best_model_ts.pt --model ecgcnn --input-length 1000 --num-classes 5

This script is defensive: it tries to import model classes from `src.models.ecg_cnn` and will fail with a clear message if imports are broken.
"""
import argparse
import os
import sys
import time
from pathlib import Path

try:
    import torch
except Exception as e:
    print("ERROR: PyTorch is required to run this script. Install it with: pip install torch")
    raise


def add_repo_root_to_path():
    # Ensure repository root is in sys.path so src.models imports work
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location='cpu')
    return ckpt


def extract_state_dict(ckpt):
    # Many training scripts save a dict with keys like 'state_dict' or 'model_state'
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
        # try to detect a state_dict by checking tensor values
        keys = list(ckpt.keys())
        if keys and isinstance(ckpt[keys[0]], torch.Tensor):
            return ckpt
    # fallback: assume ckpt itself is a state_dict-like object
    return ckpt


def action_strip(input_path, output_path):
    ckpt = load_checkpoint(input_path)
    state = extract_state_dict(ckpt)
    minimal = {
        'state_dict': state,
        'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(minimal, str(outp))
    print(f"Wrote stripped checkpoint to: {outp} (size: {outp.stat().st_size/1024:.1f} KB)")


def action_torchscript(input_path, output_path, model_name, input_length, num_classes):
    add_repo_root_to_path()
    # import model class
    try:
        from src.models.ecg_cnn import ECGCNN, ECGLSTM
    except Exception as e:
        raise RuntimeError(f"Failed to import model classes from src.models.ecg_cnn: {e}")

    ckpt = load_checkpoint(input_path)
    state = extract_state_dict(ckpt)

    if model_name.lower() == 'ecgcnn' or 'cnn' in model_name.lower():
        model = ECGCNN(input_length=input_length, num_classes=num_classes)
    elif model_name.lower() in ('ecglstm', 'lstm'):
        model = ECGLSTM(num_classes=num_classes)
    else:
        raise ValueError('Unsupported model name. Use "ecgcnn" or "ecglstm"')

    # load weights
    try:
        model.load_state_dict(state)
    except Exception:
        # try to remove potential "module." prefixes (from DataParallel)
        cleaned = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(cleaned)

    model.eval()

    # create dummy input for tracing: shape (1, 12, input_length)
    dummy = torch.randn(1, 12, input_length)
    try:
        traced = torch.jit.trace(model, dummy)
    except Exception as e:
        raise RuntimeError(f"TorchScript tracing failed: {e}")

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(outp))
    print(f"Saved TorchScript model to: {outp} (size: {outp.stat().st_size/1024:.1f} KB)")


def main():
    p = argparse.ArgumentParser(description='Optimize PyTorch model checkpoints (strip / torchscript)')
    p.add_argument('--action', choices=['strip', 'torchscript'], required=True,
                   help='Optimization action to perform')
    p.add_argument('--input', required=True, help='Input checkpoint path')
    p.add_argument('--output', required=True, help='Output path for optimized artifact')
    p.add_argument('--model', default='ecgcnn', help='Model architecture name (ecgcnn or ecglstm)')
    p.add_argument('--input-length', type=int, default=1000, help='Input length (samples) for tracing')
    p.add_argument('--num-classes', type=int, default=5, help='Number of output classes')

    args = p.parse_args()

    if args.action == 'strip':
        action_strip(args.input, args.output)
    elif args.action == 'torchscript':
        action_torchscript(args.input, args.output, args.model, args.input_length, args.num_classes)


if __name__ == '__main__':
    main()
