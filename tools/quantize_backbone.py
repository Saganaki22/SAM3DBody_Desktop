#!/usr/bin/env python3
"""
tools/quantize_backbone.py  —  run-once int8 quantization for backbone.onnx

The backbone is a DINOv2-ViT-H/14 variant (~4.8 GB FP32 with external data).
INT8 quantization cuts it to ~2.4 GB and speeds up CPU MatMul by ~1.5–2×.
CUDA Tensor Core acceleration (sm_75+, Turing+) also benefits from int8 weights.

Expected results on ORT CUDA EP
--------------------------------
  ORT's CUDA EP implements MatMulInteger by dequantizing INT8 weights to FP32
  at runtime, then running a standard FP32 GEMM.  Inference speed is therefore
  unchanged vs FP32 (~46 ms/frame on tested hardware).

  The concrete benefit is a 3× reduction in model size:
    backbone.onnx + .data  →  4.8 GB FP32
    backbone_int8.onnx + .data  →  1.6 GB INT8
  This frees ~3 GB of VRAM per inference session, leaving more headroom for
  larger batches or other resident models.

  For real INT8 Tensor Core speedup on GPU, use TensorRT EP (--trt flag) with
  INT8 calibration — that requires tensorrt Python package and is a separate
  workflow.  This script produces the model file needed as a starting point.

Modes
-----
  dynamic (default)
    Quantizes weight tensors offline — no calibration images needed.
    Only MatMul ops quantized (Conv excluded: ConvInteger not in ORT CUDA EP).
    Only constant B-operands (projection weights); attention Q@K^T stays FP32.
    Run time: ~5–15 min.

  static
    Quantizes both weights AND activations using representative crops.
    Requires --calib-dir (real images) or uses synthetic random crops.
    Better accuracy preservation vs dynamic; same CUDA EP speed caveat applies.
    Run time: substantially longer due to per-sample inference passes.

Usage
-----
    # Quick dynamic quantization (recommended):
    python3 tools/quantize_backbone.py --onnx-dir ./onnx

    # Static with your own images for calibration:
    python3 tools/quantize_backbone.py --onnx-dir ./onnx --mode static \\
        --calib-dir /path/to/person/images

    # Static with synthetic calibration (no images needed):
    python3 tools/quantize_backbone.py --onnx-dir ./onnx --mode static

After quantization, use the new backbone via CLI:
    scripts/video.sh --from input.mp4 --backbone backbone_int8.onnx ...

Or bake it in as the default at build time:
    cmake -DSAM3D_BACKBONE_QUANT=ON ..  &&  cmake --build build -j

Dependencies
------------
    pip3 install onnx onnxruntime numpy
    pip3 install opencv-python  # only needed with --calib-dir
"""

import argparse
import os
import sys
import time

# ── Dependency check ──────────────────────────────────────────────────────────
_missing = []
for _pkg in ("onnx", "onnxruntime", "numpy"):
    try:
        __import__(_pkg)
    except ImportError:
        _missing.append(_pkg)
if _missing:
    print("Missing packages:", " ".join(_missing))
    print("  pip3 install", " ".join(_missing))
    sys.exit(1)

import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

# Image normalization constants — must match preprocess.hpp
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP  = 512


# ─── Calibration data readers (static mode only) ──────────────────────────────

class SyntheticCalibReader(CalibrationDataReader):
    """
    N samples of random ImageNet-normalised crops.

    Uses a fixed RNG seed so repeated runs are deterministic.  The
    distribution (N(0,1) after normalization) is a reasonable stand-in when
    no real images are available, but real person crops give better results.
    """
    def __init__(self, input_name: str, n: int = 100):
        self._name = input_name
        rng = np.random.default_rng(42)
        self._data = [
            rng.standard_normal((1, 3, CROP, CROP)).astype(np.float32)
            for _ in range(n)
        ]

    def get_next(self):
        if not self._data:
            return None
        return {self._name: self._data.pop(0)}


class ImageCalibReader(CalibrationDataReader):
    """
    Real-image calibration reader.

    Loads up to n_samples images from calib_dir, resizes to CROP×CROP,
    applies ImageNet normalization (matching crop_and_normalise in preprocess.hpp),
    and yields them one at a time.

    Best results: use person-crop images from the same distribution as inference
    (e.g. frames extracted from training or representative videos).
    """
    def __init__(self, input_name: str, calib_dir: str, n_samples: int = 200):
        try:
            import cv2
        except ImportError:
            print("Error: opencv-python needed for --calib-dir.")
            print("  pip3 install opencv-python")
            sys.exit(1)

        self._name = input_name
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(
            os.path.join(calib_dir, f)
            for f in os.listdir(calib_dir)
            if os.path.splitext(f)[1].lower() in exts
        )[:n_samples]

        if not paths:
            print(f"No images found in {calib_dir}")
            sys.exit(1)

        print(f"  Calibration: {len(paths)} images from {calib_dir}")
        self._tensors = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (CROP, CROP))
            t = img.astype(np.float32) / 255.0
            t = (t - _MEAN) / _STD
            t = t.transpose(2, 0, 1)[np.newaxis]  # [1,3,H,W]
            self._tensors.append(t)

    def get_next(self):
        if not self._tensors:
            return None
        return {self._name: self._tensors.pop(0)}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def dir_size_gb(directory: str, prefix: str) -> float:
    """Sum of files in directory whose name starts with prefix."""
    total = sum(
        os.path.getsize(os.path.join(directory, f))
        for f in os.listdir(directory)
        if f == prefix or f.startswith(prefix + ".")
    )
    return total / 1e9


def backbone_input_name(model_path: str) -> str:
    sess = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    return sess.get_inputs()[0].name


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Quantize backbone.onnx to INT8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--onnx-dir", default="./onnx",
                    help="Directory containing backbone.onnx (default: ./onnx)")
    ap.add_argument("--input", default=None,
                    help="Override input path (default: <onnx-dir>/backbone.onnx)")
    ap.add_argument("--output", default=None,
                    help="Override output path (default: <onnx-dir>/backbone_int8.onnx)")
    ap.add_argument("--mode", choices=["dynamic", "static"], default="dynamic",
                    help="Quantization mode (default: dynamic)")
    ap.add_argument("--calib-dir", default=None,
                    help="Images for static calibration (static mode; omit for synthetic)")
    ap.add_argument("--n-calib", type=int, default=100,
                    help="Number of calibration samples (default: 100)")
    ap.add_argument("--per-channel", action="store_true",
                    help="Per-channel weight quantization (static mode; better accuracy, slower)")
    args = ap.parse_args()

    in_path  = args.input  or os.path.join(args.onnx_dir, "backbone.onnx")
    out_path = args.output or os.path.join(args.onnx_dir, "backbone_int8.onnx")

    if not os.path.exists(in_path):
        print(f"Error: {in_path} not found")
        sys.exit(1)

    in_dir  = os.path.dirname(os.path.abspath(in_path))
    in_base = os.path.basename(in_path)
    in_gb   = dir_size_gb(in_dir, in_base)

    print(f"Input:   {in_path}  ({in_gb:.2f} GB)")
    print(f"Output:  {out_path}")
    print(f"Mode:    {args.mode}")
    print(f"ORT:     {onnxruntime.__version__}")

    # Warn about memory: int8 quantization loads the full FP32 model into RAM
    import psutil
    avail_gb = psutil.virtual_memory().available / 1e9
    if avail_gb < in_gb * 1.5:
        print(f"\nWARNING: {avail_gb:.1f} GB RAM available, model is {in_gb:.2f} GB.")
        print("         Quantization needs ~1.5× model size in RAM.  Close other apps.")

    # If the input uses external data (backbone.onnx.data exists), the output
    # must also use external data — the int8 model is still >2 GB and protobuf
    # has a 2 GB hard limit for single-file serialisation.
    has_ext_data = os.path.exists(in_path + ".data")
    if has_ext_data:
        print("  External data detected — output will also use external data format.")

    t0 = time.time()

    # Restrict to MatMul only.  ORT's CUDA EP implements MatMulInteger but NOT
    # ConvInteger, so quantizing Conv ops (e.g. patch-embed) makes the model
    # fail to load on CUDA.  ViT compute is dominated by MatMul anyway — the
    # patch-embed Conv is negligible and not worth the compatibility headache.
    CUDA_SAFE_OPS = ["MatMul"]

    if args.mode == "dynamic":
        print("\nRunning dynamic quantization …")
        print("  op types:  MatMul only (ConvInteger excluded — not in ORT CUDA EP)")
        print("  weights:   constant B-operands only (attention Q@K^T stays FP32)")
        quantize_dynamic(
            in_path,
            out_path,
            op_types_to_quantize=CUDA_SAFE_OPS,
            weight_type=QuantType.QInt8,
            use_external_data_format=has_ext_data,
            # MatMulConstBOnly: only quantize weight matrices (constant B), not
            # runtime activations such as attention scores (Q@K^T).
            extra_options={"MatMulConstBOnly": True},
        )
    else:
        print("\nRunning static quantization …")
        print("  op types:  MatMul only (ConvInteger excluded — not in ORT CUDA EP)")
        input_name = backbone_input_name(in_path)
        print(f"  Input tensor name: {input_name!r}")

        if args.calib_dir:
            reader = ImageCalibReader(input_name, args.calib_dir, args.n_calib)
        else:
            print(f"  No --calib-dir given → using {args.n_calib} synthetic samples")
            reader = SyntheticCalibReader(input_name, args.n_calib)

        quantize_static(
            in_path,
            out_path,
            reader,
            op_types_to_quantize=CUDA_SAFE_OPS,
            quant_format=QuantFormat.QDQ,
            per_channel=args.per_channel,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            use_external_data_format=has_ext_data,
        )

    elapsed = time.time() - t0

    out_dir  = os.path.dirname(os.path.abspath(out_path))
    out_base = os.path.basename(out_path)
    out_gb   = dir_size_gb(out_dir, out_base)
    ratio    = in_gb / out_gb if out_gb > 0 else float("nan")

    print(f"\nDone in {elapsed:.0f}s")
    print(f"Output size: {out_gb:.2f} GB  (was {in_gb:.2f} GB, {ratio:.1f}× smaller)")
    print()
    print("Original backbone.onnx is untouched — both models coexist in the onnx/ dir.")
    print()
    print("To use the quantized backbone:")
    print(f"  CLI:    --backbone {out_base}")
    print(f"  CMake:  cmake -DSAM3D_BACKBONE_QUANT=ON ..")
    print()
    print("To compare accuracy/speed against the original FP32 model:")
    print(f"  FP32:  scripts/video.sh --from input.mp4 ...")
    print(f"  INT8:  scripts/video.sh --from input.mp4 --backbone {out_base} ...")


if __name__ == "__main__":
    # psutil is optional — fall back gracefully
    try:
        import psutil  # noqa: F401
    except ImportError:
        import types
        psutil = types.ModuleType("psutil")
        psutil.virtual_memory = lambda: types.SimpleNamespace(available=float("inf"))
        sys.modules["psutil"] = psutil

    main()
