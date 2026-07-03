"""Convert NRRD files to NumPy .npy format.

Usage examples:
    python script/convert_nrrd_to_npy.py --input /path/to/file.nrrd --output /path/to/file.npy
    python script/convert_nrrd_to_npy.py --input /path/to/file.nrrd  # output defaults to same name .npy
    python script/convert_nrrd_to_npy.py --input /path/to/dir/       # batch convert all .nrrd in directory
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import nrrd
    _BACKEND = "pynrrd"
except ImportError:
    nrrd = None

if nrrd is None:
    try:
        import nibabel as nib
        _BACKEND = "nibabel"
    except ImportError:
        nib = None
        _BACKEND = None


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert NRRD file(s) to NumPy .npy format",
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="input .nrrd file or directory containing .nrrd files",
    )
    parser.add_argument(
        "--output", default=None, type=Path,
        help="output .npy file or directory; defaults to same name/location as input",
    )
    parser.add_argument(
        "--save-header", action="store_true", default=False,
        help="also save the NRRD header as a .meta.json sidecar file",
    )
    return parser


def _load_nrrd_pynrrd(path):
    data, header = nrrd.read(str(path))
    return data, header


def _load_nrrd_nibabel(path):
    img = nib.load(str(path))
    data = np.asarray(img)
    header = dict(img.header)
    return data, header


def _load_nrrd(path):
    if _BACKEND == "pynrrd":
        return _load_nrrd_pynrrd(path)
    elif _BACKEND == "nibabel":
        return _load_nrrd_nibabel(path)
    else:
        print("Error: no NRRD reader available. Install 'pynrrd' or 'nibabel':", file=sys.stderr)
        print("  pip install pynrrd", file=sys.stderr)
        print("  pip install nibabel", file=sys.stderr)
        sys.exit(1)


def _convert_one(input_path, output_path, save_header=False):
    """Convert a single NRRD file to .npy."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"[SKIP] not found: {input_path}", file=sys.stderr)
        return False

    if output_path.suffix.lower() != ".npy":
        output_path = output_path.with_suffix(".npy")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data, header = _load_nrrd(input_path)
    np.save(str(output_path), data)
    elapsed = time.time() - t0

    print(f"[OK] {input_path.name} -> {output_path.name}")
    print(f"     shape={data.shape}, dtype={data.dtype}, "
          f"size={output_path.stat().st_size / 1e6:.1f} MB, "
          f"time={elapsed:.1f}s")

    if save_header:
        import json
        meta_path = output_path.with_suffix(".meta.json")
        # Convert any non-serializable values to strings
        serializable = {}
        for k, v in header.items():
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)
        with open(meta_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"     header saved to {meta_path.name}")

    return True


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output

    if input_path.is_dir():
        # Batch mode: convert all .nrrd files in directory
        nrrd_files = sorted(input_path.glob("*.nrrd"))
        if not nrrd_files:
            print(f"No .nrrd files found in {input_path}", file=sys.stderr)
            sys.exit(1)

        if output_path is None:
            output_path = input_path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Batch converting {len(nrrd_files)} file(s)...\n")
        ok = 0
        for f in nrrd_files:
            out = output_path / f.with_suffix(".npy").name
            if _convert_one(f, out, save_header=args.save_header):
                ok += 1
        print(f"\nDone: {ok}/{len(nrrd_files)} converted.")
    else:
        # Single file mode
        if output_path is None:
            output_path = input_path.with_suffix(".npy")
        _convert_one(input_path, output_path, save_header=args.save_header)


if __name__ == "__main__":
    main()
