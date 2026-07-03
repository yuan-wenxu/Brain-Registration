import os
import argparse
import csv
import json
import gc
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from skimage import io


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Step4: apply the Step3 transform to atlas annotation')
    parser.add_argument('--label-path', required=True, help='atlas annotation path, e.g. annotation_10.npy')
    parser.add_argument('--step3-record-json', required=True, help='Step3 record JSON; previous paths are read automatically')
    parser.add_argument('--output-path', default=None, help='output directory; default: <step3-json>.parent.parent/04.apply_label directory')
    return parser


def _cleanup_images(*objects):
    for obj in objects:
        if obj is None:
            continue
        close_fn = getattr(obj, 'close', None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
    gc.collect()


def _load_label_image(label_path, atlas_slice=None):
    label_path = Path(label_path)
    if label_path.suffix.lower() == '.npy':
        arr = np.load(label_path)
        if arr.ndim == 3:
            idx = arr.shape[0] // 2 if atlas_slice is None else int(atlas_slice)
            idx = max(0, min(arr.shape[0] - 1, idx))
            arr = arr[idx]
        image = sitk.GetImageFromArray(arr.astype(np.uint16))
        image.SetSpacing([1.0, 1.0])
    else:
        image = sitk.ReadImage(str(label_path), sitk.sitkUInt16)
    return image


def _load_nissl_image(nissl_path, atlas_slice=None):
    nissl_path = Path(nissl_path)
    if nissl_path.suffix.lower() == '.npy':
        arr = np.load(nissl_path)
        if arr.ndim == 3:
            idx = arr.shape[0] // 2 if atlas_slice is None else int(atlas_slice)
            idx = max(0, min(arr.shape[0] - 1, idx))
            arr = arr[idx]
        arr = arr.astype(np.float32)
        amin, amax = float(arr.min()), float(arr.max())
        if amax > amin:
            arr = (arr - amin) / (amax - amin)
        image = sitk.GetImageFromArray(arr)
        image.SetSpacing([1.0, 1.0])
    else:
        image = sitk.ReadImage(str(nissl_path), sitk.sitkFloat32)
    return image


def _place_on_canvas(arr, tissue_mask, canvas_width=1140, canvas_height=800, is_label=False):
    masked = arr.astype(np.float32) * tissue_mask.astype(np.float32)
    nz = np.argwhere(tissue_mask)
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    if nz.size == 0:
        return canvas.astype(np.uint16 if is_label else np.float32)

    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    crop = masked[r0:r1 + 1, c0:c1 + 1]
    mask_crop = tissue_mask[r0:r1 + 1, c0:c1 + 1].astype(np.float32)

    ch, cw = crop.shape
    mask_crop = (mask_crop > 0.5).astype(np.float32)
    if ch > canvas_height or cw > canvas_width:
        raise ValueError(
            f'Atlas slice {cw}x{ch} exceeds canvas {canvas_width}x{canvas_height}; '
            'automatic canvas scaling is disabled'
        )

    top = max(0, (canvas_height - ch) // 2)
    left = max(0, (canvas_width - cw) // 2)
    canvas[top:top + ch, left:left + cw] = crop * mask_crop

    if is_label:
        return np.rint(np.clip(canvas, 0, np.iinfo(np.uint16).max)).astype(np.uint16)
    return canvas.astype(np.float32)


def _region_stats_and_centroids(label_arr):
    h, w = label_arr.shape
    labels = label_arr.astype(np.int64)
    unique = np.unique(labels)
    unique = unique[unique > 0]
    if unique.size == 0:
        return []

    rr, cc = np.indices((h, w), dtype=np.float64)
    stats = []
    total = float(h * w)
    for lab in unique:
        mask = labels == lab
        cnt = int(mask.sum())
        if cnt <= 0:
            continue
        r_mean = float(rr[mask].mean())
        c_mean = float(cc[mask].mean())
        stats.append((int(lab), cnt, cnt / total, r_mean, c_mean))
    stats.sort(key=lambda x: x[1], reverse=True)
    return stats


def _label_to_color_rgb(label_arr):
    max_label = int(label_arr.max()) if label_arr.size > 0 else 0
    rng = np.random.default_rng(42)
    colors = np.zeros((max_label + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    for i in range(1, max_label + 1):
        colors[i] = rng.integers(40, 245, size=3, dtype=np.uint8)
    rgb = colors[label_arr]
    return rgb


def _gray_to_u8(arr):
    arr = arr.astype(np.float32)
    amin, amax = float(arr.min()), float(arr.max())
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _save_label_overlay_tif(reference_arr, label_arr, output_rgb_tif, alpha=0.45):
    base = _gray_to_u8(reference_arr)
    base_rgb = np.stack([base, base, base], axis=-1)
    label_rgb = _label_to_color_rgb(label_arr)
    out = (base_rgb.astype(np.float32) * (1.0 - alpha) + label_rgb.astype(np.float32) * alpha).astype(np.uint8)
    io.imsave(output_rgb_tif, out)


def _resample_once_with_composite(image, reference, transform_paths, interpolator=sitk.sitkLinear, fill_value=0.0):
    merged = sitk.CompositeTransform(2)
    for path in transform_paths:
        tx = sitk.ReadTransform(path)
        merged.AddTransform(tx)
    if hasattr(merged, 'FlattenTransform'):
        merged.FlattenTransform()
    return sitk.Resample(image, reference, merged, interpolator, fill_value, image.GetPixelID())


def _load_step3_inputs(step3_record_json, output_path=None):
    step3_path = Path(step3_record_json)
    if not step3_path.exists():
        raise FileNotFoundError(f'Step3 record JSON does not exist: {step3_path}')
    with open(step3_path, 'r') as f:
        step3 = json.load(f)

    def _resolve(path_value, label, base_dir=None):
        if not path_value:
            raise ValueError(f'Step3 record JSON is missing {label}')
        path = Path(path_value)
        if not path.is_absolute() and not path.exists():
            path = (base_dir or step3_path.parent) / path
        if not path.exists():
            raise FileNotFoundError(f'{label} does not exist: {path}')
        return path

    step2_path = _resolve(step3.get('step2_record_json'), 'step2_record_json')
    with open(step2_path, 'r') as f:
        step2 = json.load(f)
    reference_path = _resolve(step3.get('fixed_path'), 'fixed_path')
    nissl_path = _resolve(step3.get('adjusted_allen_gray_tif'), 'adjusted_allen_gray_tif')

    step1_path_value = step2.get('step1_record_json')
    sample_name = reference_path.stem
    if step1_path_value:
        step1_path = Path(step1_path_value)
        if not step1_path.is_absolute() and not step1_path.exists():
            step1_path = step2_path.parent / step1_path
        if step1_path.exists():
            with open(step1_path, 'r') as f:
                step1 = json.load(f)
            if step1.get('input_path'):
                sample_name = Path(step1['input_path']).stem

    output_dir = (
        Path(output_path)
        if output_path is not None
        else step3_path.parent.parent / '04.apply_label'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        'step2_record_json': step2_path,
        'reference_path': reference_path,
        'nissl_path': nissl_path,
        'output_path': output_dir / f'{sample_name}_label.tif',
    }


def apply_transform_to_label(
    label_path,
    step3_record_json,
    output_path=None,
):
    """strict apply deformation chain to atlas annotation, with optional nissl output and stats report

    parameters:
    ----
    step3_record_json: step3_registration.py output record JSON. Step2 and image paths are resolved from it.
    """

    paths = _load_step3_inputs(step3_record_json, output_path=output_path)
    step2_record_json = paths['step2_record_json']
    reference_path = paths['reference_path']
    nissl_path = paths['nissl_path']
    output_path = paths['output_path']

    if step2_record_json is not None and os.path.exists(str(step2_record_json)):
        with open(step2_record_json, 'r') as f:
            s2 = json.load(f)
        atlas_slice = int(s2['selected_slice'])

    step3_chain = []
    if step3_record_json is not None and os.path.exists(str(step3_record_json)):
        with open(step3_record_json, 'r') as f:
            s3 = json.load(f)
        if s3.get('deformation_chain_forward'):
            step3_chain = [x['transform_path'] for x in s3['deformation_chain_forward']]
        elif s3.get('transform_path'):
            step3_chain = [s3['transform_path']]

    full_chain_forward = step3_chain
    print(f'Deformation chain ({len(full_chain_forward)} transforms):')
    for p in full_chain_forward:
        if not os.path.exists(p):
            raise FileNotFoundError(f'the file {p} does not exist, please check your step2 and step3 record json inputs')
        print(f'  {p}')

    reference = sitk.ReadImage(reference_path, sitk.sitkFloat32)

    # load reference
    label = _load_label_image(label_path, atlas_slice=atlas_slice)
    label_arr = sitk.GetArrayFromImage(label).astype(np.uint16)
    label_mask = np.ones_like(label_arr, dtype=np.uint8)
    label_canvas_arr = _place_on_canvas(
        label_arr,
        label_mask,
        canvas_width=int(reference.GetWidth()),
        canvas_height=int(reference.GetHeight()),
        is_label=True,
    )
    label = sitk.GetImageFromArray(label_canvas_arr)
    label.SetSpacing([1.0, 1.0])

    # load and apply transforms
    warped_label = _resample_once_with_composite(
        label, reference, full_chain_forward,
        interpolator=sitk.sitkNearestNeighbor, fill_value=0.0,
    )

    output_path = str(output_path)
    warped_arr = sitk.GetArrayFromImage(warped_label)
    ref_arr = sitk.GetArrayFromImage(reference)
    overlay_rgb_tif = output_path.replace('_label.tif', '_overlay.tif')
    io.imsave(output_path, _label_to_color_rgb(warped_arr))
    _save_label_overlay_tif(ref_arr, warped_arr, overlay_rgb_tif, alpha=0.45)

    # option nissl tif processing
    warped_nissl_tif = None
    annotation_nissl_merge_tif = None
    if nissl_path is not None:
        nissl_img = _load_nissl_image(nissl_path, atlas_slice=atlas_slice)
        nissl_arr = sitk.GetArrayFromImage(nissl_img)
        annotation_nissl_merge_tif = output_path.replace('_label.tif', '_annotation_nissl_merge.tif')
        _save_label_overlay_tif(nissl_arr, warped_arr, annotation_nissl_merge_tif, alpha=0.45)

    stats = _region_stats_and_centroids(warped_arr)
    stats_csv = output_path.replace('_label.tif', '_region_distribution.csv')
    with open(stats_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['region_id', 'pixel_count', 'area_ratio', 'centroid_row', 'centroid_col'])
        writer.writerows(stats)

    record_json = output_path.replace('_label.tif', '_step4_record.json')
    payload = {
        'label_path': str(label_path),
        'reference_path': str(reference_path),
        'atlas_slice': atlas_slice,
        'step2_record_json': str(step2_record_json),
        'step3_record_json': str(step3_record_json),
        'step3_chain': step3_chain,
        'applied_deformation_chain_forward': full_chain_forward,
        'label_rgb_tif': str(output_path),
        'overlay_rgb_tif': str(overlay_rgb_tif),
        'nissl_path': str(nissl_path),
        'warped_nissl_tif': str(warped_nissl_tif),
        'annotation_nissl_merge_tif': str(annotation_nissl_merge_tif),
        'region_distribution_csv': str(stats_csv),
    }
    with open(record_json, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'Label RGB tif saved to {output_path}')
    print(f'Overlay on step1 tif saved to {overlay_rgb_tif}')
    if warped_nissl_tif is not None:
        print(f'Warped nissl gray tif saved to {warped_nissl_tif}')
    if annotation_nissl_merge_tif is not None:
        print(f'Annotation-Nissl merge tif saved to {annotation_nissl_merge_tif}')
    print(f'Region distribution saved to {stats_csv}')
    print(f'Step4 record saved to {record_json}')

    _cleanup_images(
        reference,
        label,
        warped_label,
        warped_arr,
        ref_arr,
        nissl_img if 'nissl_img' in locals() else None,
        nissl_arr if 'nissl_arr' in locals() else None,
    )
    return warped_label


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()

    try:
        apply_transform_to_label(
            label_path=args.label_path,
            output_path=args.output_path,
            step3_record_json=args.step3_record_json,
        )
    finally:
        _cleanup_images()
