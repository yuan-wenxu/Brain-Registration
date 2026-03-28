import os
import argparse
import csv
import json
import numpy as np
import SimpleITK as sitk
from skimage import io


def _load_label_image(label_path, atlas_slice=None):
    if label_path.endswith('.npy'):
        arr = np.load(label_path)
        if arr.ndim == 3:
            idx = arr.shape[0] // 2 if atlas_slice is None else int(atlas_slice)
            idx = max(0, min(arr.shape[0] - 1, idx))
            arr = arr[idx]
        image = sitk.GetImageFromArray(arr.astype(np.uint16))
        image.SetSpacing([1.0, 1.0])
    else:
        image = sitk.ReadImage(label_path, sitk.sitkUInt16)
    return image


def _load_nissl_image(nissl_path, atlas_slice=None):
    if nissl_path.endswith('.npy'):
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
        image = sitk.ReadImage(nissl_path, sitk.sitkFloat32)
    return image


def _place_on_canvas(arr, tissue_mask, canvas_width=1152, canvas_height=832, is_label=False):
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

    copy_h = min(ch, canvas_height)
    copy_w = min(cw, canvas_width)
    if copy_h <= 0 or copy_w <= 0:
        return canvas.astype(np.uint16 if is_label else np.float32)

    src_top = max(0, (ch - copy_h) // 2)
    src_left = max(0, (cw - copy_w) // 2)
    dst_top = max(0, (canvas_height - copy_h) // 2)
    dst_left = max(0, (canvas_width - copy_w) // 2)

    crop_view = crop[src_top:src_top + copy_h, src_left:src_left + copy_w]
    mask_view = mask_crop[src_top:src_top + copy_h, src_left:src_left + copy_w]
    canvas[dst_top:dst_top + copy_h, dst_left:dst_left + copy_w] = crop_view * mask_view

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


def apply_transform_to_label(
    label_path,
    reference_path,
    output_path,
    atlas_slice=None,
    step2_record_json=None,
    step3_record_json=None,
    nissl_path=None,
):
    """strict apply deformation chain to atlas annotation, with optional nissl output and stats report

    parameters:
    ----
    step2_record_json: step2_affine.py output record json
    step3_record_json: step3_nonlinear.py output record json
    """

    step2_chain = []
    if step2_record_json is not None and os.path.exists(str(step2_record_json)):
        with open(step2_record_json, 'r') as f:
            s2 = json.load(f)
        selected_slice = int(s2['selected_slice'])
        if atlas_slice is None:
            atlas_slice = selected_slice
        elif int(atlas_slice) != selected_slice:
            raise ValueError(
                f'atlas_slice({atlas_slice}) is different from step2 selected_slice({selected_slice}), please check your inputs'
            )
        if s2.get('deformation_chain_forward'):
            step2_chain = [x['transform_path'] for x in s2['deformation_chain_forward']]
        elif s2.get('merged_rigid_affine_transform_path'):
            step2_chain = [s2['merged_rigid_affine_transform_path']]


    step3_chain = []
    if step3_record_json is not None and os.path.exists(str(step3_record_json)):
        with open(step3_record_json, 'r') as f:
            s3 = json.load(f)
        if s3.get('deformation_chain_forward'):
            step3_chain = [x['transform_path'] for x in s3['deformation_chain_forward']]
        elif s3.get('transform_path'):
            step3_chain = [s3['transform_path']]

    full_chain_forward = step2_chain + step3_chain
    print(f'Deformation chain ({len(full_chain_forward)} transforms):')
    for p in full_chain_forward:
        if not os.path.exists(p):
            raise FileNotFoundError(f'the file {p} does not exist, please check your step2 and step3 record json inputs')
        print(f'  {p}')

    reference = sitk.ReadImage(reference_path, sitk.sitkFloat32)

    # ---- load reference ----
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

    # ---- load and apply transforms ----
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

    # ---- option nissl tif processing ----
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
        'step2_chain': step2_chain,
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
    return warped_label


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Step4: apply deformation chain to atlas annotation, with optional nissl output and stats report')
    parser.add_argument('--label-path', required=True, help='atlas annotation path (e.g. annotation_10.npy or annotation_10.tif)')
    parser.add_argument('--reference-path', required=True, help='reference image path (e.g. step1_resampled.tif)')
    parser.add_argument('--output-path', required=True, help='output path for the warped label')
    parser.add_argument('--atlas-slice', type=int, default=None, help='atlas slice number; can be left empty to infer from step2 record')
    parser.add_argument('--step2-record-json', default=None, help='step2 record json path (_step2_record.json)')
    parser.add_argument('--step3-record-json', default=None, help='step3 record json path (_step3_record.json)')
    parser.add_argument('--nissl-path', default=None, help='ara_nissl_10.npy path (optional)')
    return parser


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    apply_transform_to_label(
        label_path=args.label_path,
        reference_path=args.reference_path,
        output_path=args.output_path,
        atlas_slice=args.atlas_slice,
        step2_record_json=args.step2_record_json,
        step3_record_json=args.step3_record_json,
        nissl_path=args.nissl_path,
    )
