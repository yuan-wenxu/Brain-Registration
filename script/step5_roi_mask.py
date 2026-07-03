import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

import os
import argparse
import csv
import json
import numpy as np
import SimpleITK as sitk
from skimage import io
import gc
from pathlib import Path


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Step5: optionally generate ROI masks from the Step4 result'
    )
    parser.add_argument('--step4-record-json', required=True, help='Step4 record JSON path')
    parser.add_argument('--roi-txt-path', default=None, help='optional ROI list; Step5 is skipped when omitted')
    parser.add_argument('--structure-tree-csv', default=None, help='structure_tree_safe_2017.csv; required when ROI list is provided')
    parser.add_argument('--output-path', default=None, help='output directory; default: sibling 05.roi_mask directory')
    return parser


def _load_label_image(label_path, atlas_slice=None):
    if str(label_path).endswith('.npy'):
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


def _place_on_canvas(arr, tissue_mask, canvas_width=1140, canvas_height=800):
    masked = arr.astype(np.float32) * tissue_mask.astype(np.float32)
    nz = np.argwhere(tissue_mask)
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    if nz.size == 0:
        return canvas.astype(np.uint16)

    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    crop = masked[r0:r1 + 1, c0:c1 + 1]
    mask_crop = tissue_mask[r0:r1 + 1, c0:c1 + 1].astype(np.float32)

    ch, cw = crop.shape
    mask_crop = (mask_crop > 0.5).astype(np.float32)

    copy_h = min(ch, canvas_height)
    copy_w = min(cw, canvas_width)
    if copy_h <= 0 or copy_w <= 0:
        return canvas.astype(np.uint16)

    src_top = max(0, (ch - copy_h) // 2)
    src_left = max(0, (cw - copy_w) // 2)
    dst_top = max(0, (canvas_height - copy_h) // 2)
    dst_left = max(0, (canvas_width - copy_w) // 2)

    crop_view = crop[src_top:src_top + copy_h, src_left:src_left + copy_w]
    mask_view = mask_crop[src_top:src_top + copy_h, src_left:src_left + copy_w]
    canvas[dst_top:dst_top + copy_h, dst_left:dst_left + copy_w] = crop_view * mask_view
    return np.rint(np.clip(canvas, 0, np.iinfo(np.uint16).max)).astype(np.uint16)


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
def _resample_once_with_composite(image, reference, transform_paths):
    merged = sitk.CompositeTransform(2)
    for path in transform_paths:
        tx = sitk.ReadTransform(path)
        merged.AddTransform(tx)
    if hasattr(merged, 'FlattenTransform'):
        merged.FlattenTransform()
    return sitk.Resample(
        image,
        reference,
        merged,
        sitk.sitkNearestNeighbor,
        0.0,
        image.GetPixelID(),
    )


def _load_roi_list(roi_txt_path):
    rois = []
    with open(roi_txt_path, 'r') as f:
        for line in f:
            item = line.strip()
            if not item or item.startswith('#'):
                continue
            rois.append(item)
    return rois


def _to_int_or_none(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _normalize_text(value):
    return str(value).strip().lower() if value is not None else ''


def _load_structure_tree(structure_tree_csv):
    rows = []
    with open(structure_tree_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = _to_int_or_none(row.get('id'))
            if sid is None:
                continue
            rows.append(
                {
                    'id': sid,
                    'acronym': str(row.get('acronym', '')).strip(),
                    'name': str(row.get('name', '')).strip(),
                    'safe_name': str(row.get('safe_name', '')).strip(),
                    'structure_id_path': str(row.get('structure_id_path', '')).strip(),
                }
            )
    return rows


def _collect_descendant_ids(root_id, all_rows):
    needle = f'/{int(root_id)}/'
    out = set()
    for row in all_rows:
        path = row['structure_id_path']
        if needle in path:
            out.add(int(row['id']))
    out.add(int(root_id))
    return out


def _resolve_roi(roi_token, all_rows):
    token_norm = _normalize_text(roi_token)
    exact_acronym = [r for r in all_rows if _normalize_text(r['acronym']) == token_norm]
    if exact_acronym:
        return exact_acronym, 'acronym'

    return [], 'not_found'


def _sanitize_filename(name):
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(name))
    return safe.strip('_') or 'roi'


def _build_warped_label_array(step4_record_json):
    with open(step4_record_json, 'r') as f:
        s4 = json.load(f)

    required_fields = ['label_path', 'reference_path', 'applied_deformation_chain_forward']
    for key in required_fields:
        if key not in s4:
            raise KeyError(f'{key} is missing in step4 record: {step4_record_json}')

    label_path = s4['label_path']
    reference_path = s4['reference_path']
    atlas_slice = s4.get('atlas_slice')
    transform_paths = s4['applied_deformation_chain_forward']

    for p in transform_paths:
        if not os.path.exists(str(p)):
            raise FileNotFoundError(f'transform file not found: {p}')

    reference = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
    label = _load_label_image(label_path, atlas_slice=atlas_slice)
    label_arr = sitk.GetArrayFromImage(label).astype(np.uint16)
    label_mask = np.ones_like(label_arr, dtype=np.uint8)
    label_canvas_arr = _place_on_canvas(
        label_arr,
        label_mask,
        canvas_width=int(reference.GetWidth()),
        canvas_height=int(reference.GetHeight()),
    )
    label_canvas = sitk.GetImageFromArray(label_canvas_arr)
    label_canvas.SetSpacing([1.0, 1.0])

    warped_label = _resample_once_with_composite(label_canvas, reference, transform_paths)
    warped_arr = sitk.GetArrayFromImage(warped_label).astype(np.uint16)
    return warped_arr, s4


def _load_step1_info(step4_record_json):
    """Load Step1 record to get original image shape and canvas info."""
    with open(step4_record_json, 'r') as f:
        s4 = json.load(f)
    
    step4_dir = os.path.dirname(step4_record_json)
    step1_dir = os.path.join(os.path.dirname(step4_dir), '01.preprocess')
    
    # Find step1 record
    step1_records = []
    for fname in os.listdir(step1_dir):
        if fname.endswith('_step1_record.json'):
            step1_records.append(os.path.join(step1_dir, fname))
    
    if not step1_records:
        return None
    
    with open(step1_records[0], 'r') as f:
        s1 = json.load(f)
    
    return s1


def _read_image_hw(image_path):
    """Read image height/width from metadata only (avoid loading large image into memory)."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(image_path))
    reader.ReadImageInformation()
    w, h = reader.GetSize()
    return int(h), int(w)


def _scale_mask_to_fullres(mask_arr, step1_info):
    """Scale a mask from the Step4 reference canvas to the Step1 full-resolution canvas."""
    if step1_info is None:
        return None

    params = step1_info.get('params', {})
    stats = step1_info.get('stats', {})
    outputs = step1_info.get('outputs', {})
    fullres_canvas_path = outputs.get('masked_fullres_canvas_path')
    fullres_mask_canvas_path = outputs.get('mask_fullres_canvas_path')

    target_shape = None

    recorded_shape = params.get('fullres_canvas_shape')
    if not recorded_shape:
        recorded_shape = stats.get('fullres_canvas_shape')
    if isinstance(recorded_shape, (list, tuple)) and len(recorded_shape) == 2:
        target_shape = (int(recorded_shape[0]), int(recorded_shape[1]))

    if target_shape is None and fullres_canvas_path is not None and os.path.exists(fullres_canvas_path):
        target_shape = _read_image_hw(fullres_canvas_path)
    elif target_shape is None and fullres_mask_canvas_path is not None and os.path.exists(fullres_mask_canvas_path):
        target_shape = _read_image_hw(fullres_mask_canvas_path)
    elif target_shape is None:
        target_shape = tuple(step1_info['stats']['input_shape'][:2])

    if mask_arr.shape == tuple(target_shape):
        return mask_arr

    try:
        from skimage import transform
        mask_scaled = transform.resize(
            mask_arr.astype(np.uint8),
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
        return (mask_scaled > 0).astype(np.uint8)
    except Exception as e:
        print(f'Warning: failed to scale mask to fullres: {e}')
        return None


def generate_roi_masks(
    step4_record_json,
    roi_txt_path,
    structure_tree_csv,
    output_dir,
):
    if not os.path.exists(step4_record_json):
        raise FileNotFoundError(f'step4 record not found: {step4_record_json}')
    if not os.path.exists(roi_txt_path):
        raise FileNotFoundError(f'roi txt not found: {roi_txt_path}')
    if not os.path.exists(structure_tree_csv):
        raise FileNotFoundError(f'structure tree csv not found: {structure_tree_csv}')

    os.makedirs(output_dir, exist_ok=True)

    roi_list = _load_roi_list(roi_txt_path)
    if len(roi_list) == 0:
        raise ValueError(f'roi list is empty: {roi_txt_path}')

    all_rows = _load_structure_tree(structure_tree_csv)
    warped_arr, s4 = _build_warped_label_array(step4_record_json)
    total_pixels = float(warped_arr.size)
    
    # Load Step1 info for high-resolution scaling
    step1_info = _load_step1_info(step4_record_json)

    report_rows = []
    mask_paths = {}
    fullres_mask_paths = {}
    present_ids = set(int(v) for v in np.unique(warped_arr))
    
    for roi in roi_list:
        matched_rows, match_mode = _resolve_roi(roi, all_rows)
        if not matched_rows:
            report_rows.append([roi, match_mode, '', 0, 0, 0, 0.0, '', ''])
            continue

        all_ids = set()
        root_ids = []
        for row in matched_rows:
            root_id = int(row['id'])
            root_ids.append(root_id)
            all_ids.update(_collect_descendant_ids(root_id, all_rows))

        all_ids_sorted = sorted(all_ids)

        # Fast pre-check: if no label id exists in this image, skip full mask rendering.
        if present_ids.isdisjoint(all_ids_sorted):
            print(f'Skipping {roi}: no matched label ids in warped label image')
            report_rows.append(
                [
                    roi,
                    match_mode,
                    '|'.join(str(x) for x in sorted(root_ids)),
                    len(root_ids),
                    len(all_ids_sorted),
                    0,
                    0.0,
                    '|'.join(str(x) for x in all_ids_sorted),
                    '',
                ]
            )
            continue

        mask = np.isin(warped_arr, all_ids_sorted)
        pixel_count = int(mask.sum())
        area_ratio = float(pixel_count / total_pixels) if total_pixels > 0 else 0.0
        
        # Skip empty masks (no pixels)
        if pixel_count == 0:
            print(f'Skipping {roi}: no pixels in mask')
            report_rows.append(
                [
                    roi,
                    match_mode,
                    '|'.join(str(x) for x in sorted(root_ids)),
                    len(root_ids),
                    len(all_ids_sorted),
                    pixel_count,
                    area_ratio,
                    '|'.join(str(x) for x in all_ids_sorted),
                    '',
                ]
            )
            continue

        # Save at the Step4 reference canvas resolution (currently 1140x800, width x height).
        out_name = f'{_sanitize_filename(roi)}_mask.tif'
        out_path = os.path.join(output_dir, out_name)
        PIL.Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)
        mask_paths[roi] = out_path
        
        # Generate high-resolution grayscale mask
        fullres_mask_path = None
        if step1_info is not None:
            try:
                mask_fullres = _scale_mask_to_fullres(mask, step1_info)
                if mask_fullres is not None:
                    gray = (mask_fullres.astype(np.uint8) * 255)
                    out_name_gray = f'{_sanitize_filename(roi)}_mask_fullres_gray.png'
                    fullres_mask_path = os.path.join(output_dir, out_name_gray)
                    PIL.Image.fromarray(gray).save(fullres_mask_path)
                    fullres_mask_paths[roi] = fullres_mask_path
            except Exception as e:
                print(f'Warning: failed to generate fullres grayscale mask for {roi}: {e}')

        report_rows.append(
            [
                roi,
                match_mode,
                '|'.join(str(x) for x in sorted(root_ids)),
                len(root_ids),
                len(all_ids_sorted),
                pixel_count,
                area_ratio,
                '|'.join(str(x) for x in all_ids_sorted),
                fullres_mask_path if fullres_mask_path else '',
            ]
        )

    report_csv = os.path.join(output_dir, 'roi_mask_report.csv')
    with open(report_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'roi',
                'match_mode',
                'matched_root_ids',
                'matched_root_count',
                'descendant_inclusive_id_count',
                'pixel_count',
                'area_ratio',
                'descendant_inclusive_ids',
                'fullres_gray_mask_path',
            ]
        )
        writer.writerows(report_rows)

    record_json = os.path.join(output_dir, 'step5_record.json')
    payload = {
        'step4_record_json': str(step4_record_json),
        'label_path': str(s4.get('label_path')),
        'reference_path': str(s4.get('reference_path')),
        'atlas_slice': s4.get('atlas_slice'),
        'applied_deformation_chain_forward': s4.get('applied_deformation_chain_forward', []),
        'roi_txt_path': str(roi_txt_path),
        'structure_tree_csv': str(structure_tree_csv),
        'output_dir': str(output_dir),
        'roi_count': len(roi_list),
        'roi_mask_report_csv': str(report_csv),
        'roi_mask_tifs': mask_paths,
        'roi_mask_fullres_gray_pngs': fullres_mask_paths,
        'step1_info_available': step1_info is not None,
    }
    with open(record_json, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'ROI masks saved under: {output_dir}')
    print(f'ROI mask report saved to: {report_csv}')
    if fullres_mask_paths:
        print(f'High-resolution grayscale masks saved for {len(fullres_mask_paths)} ROIs')
    print(f'Step5 record saved to: {record_json}')
    _cleanup_images(warped_arr, all_rows, roi_list)
    return report_csv, record_json


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    if args.roi_txt_path is None:
        print('Step5 skipped: --roi-txt-path was not provided')
    else:
        if args.structure_tree_csv is None:
            raise ValueError('--structure-tree-csv is required when --roi-txt-path is provided')
        step4_record_path = Path(args.step4_record_json)
        output_path = (
            Path(args.output_path)
            if args.output_path is not None
            else step4_record_path.parent.parent / '05.roi_mask'
        )
        try:
            generate_roi_masks(
                step4_record_json=step4_record_path,
                roi_txt_path=Path(args.roi_txt_path),
                structure_tree_csv=Path(args.structure_tree_csv),
                output_dir=output_path,
            )
        finally:
            _cleanup_images()
