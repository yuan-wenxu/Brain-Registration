import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

import argparse
import json
import gc
from pathlib import Path
import numpy as np
from skimage import io, transform, color, measure
from skimage.filters import threshold_otsu, gaussian
from skimage import morphology
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from scipy.signal import find_peaks


CANVAS_WIDTH = 1140
CANVAS_HEIGHT = 800


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Step1: preprocess brain slice image',
        add_help=False,
    )
    parser.add_argument('--help', action='help', help='show this help message and exit')
    parser.add_argument('--input-path', required=True, help='input image path')
    parser.add_argument(
        '--output-path',
        default=None,
        help='output directory; default: <input directory>/01.preprocess',
    )
    parser.add_argument('--input-res', type=float, default=0.294, help='input resolution (um/px)')
    parser.add_argument('--target-res', type=float, default=10.0, help='target resolution (um/px)')
    parser.add_argument(
        '--grayscale-mode',
        choices=('nissl', 'rgb'),
        default='rgb',
        help='RGB grayscale conversion: stain optical density or standard luminance (default: rgb)',
    )
    parser.add_argument(
        '--rotation',
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help='counterclockwise input rotation in degrees (default: 0)',
    )
    parser.add_argument(
        '--brain-layout',
        choices=('auto', 'whole', 'left', 'right'),
        default='auto',
        help='brain tissue layout; half brains are aligned by their medial edge (default: auto)',
    )
    parser.add_argument(
        '--replace-background-values',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='remove white-gray mosaic background by replacing uint8 values 255 and 204 with zero (default: enabled)',
    )
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


def _replace_background_values(img_arr):
    replaced_pixels = 0
    if img_arr.dtype != np.uint8:
        return img_arr, 0

    if img_arr.ndim == 3 and img_arr.shape[2] >= 3:
        is_255 = (img_arr[..., 0] == 255) & (img_arr[..., 1] == 255) & (img_arr[..., 2] == 255)
        is_204 = (img_arr[..., 0] == 204) & (img_arr[..., 1] == 204) & (img_arr[..., 2] == 204)
        hit = is_255 | is_204
        replaced_pixels = int(np.count_nonzero(hit))
        if replaced_pixels > 0:
            img_arr[hit, 0] = 0
            img_arr[hit, 1] = 0
            img_arr[hit, 2] = 0
    else:
        is_255 = img_arr == 255
        is_204 = img_arr == 204
        hit = is_255 | is_204
        replaced_pixels = int(np.count_nonzero(hit))
        if replaced_pixels > 0:
            img_arr[hit] = 0

    return img_arr, replaced_pixels


def _normalize_gray(gray):
    gray = gray.astype(np.float32)
    finite = np.isfinite(gray)
    if not np.any(finite):
        return np.zeros_like(gray, dtype=np.float32)
    low = float(np.min(gray[finite]))
    high = float(np.max(gray[finite]))
    if high > low:
        gray = (gray - low) / (high - low)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)
    return np.clip(gray, 0.0, 1.0)


def _nissl_rgb_to_gray(img_arr):
    """Extract purple-blue Nissl stain concentration in optical-density space."""
    rgb = img_arr[..., :3].astype(np.float32)
    if np.issubdtype(img_arr.dtype, np.integer):
        rgb /= float(np.iinfo(img_arr.dtype).max)
    else:
        max_value = float(np.nanmax(rgb)) if rgb.size else 1.0
        if max_value > 1.0:
            rgb /= max_value
    rgb = np.clip(rgb, 0.0, 1.0)

    # Exclude black canvas/mosaic pixels: black in RGB is not Nissl stain.
    valid = np.max(rgb, axis=2) > 0.02
    optical_density = -np.log(np.clip(rgb, 1.0 / 255.0, 1.0))
    # A normalized purple-blue nuclear/Nissl stain direction.
    stain_vector = np.array([0.65, 0.70, 0.29], dtype=np.float32)
    stain_vector /= np.linalg.norm(stain_vector)
    concentration = np.sum(optical_density * stain_vector, axis=2)
    concentration[~valid] = 0.0

    tissue_values = concentration[valid & np.isfinite(concentration)]
    if tissue_values.size:
        low, high = np.percentile(tissue_values, [1.0, 99.0])
        if high > low:
            concentration = (concentration - float(low)) / float(high - low)
    concentration = np.clip(concentration, 0.0, 1.0)
    concentration[~valid] = 0.0
    return concentration.astype(np.float32)


def _convert_to_gray(img_arr, grayscale_mode='rgb'):
    if img_arr.ndim != 3:
        return _normalize_gray(img_arr)
    if grayscale_mode == 'nissl':
        return _nissl_rgb_to_gray(img_arr)
    if grayscale_mode == 'rgb':
        return color.rgb2gray(img_arr[..., :3]).astype(np.float32)
    raise ValueError('grayscale_mode must be one of: nissl, rgb')


def _extract_brain_mask(gray_resampled):
    x = gray_resampled.astype(np.float32)
    amin, amax = float(x.min()), float(x.max())
    if amax > amin:
        x = (x - amin) / (amax - amin)

    smooth = gaussian(x, sigma=1.2, preserve_range=True)
    h, w = smooth.shape
    bw = max(8, int(min(h, w) * 0.03))
    border = np.concatenate([
        smooth[:bw, :].ravel(),
        smooth[-bw:, :].ravel(),
        smooth[:, :bw].ravel(),
        smooth[:, -bw:].ravel(),
    ])
    bg_val = float(np.median(border))

    diff = np.abs(smooth - bg_val)
    dmin, dmax = float(diff.min()), float(diff.max())
    min_px = max(80, int(x.size * 0.0003))
    hole_px = max(80, int(x.size * 0.0003))

    def _clean_mask(mask_in):
        mask_out = binary_fill_holes(mask_in)
        mask_out = morphology.remove_small_objects(mask_out.astype(bool), min_size=min_px)
        mask_out = morphology.remove_small_holes(mask_out.astype(bool), area_threshold=hole_px)
        mask_out = morphology.binary_closing(mask_out, morphology.disk(10))
        labeled = measure.label(mask_out)
        if labeled.max() > 0:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            largest = int(np.argmax(counts))
            mask_out = labeled == largest
        mask_out = binary_fill_holes(mask_out)
        return mask_out.astype(bool)

    if dmax <= dmin + 1e-8:
        mask = _clean_mask(x > 0)
    else:
        th = float(threshold_otsu(diff))
        th_loose = max(dmin, th * 0.55)
        mask = _clean_mask(diff > th_loose)

    coverage = float(mask.mean())
    if coverage > 0.99:
        th_strict = float(np.percentile(diff, 85.0))
        stricter = _clean_mask(diff > th_strict)
        stricter_cov = float(stricter.mean())
        if 0.01 < stricter_cov < coverage:
            mask = stricter

    coverage = float(mask.mean())
    if coverage < 0.12:
        th_relax = float(np.percentile(diff, 55.0))
        relaxed = _clean_mask(diff > th_relax)
        relaxed = morphology.binary_dilation(relaxed, morphology.disk(2))
        relaxed = _clean_mask(relaxed)
        relaxed_cov = float(relaxed.mean())
        if relaxed_cov > coverage:
            mask = relaxed

    return mask.astype(bool)


def _infer_brain_layout(tissue_mask):
    """Infer whether a mask is a whole brain or a left/right hemisphere.

    A hemisected brain usually has one nearly straight vertical medial edge.
    """
    nz = np.argwhere(tissue_mask)
    if nz.size == 0:
        return 'whole', {'reason': 'empty_mask'}

    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    crop = tissue_mask[r0:r1 + 1, c0:c1 + 1]
    rows = np.flatnonzero(crop.any(axis=1))
    if rows.size < 20:
        return 'whole', {'reason': 'insufficient_rows'}

    # Ignore curved superior/inferior tips and assess the central 70% of rows.
    lo = int(rows.size * 0.15)
    hi = max(lo + 1, int(rows.size * 0.85))
    rows = rows[lo:hi]
    left_edges = np.array([np.flatnonzero(crop[r])[0] for r in rows], dtype=np.float32)
    right_edges = np.array([np.flatnonzero(crop[r])[-1] for r in rows], dtype=np.float32)

    def _straightness(edge):
        x = np.arange(edge.size, dtype=np.float32)
        trend = np.polyval(np.polyfit(x, edge, 1), x)
        return float(np.median(np.abs(edge - trend)))

    left_score = _straightness(left_edges)
    right_score = _straightness(right_edges)
    width = max(1, int(c1 - c0 + 1))
    straight_threshold = max(2.0, width * 0.025)
    confidence_ratio = 0.6

    if right_score <= straight_threshold and right_score < left_score * confidence_ratio:
        layout = 'left'
    elif left_score <= straight_threshold and left_score < right_score * confidence_ratio:
        layout = 'right'
    else:
        layout = 'whole'
    return layout, {
        'left_edge_straightness': left_score,
        'right_edge_straightness': right_score,
        'straightness_threshold': float(straight_threshold),
    }


def _estimate_anatomical_midline(tissue_mask, brain_layout='whole'):
    """Estimate the midline by local mirror symmetry around a dorsal valley."""
    nz = np.argwhere(tissue_mask)
    if nz.size == 0:
        return None, {'confident': False, 'reason': 'empty_mask'}
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    crop = tissue_mask[r0:r1 + 1, c0:c1 + 1].astype(bool)
    h, w = crop.shape
    if h < 20 or w < 20:
        return None, {'confident': False, 'reason': 'mask_too_small'}

    valid = crop.any(axis=0)
    top = np.full(w, np.nan, dtype=np.float32)
    for col in np.flatnonzero(valid):
        top[col] = float(np.flatnonzero(crop[:, col])[0])
    valid_cols = np.flatnonzero(valid)
    top = np.interp(np.arange(w), valid_cols, top[valid_cols]).astype(np.float32)

    # A geometrically low point on the dorsal contour has a larger row coordinate.
    smooth_top = gaussian_filter1d(top, sigma=max(1.0, w * 0.01), mode='nearest')
    min_valley_depth = max(2.0, h * 0.025)
    valley_indices, _ = find_peaks(
        smooth_top,
        distance=max(5, int(w * 0.08)),
        prominence=min_valley_depth,
    )
    valley_indices = valley_indices[
        (valley_indices >= int(w * 0.08)) & (valley_indices <= int(w * 0.92))
    ]
    min_compare_length = max(6, int(w * 0.06))
    max_compare_length = max(min_compare_length, int(w * 0.25))
    max_symmetry_mae = max(2.0, h * 0.04)

    if brain_layout == 'right':
        ordered_valleys = sorted(int(x) for x in valley_indices)
    elif brain_layout == 'left':
        ordered_valleys = sorted((int(x) for x in valley_indices), reverse=True)
    else:
        ordered_valleys = [int(x) for x in valley_indices]

    accepted = []
    for valley_col in ordered_valleys:
        compare_length = min(valley_col, w - valley_col - 1, max_compare_length)
        if compare_length < min_compare_length:
            continue
        left_profile = smooth_top[valley_col - compare_length:valley_col][::-1]
        right_profile = smooth_top[valley_col + 1:valley_col + 1 + compare_length]

        # Compare shapes relative to the valley height, not their absolute image rows.
        valley_height = float(smooth_top[valley_col])
        left_relative = left_profile - valley_height
        right_relative = right_profile - valley_height
        symmetry_mae = float(np.mean(np.abs(left_relative - right_relative)))
        left_depth = float(valley_height - np.min(left_profile))
        right_depth = float(valley_height - np.min(right_profile))
        bilateral_depth = min(left_depth, right_depth)
        if symmetry_mae > max_symmetry_mae or bilateral_depth < min_valley_depth:
            continue
        accepted.append({
            'valley_col': valley_col,
            'compare_length': int(compare_length),
            'symmetry_mae': symmetry_mae,
            'left_depth': left_depth,
            'right_depth': right_depth,
            'bilateral_depth': bilateral_depth,
            'score': bilateral_depth - symmetry_mae,
        })
        # For a known dominant hemisphere, traversal order is part of the method.
        if brain_layout in ('left', 'right'):
            break

    if not accepted:
        return None, {
            'confident': False,
            'reason': 'no_locally_symmetric_dorsal_valley',
            'brain_layout': str(brain_layout),
            'detected_valley_cols_in_bbox': [int(x) for x in valley_indices],
            'min_compare_length': int(min_compare_length),
            'max_compare_length': int(max_compare_length),
            'max_symmetry_mae': float(max_symmetry_mae),
            'min_valley_depth': float(min_valley_depth),
        }
    best = accepted[0] if brain_layout in ('left', 'right') else max(
        accepted, key=lambda item: item['score']
    )
    local_col = best['valley_col']

    left_area = int(np.count_nonzero(crop[:, :local_col]))
    right_area = int(np.count_nonzero(crop[:, local_col + 1:]))
    total_area = max(1, left_area + right_area)
    smaller_side_fraction = min(left_area, right_area) / total_area
    confident = bool(smaller_side_fraction >= 0.03)
    diagnostics = {
        'confident': confident,
        'method': 'local_mirror_symmetry_around_dorsal_valley',
        'brain_layout': str(brain_layout),
        'midline_col_in_bbox': local_col,
        'midline_col_in_image': int(c0 + local_col),
        'compare_length': best['compare_length'],
        'symmetry_mae': best['symmetry_mae'],
        'max_symmetry_mae': float(max_symmetry_mae),
        'left_valley_depth': best['left_depth'],
        'right_valley_depth': best['right_depth'],
        'bilateral_valley_depth': best['bilateral_depth'],
        'min_valley_depth': float(min_valley_depth),
        'smaller_side_fraction': float(smaller_side_fraction),
    }
    return (int(c0 + local_col) if confident else None), diagnostics


def _canvas_position(
    mask_crop,
    canvas_height,
    canvas_width,
    brain_layout,
    midline_col_in_crop=None,
):
    ch, cw = mask_crop.shape
    top = max(0, (canvas_height - ch) // 2)
    if midline_col_in_crop is not None:
        medial_col = float(midline_col_in_crop)
        left = canvas_width // 2 - medial_col
    elif brain_layout == 'left':
        row_edges = [np.flatnonzero(row)[-1] for row in mask_crop if row.any()]
        medial_col = int(np.median(row_edges))
        left = canvas_width // 2 - medial_col
    elif brain_layout == 'right':
        row_edges = [np.flatnonzero(row)[0] for row in mask_crop if row.any()]
        medial_col = int(np.median(row_edges))
        left = canvas_width // 2 - medial_col
    else:
        left = (canvas_width - cw) // 2
    left = int(round(min(max(0, left), max(0, canvas_width - cw))))
    return top, left


def _save_masked_brain_on_canvas(
    gray_resampled,
    tissue_mask,
    output_path,
    canvas_width=CANVAS_WIDTH,
    canvas_height=CANVAS_HEIGHT,
    mask_canvas_output_path=None,
    brain_layout='whole',
    anatomical_midline_col=None,
):
    masked = gray_resampled.astype(np.float32) * tissue_mask.astype(np.float32)
    nz = np.argwhere(tissue_mask)
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    mask_canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    if nz.size == 0:
        if output_path is not None:
            io.imsave(output_path, canvas.astype(np.uint16))
        if mask_canvas_output_path is not None:
            io.imsave(mask_canvas_output_path, mask_canvas)
        return

    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    crop = masked[r0:r1 + 1, c0:c1 + 1]
    mask_crop = tissue_mask[r0:r1 + 1, c0:c1 + 1].astype(np.float32)

    ch, cw = crop.shape
    if ch > canvas_height or cw > canvas_width:
        raise ValueError(
            f'Tissue crop {cw}x{ch} exceeds canvas {canvas_width}x{canvas_height}; '
            'automatic canvas scaling is disabled'
        )
    mask_crop = (mask_crop > 0.5).astype(np.uint8)

    midline_col_in_crop = None
    if anatomical_midline_col is not None:
        midline_col_in_crop = float(anatomical_midline_col) - float(c0)
    top, left = _canvas_position(
        mask_crop,
        canvas_height,
        canvas_width,
        brain_layout,
        midline_col_in_crop=midline_col_in_crop,
    )
    canvas[top:top + ch, left:left + cw] = crop
    mask_canvas[top:top + ch, left:left + cw] = (mask_crop * 255).astype(np.uint8)

    canvas_u16 = (np.clip(canvas, 0, 1) * 65535).astype(np.uint16)
    if output_path is not None:
        io.imsave(output_path, canvas_u16)
    if mask_canvas_output_path is not None:
        io.imsave(mask_canvas_output_path, mask_canvas)
    
    return canvas, mask_canvas


def _save_fullres_canvas(
    gray_original,
    tissue_mask_downsampled,
    downsampled_shape,
    output_path,
    canvas_width=CANVAS_WIDTH,
    canvas_height=CANVAS_HEIGHT,
    brain_layout='whole',
    anatomical_midline_col=None,
):
    """Generate canvas at original resolution using the downsampled mask.
    
    The canvas dimensions at original resolution are: canvas_width / downsampling_scale, canvas_height / downsampling_scale.
    """
    scale_factor = np.array(downsampled_shape, dtype=np.float32) / np.array(gray_original.shape, dtype=np.float32)
    fullres_canvas_h = int(np.round(canvas_height / scale_factor[0]))
    fullres_canvas_w = int(np.round(canvas_width / scale_factor[1]))
    
    tissue_mask_fullres = transform.resize(
        tissue_mask_downsampled.astype(np.float32),
        gray_original.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(bool)
    
    masked = gray_original.astype(np.float32) * tissue_mask_fullres.astype(np.float32)
    nz = np.argwhere(tissue_mask_fullres)
    canvas = np.zeros((fullres_canvas_h, fullres_canvas_w), dtype=np.float32)
    
    if nz.size == 0:
        if output_path is not None:
            io.imsave(output_path, canvas.astype(np.uint16))
        return
    
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    crop = masked[r0:r1 + 1, c0:c1 + 1]
    
    ch, cw = crop.shape
    if ch > fullres_canvas_h or cw > fullres_canvas_w:
        raise ValueError(
            f'Full-resolution tissue crop {cw}x{ch} exceeds canvas '
            f'{fullres_canvas_w}x{fullres_canvas_h}; automatic canvas scaling is disabled'
        )
    midline_col_fullres = None
    if anatomical_midline_col is not None:
        midline_col_fullres = float(anatomical_midline_col) / float(scale_factor[1])
    
    mask_crop = tissue_mask_fullres[r0:r1 + 1, c0:c1 + 1]
    midline_col_in_crop = None
    if anatomical_midline_col is not None:
        midline_col_in_crop = midline_col_fullres - float(c0)
    top, left = _canvas_position(
        mask_crop,
        fullres_canvas_h,
        fullres_canvas_w,
        brain_layout,
        midline_col_in_crop=midline_col_in_crop,
    )
    canvas[top:top + ch, left:left + cw] = crop
    
    canvas_u16 = (np.clip(canvas, 0, 1) * 65535).astype(np.uint16)
    if output_path is not None:
        io.imsave(output_path, canvas_u16)
    
    return canvas, (int(fullres_canvas_h), int(fullres_canvas_w))


def preprocess_image(
    input_path,
    gray_output_path,
    resampled_output_path,
    mask_output_path,
    input_res=0.294,
    target_res=10.0,
    replace_background_values=True,
    rotation=0,
    brain_layout='auto',
    grayscale_mode='rgb',
):
    """
    Step1: gray image + resample
    1) gray_output
    2) resampled_output
    """
    print(f'Loading {input_path} ...')
    img_arr = io.imread(input_path)
    original_input_shape = img_arr.shape
    print(f'Shape: {img_arr.shape}, dtype: {img_arr.dtype}')

    if rotation not in (0, 90, 180, 270):
        raise ValueError('rotation must be one of: 0, 90, 180, 270')
    if rotation:
        img_arr = np.rot90(img_arr, k=rotation // 90)
        print(f'Input rotated {rotation} degrees counterclockwise; shape: {img_arr.shape}')

    replaced_pixels = 0
    if replace_background_values:
        img_arr, replaced_pixels = _replace_background_values(img_arr)
        if replaced_pixels > 0:
            print(
                f'White-gray mosaic background cleanup: replaced {replaced_pixels} pixels '
                'of (255,255,255)/(204,204,204) with black'
            )

    gray = _convert_to_gray(img_arr, grayscale_mode=grayscale_mode)
    print(f'Grayscale conversion mode: {grayscale_mode}')

    gray_u16 = (np.clip(gray, 0, 1) * 65535).astype(np.uint16)
    io.imsave(gray_output_path, gray_u16)

    scale = float(input_res) / float(target_res)
    gray_resampled = transform.rescale(gray, scale, preserve_range=True, anti_aliasing=True).astype(np.float32)
    resampled_u16 = (np.clip(gray_resampled, 0, 1) * 65535).astype(np.uint16)
    io.imsave(resampled_output_path, resampled_u16)

    print(f'Gray image saved to {gray_output_path}')
    print(f'Resampled gray image saved to {resampled_output_path}')
    print(f'Resample scale: {scale:.6f} ({input_res} -> {target_res} um/px)')

    tissue_mask = _extract_brain_mask(gray_resampled)
    io.imsave(mask_output_path, (tissue_mask.astype(np.uint8) * 255))

    if brain_layout not in ('auto', 'whole', 'left', 'right'):
        raise ValueError('brain_layout must be one of: auto, whole, left, right')
    if brain_layout == 'auto':
        resolved_brain_layout, layout_diagnostics = _infer_brain_layout(tissue_mask)
    else:
        resolved_brain_layout = brain_layout
        layout_diagnostics = {
            'reason': 'manual_layout_selected',
            'inference_skipped': True,
        }

    if resolved_brain_layout in ('left', 'right'):
        anatomical_midline_col, midline_diagnostics = _estimate_anatomical_midline(
            tissue_mask, resolved_brain_layout,
        )
    else:
        anatomical_midline_col = None
        midline_diagnostics = {
            'reason': 'whole_brain_uses_image_center',
            'estimation_skipped': True,
        }
    print(
        f'Brain layout: requested={brain_layout}, resolved={resolved_brain_layout}'
    )
    if resolved_brain_layout == 'whole':
        print('Whole brain selected; midline estimation skipped and image center used')
    elif anatomical_midline_col is not None:
        print(f'Anatomical midline detected at input column {anatomical_midline_col}')
    else:
        print('No confident internal midline detected; using layout fallback alignment')

    canvas_output_path = str(mask_output_path).replace('_mask.tif', '_masked_on_1140x800_black.tif')
    mask_canvas_output_path = str(mask_output_path).replace('_mask.tif', '_mask_on_1140x800_black.tif')
    _, _ = _save_masked_brain_on_canvas(
        gray_resampled,
        tissue_mask,
        canvas_output_path,
        canvas_width=CANVAS_WIDTH,
        canvas_height=CANVAS_HEIGHT,
        mask_canvas_output_path=mask_canvas_output_path,
        brain_layout=resolved_brain_layout,
        anatomical_midline_col=anatomical_midline_col,
    )
    
    # Generate high-resolution canvas at original image resolution
    fullres_canvas_output_path = str(mask_output_path).replace('_mask.tif', '_masked_on_fullres_black.tif')
    _, fullres_canvas_shape = _save_fullres_canvas(
        gray,
        tissue_mask,
        gray_resampled.shape,
        fullres_canvas_output_path,
        canvas_width=CANVAS_WIDTH,
        canvas_height=CANVAS_HEIGHT,
        brain_layout=resolved_brain_layout,
        anatomical_midline_col=anatomical_midline_col,
    )

    nz = np.argwhere(tissue_mask)
    if nz.size > 0:
        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0)
        mask_bbox = [int(r0), int(c0), int(r1), int(c1)]
    else:
        mask_bbox = None

    record_json_path = str(mask_output_path).replace('_mask.tif', '_step1_record.json')
    payload = {
        'input_path': str(input_path),
        'outputs': {
            'gray_path': str(gray_output_path),
            'resampled_path': str(resampled_output_path),
            'mask_path': str(mask_output_path),
            'masked_canvas_path': str(canvas_output_path),
            'mask_canvas_path': str(mask_canvas_output_path),
            'masked_fullres_canvas_path': str(fullres_canvas_output_path),
        },
        'params': {
            'input_res_um_per_px': float(input_res),
            'target_res_um_per_px': float(target_res),
            'replace_background_values': bool(replace_background_values),
            'rotation_degrees_counterclockwise': int(rotation),
            'brain_layout_requested': str(brain_layout),
            'brain_layout_resolved': str(resolved_brain_layout),
            'grayscale_mode': str(grayscale_mode),
            'resample_scale': float(scale),
            'canvas_width': CANVAS_WIDTH,
            'canvas_height': CANVAS_HEIGHT,
            'fullres_canvas_shape': [int(fullres_canvas_shape[0]), int(fullres_canvas_shape[1])],
        },
        'stats': {
            'input_shape': [int(v) for v in original_input_shape],
            'processed_input_shape': [int(v) for v in img_arr.shape],
            'background_replaced_pixels': int(replaced_pixels),
            'gray_shape': [int(v) for v in gray.shape],
            'resampled_shape': [int(v) for v in gray_resampled.shape],
            'fullres_canvas_shape': [int(fullres_canvas_shape[0]), int(fullres_canvas_shape[1])],
            'mask_coverage': float(tissue_mask.mean()),
            'mask_bbox_r0c0r1c1': mask_bbox,
            'brain_layout_diagnostics': layout_diagnostics,
            'anatomical_midline_diagnostics': midline_diagnostics,
        },
    }
    with open(record_json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'Tissue mask saved to {mask_output_path}')
    print(f'Mask coverage: {tissue_mask.mean() * 100:.2f}%')
    print(f'Masked brain on 1140x800 black canvas saved to {canvas_output_path}')
    print(f'Masked brain on fullres black canvas saved to {fullres_canvas_output_path}')
    print(f'Mask on 1140x800 black canvas saved to {mask_canvas_output_path}')
    print(f'Step1 record json saved to {record_json_path}')

    _cleanup_images(img_arr, gray, gray_resampled, tissue_mask)

    return record_json_path


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path.parent / '01.preprocess'
    output_path.mkdir(parents=True, exist_ok=True)
    image_name = input_path.stem

    try:
        record_json_path = preprocess_image(
            input_path=input_path,
            gray_output_path=output_path / f'{image_name}_gray.tif',
            resampled_output_path=output_path / f'{image_name}_resampled.tif',
            mask_output_path=output_path / f'{image_name}_mask.tif',
            input_res=args.input_res,
            target_res=args.target_res,
            replace_background_values=args.replace_background_values,
            rotation=args.rotation,
            brain_layout=args.brain_layout,
            grayscale_mode=args.grayscale_mode,
        )
    finally:
        _cleanup_images()
