import os
import argparse
import random
import csv
import json
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from skimage import io, transform, filters
import SimpleITK as sitk
from matplotlib import cm
from matplotlib import pyplot as plt
from skimage import color
from step1_preprocess import _extract_brain_mask, _save_masked_brain_on_canvas
from utils.registration import rigid_register, affine_register, compose_transforms


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Step2: select the best matching atlas slice')
    parser.add_argument(
        '--step1-record-json',
        required=True,
        help='Step1 output record JSON; fixed image and mask are read automatically')
    parser.add_argument(
        '--moving-path',
        required=True,
        help='3D atlas npy path (e.g., ara_nissl_10.npy)')
    parser.add_argument(
        '--output-path',
        default=None,
        help='output directory; default: <step1-json>.parent.parent/02.select_slice')
    parser.add_argument(
        '--atlas-slice',
        type=int,
        default=None,
        help='optional expected atlas slice used as the search/prior center')
    parser.add_argument(
        '--slice-search-radius',
        type=int,
        default=200,
        help='slice search radius for step2')
    parser.add_argument(
        '--slice-search-step',
        type=int,
        default=20,
        help='slice search step for step2')
    parser.add_argument('--search-resize-max', type=float, default=0.6, help='scale factor used to resize images during slice search')
    parser.add_argument('--random-seed', type=int, default=42, help='random seed (for reproducible registration)')
    parser.add_argument('--workers', type=int, default=8, help='number of slice-scoring processes working concurrently (recommended 2-8)')
    parser.add_argument('--neighbor-smooth-sigma', type=float, default=3.0, help='sigma of Gaussian smoothing kernel used for neighbor slice score regularization')
    return parser


_PROCESS_WORKER_STATE = {}


def _init_slice_worker(moving_path, fixed_search, dense_weight_map, mask_search, brain_layout='whole', cut_column=None):
    """Initialize immutable worker data once instead of sending it per slice."""
    global _PROCESS_WORKER_STATE
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    _PROCESS_WORKER_STATE = {
        'atlas': np.load(str(moving_path), mmap_mode='r'),
        'fixed_search': fixed_search,
        'dense_weight_map': dense_weight_map,
        'mask_search': mask_search,
        'brain_layout': brain_layout,
        'cut_column': cut_column,
    }


def _evaluate_slice_process(task):
    """Process-pool entry point; must remain at module scope for spawning."""
    idx, stage, seed, scale = task
    state = _PROCESS_WORKER_STATE
    moving = _normalize_01(state['atlas'][idx])
    if scale < 1.0:
        moving = transform.rescale(
            moving, scale, preserve_range=True, anti_aliasing=True,
        ).astype(np.float32)
    else:
        moving = moving.astype(np.float32)
    scores = _register_rigid_affine_for_score(
        state['fixed_search'],
        moving,
        idx,
        seed=seed,
        dense_weight_map=state['dense_weight_map'],
        score_mask=state['mask_search'],
        brain_layout=state['brain_layout'],
        cut_column=state['cut_column'],
    )
    return {'slice_idx': idx, 'stage': stage, **scores}


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

def _set_global_determinism(seed=42, sitk_threads=1):
    np.random.seed(int(seed))
    random.seed(int(seed))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(sitk_threads))


def _normalize_01(arr):
    arr = arr.astype(np.float32)
    amin, amax = float(arr.min()), float(arr.max())
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    return arr


def _load_fixed_2d(path):
    arr = io.imread(path)
    if arr.ndim == 3:
        arr = color.rgb2gray(arr)
    return _normalize_01(arr)


def _load_step1_inputs(step1_record_json):
    record_path = Path(step1_record_json)
    if not record_path.exists():
        raise FileNotFoundError(f'Step1 record JSON does not exist: {record_path}')
    with open(record_path, 'r') as f:
        record = json.load(f)
    outputs = record.get('outputs', {})
    required = ('masked_canvas_path', 'mask_canvas_path')
    missing = [key for key in required if not outputs.get(key)]
    if missing:
        raise ValueError(f'Step1 record JSON is missing outputs fields: {missing}')

    def _resolve(path_value):
        path = Path(path_value)
        if not path.is_absolute() and not path.exists():
            path = record_path.parent / path
        return path

    fixed_path = _resolve(
        outputs.get('downstream_canvas_path') or outputs['masked_canvas_path']
    )
    mask_path = _resolve(outputs['mask_canvas_path'])
    for name, path in (('fixed image', fixed_path), ('mask', mask_path)):
        if not path.exists():
            raise FileNotFoundError(f'Step1 {name} does not exist: {path}')
    brain_layout = record.get('params', {}).get('brain_layout_resolved', 'whole')
    return fixed_path, mask_path, record, brain_layout


def _hemisphere_registration_bbox(image_shape, brain_layout, cut_column=None):
    """Return a full-height crop split at the specimen's detected cut edge."""
    height, width = image_shape[:2]
    split = width // 2 if cut_column is None else int(round(cut_column))
    split = min(max(1, split), width - 1)
    if brain_layout == 'right':
        return 0, split, height, width
    if brain_layout == 'left':
        return 0, 0, height, split + 1
    return None


def _atlas_canvas_for_layout(atlas_slice, brain_layout, canvas_shape, cut_column=None):
    """Place Allen in full-canvas coordinates, then crop at the specimen edge."""
    atlas_slice = np.asarray(atlas_slice, dtype=np.float32)
    atlas_mask = _extract_brain_mask(atlas_slice)
    canvas_height, canvas_width = canvas_shape[:2]
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    mask_canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    source = atlas_slice * atlas_mask.astype(np.float32)
    copy_height = min(source.shape[0], canvas_height)
    copy_width = min(source.shape[1], canvas_width)
    source_r0 = max(0, (source.shape[0] - copy_height) // 2)
    source_c0 = max(0, (source.shape[1] - copy_width) // 2)
    target_r0 = max(0, (canvas_height - copy_height) // 2)
    destination_c0 = max(0, (canvas_width - copy_width) // 2)
    source_region = source[
        source_r0:source_r0 + copy_height,
        source_c0:source_c0 + copy_width,
    ]
    source_mask_region = atlas_mask[
        source_r0:source_r0 + copy_height,
        source_c0:source_c0 + copy_width,
    ]
    canvas[
        target_r0:target_r0 + copy_height,
        destination_c0:destination_c0 + copy_width,
    ] = source_region
    mask_canvas[
        target_r0:target_r0 + copy_height,
        destination_c0:destination_c0 + copy_width,
    ] = source_mask_region.astype(np.uint8) * 255
    bbox = _hemisphere_registration_bbox(canvas.shape, brain_layout, cut_column)
    if bbox is not None:
        _, c0, _, c1 = bbox
        if c0 > 0:
            canvas[:, :c0] = 0
            mask_canvas[:, :c0] = 0
        if c1 < canvas_width:
            canvas[:, c1:] = 0
            mask_canvas[:, c1:] = 0
    return canvas, mask_canvas


def _to_sitk(arr):
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing([1.0, 1.0])
    return img


def _edge_ncc(a, b, mask=None):
    ga = filters.sobel(a)
    gb = filters.sobel(b)
    if mask is not None:
        m = mask > 0.5
        if np.count_nonzero(m) < 16:
            return 0.0
        ga = ga[m]
        gb = gb[m]
    ga = ga - ga.mean()
    gb = gb - gb.mean()
    denom = np.sqrt((ga * ga).sum()) * np.sqrt((gb * gb).sum())
    if denom <= 1e-8:
        return 0.0
    return float((ga * gb).sum() / denom)


def _build_dense_focus_weight(arr, percentile=75.0, gamma=2.0):
    x = _normalize_01(arr)
    threshold = float(np.percentile(x, percentile))
    denom = max(1e-6, 1.0 - threshold)
    focus = np.clip((x - threshold) / denom, 0.0, 1.0)
    weight = 0.1 + 0.9 * np.power(focus, gamma)
    return weight.astype(np.float32)


def _weighted_ncc(a, b, w):
    wsum = float(np.sum(w))
    if wsum <= 1e-8:
        return 0.0
    ma = float(np.sum(w * a) / wsum)
    mb = float(np.sum(w * b) / wsum)
    da = a - ma
    db = b - mb
    num = float(np.sum(w * da * db))
    den = np.sqrt(float(np.sum(w * da * da)) * float(np.sum(w * db * db)))
    if den <= 1e-8:
        return 0.0
    return num / den


def _normalized_mutual_information(a, b, mask=None, bins=64):
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.asarray(mask) > 0.5
    av = np.asarray(a, dtype=np.float32)[valid]
    bv = np.asarray(b, dtype=np.float32)[valid]
    if av.size < 100:
        return 0.0
    hist, _, _ = np.histogram2d(av, bv, bins=int(bins), range=((0, 1), (0, 1)))
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    pxy = hist / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz_xy = pxy > 0
    nz_x = px > 0
    nz_y = py > 0
    hxy = -float(np.sum(pxy[nz_xy] * np.log(pxy[nz_xy])))
    hx = -float(np.sum(px[nz_x] * np.log(px[nz_x])))
    hy = -float(np.sum(py[nz_y] * np.log(py[nz_y])))
    if hx + hy <= 1e-12:
        return 0.0
    return float(np.clip((hx + hy - hxy) * 2.0 / (hx + hy), 0.0, 1.0))


def _mask_dice(a, b):
    aa = np.asarray(a) > 0.5
    bb = np.asarray(b) > 0.5
    denominator = int(np.count_nonzero(aa)) + int(np.count_nonzero(bb))
    if denominator == 0:
        return 0.0
    return 2.0 * float(np.count_nonzero(aa & bb)) / float(denominator)


def _boundary_similarity(a, b, distance_scale=8.0):
    aa = np.asarray(a) > 0.5
    bb = np.asarray(b) > 0.5
    edge_a = aa ^ binary_erosion(aa)
    edge_b = bb ^ binary_erosion(bb)
    if not np.any(edge_a) or not np.any(edge_b):
        return 0.0
    distance_to_a = distance_transform_edt(~edge_a)
    distance_to_b = distance_transform_edt(~edge_b)
    symmetric_distance = 0.5 * (
        float(np.mean(distance_to_b[edge_a]))
        + float(np.mean(distance_to_a[edge_b]))
    )
    return float(np.exp(-symmetric_distance / float(distance_scale)))


def _affine_deformation_penalty(affine_transform):
    transform_value = affine_transform
    while not hasattr(transform_value, 'GetMatrix') and hasattr(
        transform_value, 'GetNumberOfTransforms'
    ):
        if transform_value.GetNumberOfTransforms() == 0:
            return float('inf')
        transform_value = transform_value.GetNthTransform(0)
    if not hasattr(transform_value, 'GetMatrix'):
        return float('inf')
    matrix = np.asarray(
        transform_value.GetMatrix(), dtype=np.float64,
    ).reshape(2, 2)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    singular_values = np.clip(singular_values, 1e-6, None)
    scale_penalty = float(np.mean(np.abs(np.log(singular_values))))
    anisotropy_penalty = float(abs(np.log(singular_values[0] / singular_values[1])))
    normalized_columns = matrix / np.maximum(
        np.linalg.norm(matrix, axis=0, keepdims=True), 1e-8,
    )
    shear_penalty = float(abs(np.dot(normalized_columns[:, 0], normalized_columns[:, 1])))
    return scale_penalty + 0.5 * anisotropy_penalty + 0.5 * shear_penalty


def _deterministic_internal_similarity(fixed_arr, moved_arr, fixed_mask):
    fixed_s4 = gaussian_filter(fixed_arr.astype(np.float32), sigma=4.0)
    moved_s4 = gaussian_filter(moved_arr.astype(np.float32), sigma=4.0)
    fixed_s8 = gaussian_filter(fixed_arr.astype(np.float32), sigma=8.0)
    moved_s8 = gaussian_filter(moved_arr.astype(np.float32), sigma=8.0)
    nmi = _normalized_mutual_information(fixed_s4, moved_s4, mask=fixed_mask)
    ncc_s4 = _weighted_ncc(fixed_s4, moved_s4, fixed_mask.astype(np.float32))
    ncc_s8 = _weighted_ncc(fixed_s8, moved_s8, fixed_mask.astype(np.float32))
    multiscale_ncc = 0.6 * ncc_s4 + 0.4 * ncc_s8
    return float(nmi), float(multiscale_ncc), fixed_s4, moved_s4


def _mind_descriptor_2d(image, patch_sigma=1.5):
    """Compact modality-independent local self-similarity descriptor."""
    source = gaussian_filter(np.asarray(image, dtype=np.float32), sigma=2.0)
    shifts = ((0, 2), (0, -2), (2, 0), (-2, 0), (2, 2), (2, -2), (-2, 2), (-2, -2))
    distances = []
    for dy, dx in shifts:
        shifted = np.roll(source, shift=(dy, dx), axis=(0, 1))
        distances.append(gaussian_filter(np.square(source - shifted), sigma=patch_sigma))
    distances = np.stack(distances, axis=0)
    local_variance = np.mean(distances, axis=0, keepdims=True)
    descriptor = np.exp(-distances / np.maximum(local_variance, 1e-6))
    return descriptor.astype(np.float32)


def _mind_similarity(a, b, mask):
    descriptor_a = _mind_descriptor_2d(a)
    descriptor_b = _mind_descriptor_2d(b)
    weight = (np.asarray(mask) > 0.5).astype(np.float32)
    similarities = [
        _weighted_ncc(descriptor_a[channel], descriptor_b[channel], weight)
        for channel in range(descriptor_a.shape[0])
    ]
    return float(np.mean(similarities))


def _signed_mask_distance(mask, clip_distance=50.0):
    binary = np.asarray(mask) > 0.5
    signed = distance_transform_edt(binary) - distance_transform_edt(~binary)
    signed = np.clip(signed, -float(clip_distance), float(clip_distance))
    return ((signed + float(clip_distance)) / (2.0 * float(clip_distance))).astype(np.float32)


def _register_rigid_affine_for_score(fixed_arr, moving_arr, idx, seed=0, dense_weight_map=None, score_mask=None, brain_layout='whole', cut_column=None):
    moving_canvas, moving_mask_canvas = _atlas_canvas_for_layout(
        moving_arr,
        brain_layout,
        fixed_arr.shape,
        cut_column=cut_column,
    )

    # Step1 aligns the detected anatomical cut line to the canvas center. Keep
    # the complete superior/inferior extent and split both images vertically at
    # that line; do not derive the registration window from the tissue-mask bbox.
    bbox = _hemisphere_registration_bbox(fixed_arr.shape, brain_layout, cut_column)
    if bbox is not None:
        r0, c0, r1, c1 = bbox
        fixed_reg = fixed_arr[r0:r1, c0:c1].copy()
        moving_reg = moving_canvas[r0:r1, c0:c1].copy()
        moving_mask_reg = moving_mask_canvas[r0:r1, c0:c1].copy()
        score_mask_reg = score_mask[r0:r1, c0:c1] if score_mask is not None else None
        dense_weight_reg = dense_weight_map[r0:r1, c0:c1] if dense_weight_map is not None else None
    else:
        fixed_reg = fixed_arr
        moving_reg = moving_canvas
        moving_mask_reg = moving_mask_canvas
        score_mask_reg = score_mask
        dense_weight_reg = dense_weight_map

    fixed = _to_sitk(fixed_reg)
    moving = _to_sitk(moving_reg)
    fixed_mask = score_mask_reg > 0.5 if score_mask_reg is not None else fixed_reg > 0

    if brain_layout in ('left', 'right'):
        fixed_distance = _to_sitk(_signed_mask_distance(fixed_mask))
        moving_distance = _to_sitk(_signed_mask_distance(moving_mask_reg))
        try:
            shape_rigid = rigid_register(
                fixed_distance,
                moving_distance,
                sampling_strategy='regular',
                sampling_percentage=0.5,
                sampling_seed=seed + 7,
                learning_rate=0.08,
                number_of_iterations=180,
                shrink_factors=(4, 2, 1),
                smoothing_sigmas=(2, 1, 0),
            )
            rigid_image = sitk.Resample(
                moving,
                fixed,
                shape_rigid['transform'],
                sitk.sitkLinear,
                0.0,
                moving.GetPixelID(),
            )
            rigid = dict(shape_rigid)
            rigid['image'] = rigid_image
            rigid['name'] = 'shape_rigid'
            rigid_initializer = 'shape_distance'
        except RuntimeError:
            rigid = rigid_register(
                fixed,
                moving,
                sampling_seed=seed + 7,
                learning_rate=0.08,
                number_of_iterations=180,
                shrink_factors=(4, 2, 1),
                smoothing_sigmas=(2, 1, 0),
            )
            rigid_initializer = 'intensity_fallback'
    else:
        rigid = rigid_register(
            fixed, moving,
            sampling_seed=seed + 7,
            learning_rate=0.1,
            number_of_iterations=140,
            shrink_factors=(4, 2, 1),
            smoothing_sigmas=(2, 1, 0),
        )
        rigid_initializer = 'intensity'
    try:
        affine = affine_register(
            fixed, rigid['image'],
            center=rigid['transform'].GetCenter(),
            sampling_seed=seed + 13,
            learning_rate=0.15,
            number_of_iterations=220,
            shrink_factors=(4, 2, 1),
            smoothing_sigmas=(2, 1, 0),
        )
        affine_status = 'optimized'
    except RuntimeError:
        affine = {
            'name': 'affine_fallback',
            'transform': sitk.AffineTransform(2),
            'image': rigid['image'],
            'metric': float('nan'),
            'iterations': 0,
            'stop_reason': 'affine registration failed; rigid result retained',
        }
        affine_status = 'rigid_fallback'
    rigid_arr = sitk.GetArrayFromImage(rigid['image']).astype(np.float32)
    moved_arr = sitk.GetArrayFromImage(affine['image']).astype(np.float32)
    moving_mask_image = _to_sitk((moving_mask_reg > 0).astype(np.float32))
    rigid_mask_image = sitk.Resample(
        moving_mask_image,
        fixed,
        rigid['transform'],
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.sitkFloat32,
    )
    rigid_affine = compose_transforms(rigid['transform'], affine['transform'])
    moved_mask_image = sitk.Resample(
        moving_mask_image,
        fixed,
        rigid_affine,
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.sitkFloat32,
    )
    rigid_mask = sitk.GetArrayFromImage(rigid_mask_image) > 0.5
    moved_mask = sitk.GetArrayFromImage(moved_mask_image) > 0.5
    rigid_nmi, rigid_ncc, _, _ = _deterministic_internal_similarity(
        fixed_reg, rigid_arr, fixed_mask,
    )
    nmi, low_frequency_ncc, fixed_low, moved_low = _deterministic_internal_similarity(
        fixed_reg, moved_arr, fixed_mask,
    )
    tissue_dice = _mask_dice(fixed_mask, moved_mask)
    rigid_tissue_dice = _mask_dice(fixed_mask, rigid_mask)
    boundary_similarity = _boundary_similarity(fixed_mask, moved_mask)
    affine_penalty = _affine_deformation_penalty(affine['transform'])
    edge_ncc = _edge_ncc(fixed_low, moved_low, mask=fixed_mask)
    rigid_mind_ncc = _mind_similarity(fixed_reg, rigid_arr, fixed_mask)
    mind_ncc = _mind_similarity(fixed_reg, moved_arr, fixed_mask)

    rigid_internal = 0.45 * rigid_nmi + 0.55 * rigid_ncc
    affine_internal = 0.45 * nmi + 0.55 * low_frequency_ncc
    rigid_weight = 0.65 if brain_layout in ('left', 'right') else 0.40
    internal_similarity = rigid_weight * rigid_internal + (1.0 - rigid_weight) * affine_internal
    shape_similarity = 0.5 * rigid_tissue_dice + 0.5 * tissue_dice
    score = (
        -0.85 * internal_similarity
        -0.05 * shape_similarity
        -0.05 * boundary_similarity
        +0.05 * affine_penalty
    )

    return {
        'rigid_mi': rigid['metric'],
        'rigid_iterations': rigid['iterations'],
        'rigid_stop_reason': rigid['stop_reason'],
        'rigid_initializer': rigid_initializer,
        'affine_mi': affine['metric'],
        'affine_iterations': affine['iterations'],
        'affine_stop_reason': affine['stop_reason'],
        'affine_status': affine_status,
        'rigid_normalized_mi': float(rigid_nmi),
        'rigid_multiscale_ncc': float(rigid_ncc),
        'rigid_tissue_dice': float(rigid_tissue_dice),
        'rigid_mind_ncc': float(rigid_mind_ncc),
        'normalized_mi': float(nmi),
        'low_frequency_ncc': float(low_frequency_ncc),
        'tissue_dice': float(tissue_dice),
        'boundary_similarity': float(boundary_similarity),
        'affine_deformation_penalty': float(affine_penalty),
        'internal_similarity': float(internal_similarity),
        'mind_ncc': float(mind_ncc),
        'edge_ncc': edge_ncc,
        'score': float(score),
    }


def _save_weight_map_tif(weight_map, output_path):
    w = np.clip(weight_map.astype(np.float32), 0.0, 1.0)
    w_u16 = (np.clip(w, 0, 1) * 65535).astype(np.uint16)
    io.imsave(output_path, w_u16)


def _apply_neighbor_score_regularization(records, neighbor_smooth_sigma=3.0):
    if not records:
        return

    per_slice_best = {}
    for row in records:
        idx = int(row['slice_idx'])
        score = float(row['score'])
        if idx not in per_slice_best or score < per_slice_best[idx]:
            per_slice_best[idx] = score

    sorted_slices = np.asarray(sorted(per_slice_best.keys()), dtype=np.float64)
    slice_scores = np.asarray(
        [per_slice_best[int(idx)] for idx in sorted_slices], dtype=np.float64,
    )
    sigma = max(0.5, float(neighbor_smooth_sigma))
    # Candidate spacing changes between coarse, refine, and ultra-refine stages.
    # Weight by the actual atlas-index distance instead of treating the sparse
    # score array as uniformly sampled. Normalize each row independently so
    # search boundaries are not biased by duplicated edge values.
    distances = sorted_slices[:, None] - sorted_slices[None, :]
    weights = np.exp(-0.5 * np.square(distances / sigma))
    weights[np.abs(distances) > 3.0 * sigma] = 0.0
    weight_sums = np.sum(weights, axis=1)
    smoothed_arr = (weights @ slice_scores) / np.maximum(weight_sums, 1e-12)
    smoothed = {
        int(idx): float(smoothed_arr[i]) for i, idx in enumerate(sorted_slices)
    }

    for row in records:
        idx = int(row['slice_idx'])
        slice_score_smoothed = float(smoothed[idx])
        row['slice_score_smoothed'] = slice_score_smoothed


def _update_composite_scores(records, brain_layout='whole'):
    """Compute an absolute image-only score independent of the search range."""
    if not records:
        return
    for row in records:
        quality = (
            0.53 * float(row['mind_ncc'])
            + 0.30 * float(row['rigid_normalized_mi'])
            + 0.08 * float(row['rigid_multiscale_ncc'])
            + 0.06 * float(row['boundary_similarity'])
            - 0.02 * float(row['affine_deformation_penalty'])
            + 0.01 * float(row['tissue_dice'])
        )
        row['score'] = -float(quality)


def _rank_slices_by_smoothed_score(records):
    per_slice = {}
    for row in records:
        idx = int(row['slice_idx'])
        smoothed = float(row['slice_score_smoothed'])
        raw = float(row['score'])
        if idx not in per_slice or (smoothed, raw) < (per_slice[idx][0], per_slice[idx][1]):
            per_slice[idx] = (smoothed, raw, row)
    ranked = sorted(per_slice.items(), key=lambda x: (x[1][0], x[1][1]))
    return ranked


def _rank_slices_by_raw_score(records):
    per_slice = {}
    for row in records:
        idx = int(row['slice_idx'])
        raw = float(row['score'])
        if idx not in per_slice or raw < per_slice[idx][0]:
            per_slice[idx] = (raw, row)
    return sorted(per_slice.items(), key=lambda item: item[1][0])


def _top_candidate_slices(records, smoothed_count, raw_count):
    """Keep both stable neighborhood minima and strong individual matches."""
    smoothed = [
        int(item[0])
        for item in _rank_slices_by_smoothed_score(records)[:smoothed_count]
    ]
    raw = [
        int(item[0])
        for item in _rank_slices_by_raw_score(records)[:raw_count]
    ]
    return list(dict.fromkeys(smoothed + raw))


def _save_slice_score_curve(records, output_png_path):
    ranked = _rank_slices_by_smoothed_score(records)
    if len(ranked) == 0:
        return
    by_slice = sorted(ranked, key=lambda item: int(item[0]))
    x = [int(item[0]) for item in by_slice]
    raw_y = [float(item[1][1]) for item in by_slice]
    smooth_y = [float(item[1][0]) for item in by_slice]
    plt.figure(figsize=(5, 4), dpi=300)
    plt.plot(x, raw_y, color='#4f81bd', linewidth=1.4, alpha=0.75, label='raw_score')
    plt.plot(x, smooth_y, color='#c0504d', linewidth=2.0, alpha=0.95, label='smoothed_score')
    plt.xlabel('Atlas Slice Index', fontsize=10)
    plt.ylabel('Score (lower is better)', fontsize=10)
    plt.title('Step2 Slice Score Curve', fontsize=12)
    plt.grid(True, alpha=0.25)
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(output_png_path, bbox_inches='tight')
    plt.close()


def select_slice(
    step1_record_json,
    moving_path,
    output_path=None,
    atlas_slice=None,
    slice_search_radius=200,
    slice_search_step=20,
    search_resize_max=0.6,
    random_seed=42,
    workers=1,
    neighbor_smooth_sigma=3.0,
):
    fixed_path, mask_path, step1_record, brain_layout = _load_step1_inputs(step1_record_json)
    workers = max(1, int(workers))
    _set_global_determinism(seed=random_seed, sitk_threads=1)

    fixed_full = _load_fixed_2d(fixed_path)
    cut_column = step1_record.get('stats', {}).get('cut_edge_canvas_col')
    atlas = np.load(moving_path, mmap_mode='r')
    if atlas.ndim != 3:
        raise ValueError('moving_path must be a 3D atlas npy file')

    if brain_layout in ('left', 'right'):
        cut_description = (
            f'specimen cut edge column {cut_column}'
            if cut_column is not None
            else f'canvas center fallback column {fixed_full.shape[1] // 2}'
        )
        print(
            f'Half-brain layout detected: {brain_layout}; atlas will be cropped '
            f'at {cut_description} for scoring'
        )
    else:
        print(f'Brain layout: {brain_layout}')

    total = atlas.shape[0]
    center = 650 if atlas_slice is None else int(atlas_slice)
    center = max(0, min(total - 1, center))

    # load mask
    mask_search = None
    if mask_path is not None and os.path.exists(str(mask_path)):
        mask_full = io.imread(str(mask_path))
        mask_full = (mask_full > 127).astype(np.float32)
        if mask_full.shape != fixed_full.shape:
            mask_full = transform.resize(
                mask_full, fixed_full.shape, order=0,
                preserve_range=True, anti_aliasing=False,
            ).astype(np.float32)

        mask_search = transform.rescale(mask_full, search_resize_max, order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
        mask_search = (mask_search > 0.5).astype(np.float32)

        print(f'Tissue mask applied to weight map, coverage={mask_full.mean()*100:.1f}%')
    else:
        mask_full = None
        print('No tissue mask provided; using full-image weight map.')

    fixed_scoring_full = fixed_full

    fixed_search = transform.rescale(
        fixed_scoring_full,
        search_resize_max,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    cut_column_search = None
    if cut_column is not None:
        cut_column_search = float(cut_column) * fixed_search.shape[1] / fixed_full.shape[1]
    dense_weight_full = _build_dense_focus_weight(
        fixed_scoring_full, percentile=85.0, gamma=2.0,
    )

    if dense_weight_full.shape != fixed_search.shape:
        dense_weight_map = transform.resize(
            dense_weight_full,
            fixed_search.shape,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
    else:
        dense_weight_map = dense_weight_full.copy().astype(np.float32)
    if mask_search is not None:
        dense_weight_map = dense_weight_map * mask_search
    dense_weight_map = np.clip(dense_weight_map, 0.0, 1.0)

    effective_radius = slice_search_radius
    coarse_left = max(0, center - effective_radius)
    coarse_right = min(total - 1, center + effective_radius)
    coarse_candidates = list(range(coarse_left, coarse_right + 1, slice_search_step))
    if center not in coarse_candidates:
        coarse_candidates.append(center)
    coarse_candidates = sorted(set(coarse_candidates))

    print(f'Atlas shape: {atlas.shape}, center slice: {center}')
    print(f'Coarse range: [{coarse_left}, {coarse_right}], step={slice_search_step}, n={len(coarse_candidates)}')

    records = []
    evaluated = {}
    process_pool = None
    if workers > 1:
        process_pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_slice_worker,
            initargs=(str(moving_path), fixed_search, dense_weight_map, mask_search, brain_layout, cut_column_search),
        )

    def _evaluate_slice(idx, stage, seed_offset, scale):
        moving = _normalize_01(atlas[idx])
        if scale < 1.0:
            moving = transform.rescale(moving, scale, preserve_range=True, anti_aliasing=True).astype(np.float32)
        else:
            moving = moving.astype(np.float32)
        s = _register_rigid_affine_for_score(
            fixed_search,
            moving,
            idx,
            seed=int(random_seed) + seed_offset,
            dense_weight_map=dense_weight_map,
            score_mask=mask_search,
            brain_layout=brain_layout,
            cut_column=cut_column_search,
        )
        return {'slice_idx': idx, 'stage': stage, **s}

    def _evaluate_batch(indices, stage, seed_offset, scale):
        pending = [idx for idx in indices if idx not in evaluated]
        if not pending:
            return
        if workers <= 1:
            for idx in pending:
                row = _evaluate_slice(idx, stage, seed_offset, scale)
                records.append(row)
                evaluated[idx] = row
            return

        active_workers = min(workers, len(pending))
        print(
            f'Process-parallel evaluating {len(pending)} slices at stage={stage} '
            f'with workers={active_workers}'
        )
        futures = {
            process_pool.submit(
                _evaluate_slice_process,
                (idx, stage, int(random_seed) + seed_offset, scale),
            ): idx
            for idx in pending
        }
        try:
            for future in as_completed(futures):
                row = future.result()
                records.append(row)
                evaluated[row['slice_idx']] = row
        except Exception:
            for future in futures:
                future.cancel()
            process_pool.shutdown(wait=True, cancel_futures=True)
            raise

    _evaluate_batch(coarse_candidates, 'coarse', seed_offset=0, scale=search_resize_max)
    _update_composite_scores(
        records,
        brain_layout=brain_layout,
    )
    _apply_neighbor_score_regularization(records, neighbor_smooth_sigma=neighbor_smooth_sigma)

    topk_slices = _top_candidate_slices(records, smoothed_count=3, raw_count=3)
    if brain_layout in ('left', 'right'):
        topk_slices = list(dict.fromkeys(topk_slices + [center]))

    refine_candidates = set()
    fine_step = max(1, slice_search_step // 6)
    fine_radius = max(10, int(slice_search_step * 0.75))
    for c in topk_slices:
        for i in range(max(coarse_left, c - fine_radius), min(coarse_right, c + fine_radius) + 1, fine_step):
            refine_candidates.add(i)
    refine_candidates = sorted(refine_candidates)
    print(f'Refine candidates around top-k: n={len(refine_candidates)}')

    # Use identical metric samples in every stage so scores remain comparable.
    _evaluate_batch(refine_candidates, 'refine', seed_offset=0, scale=search_resize_max)
    _update_composite_scores(records, brain_layout=brain_layout)
    _apply_neighbor_score_regularization(records, neighbor_smooth_sigma=neighbor_smooth_sigma)

    ultra_centers = sorted(
        _top_candidate_slices(records, smoothed_count=2, raw_count=2)
    )
    ultra_radius = max(6, slice_search_step // 2)
    ultra_candidates = set()
    for c in ultra_centers:
        for i in range(max(coarse_left, c - ultra_radius), min(coarse_right, c + ultra_radius) + 1):
            ultra_candidates.add(i)
    ultra_candidates = sorted(ultra_candidates)
    print(f'Ultra-refine candidates around best-2: n={len(ultra_candidates)}')
    _evaluate_batch(ultra_candidates, 'ultra', seed_offset=0, scale=search_resize_max)
    if process_pool is not None:
        process_pool.shutdown(wait=True)
    _update_composite_scores(records, brain_layout=brain_layout)
    _apply_neighbor_score_regularization(records, neighbor_smooth_sigma=neighbor_smooth_sigma)

    ranked_slices = _rank_slices_by_smoothed_score(records)
    if len(ranked_slices) == 0:
        raise RuntimeError('No slice scores were computed in Step2.')
    best = ranked_slices[0][1][2]
    best_slice = int(best['slice_idx'])
    print(
        f'Best slice selected: {best_slice}, '
        f'score_smoothed={best["slice_score_smoothed"]:.6f}, '
        f'base_score={best["score"]:.6f}, nmi={best["normalized_mi"]:.6f}, '
        f'dice={best["tissue_dice"]:.6f}'
    )

    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(step1_record_json).parent.parent / '02.select_slice'
    )
    output_path.mkdir(parents=True, exist_ok=True)
    moving_best_full = _normalize_01(atlas[best_slice])
    # Crop Allen at the straight cut edge detected from the experimental image.
    selected_canvas, _ = _atlas_canvas_for_layout(
        moving_best_full,
        brain_layout,
        fixed_full.shape,
        cut_column=cut_column,
    )
    selected_slice_path = output_path / 'selected_slice.tif'
    io.imsave(selected_slice_path, (np.clip(selected_canvas, 0, 1) * 65535).astype(np.uint16))

    dense_weight_tif = output_path / "dense_weight.tif"
    dense_weight_full = dense_weight_full * mask_full if mask_search is not None else dense_weight_full
    _save_weight_map_tif(dense_weight_full, dense_weight_tif)

    slice_score_curve_png = output_path / "slice_score_curve.png"
    _save_slice_score_curve(records, slice_score_curve_png)

    metrics_csv = output_path / "slice_search_metrics.csv"
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'slice_idx', 'stage',
                'rigid_mi', 'rigid_iterations', 'rigid_stop_reason',
                'rigid_initializer',
                'affine_mi', 'affine_iterations', 'affine_stop_reason',
                'affine_status',
                'rigid_normalized_mi', 'rigid_multiscale_ncc',
                'rigid_tissue_dice', 'rigid_mind_ncc',
                'normalized_mi', 'low_frequency_ncc', 'tissue_dice',
                'boundary_similarity', 'affine_deformation_penalty',
                'internal_similarity', 'mind_ncc', 'edge_ncc', 'score',
                'slice_score_smoothed',
            ],
        )
        writer.writeheader()
        for r in sorted(records, key=lambda x: (x['slice_idx'], x['stage'])):
            writer.writerow(r)

    record_json = output_path / "step2_record.json"
    payload = {
        'fixed_path': str(fixed_path),
        'step1_record_json': str(step1_record_json),
        'moving_path': str(moving_path),
        'selected_slice': best_slice,
        'search': {
            'center': center,
            'slice_search_radius': slice_search_radius,
            'slice_search_step': slice_search_step,
            'refine_step': fine_step,
            'refine_radius': fine_radius,
            'ultra_refine_radius': ultra_radius,
            'search_resize_max': search_resize_max,
            'random_seed': int(random_seed),
            'workers': int(workers),
            'parallel_backend': 'process' if workers > 1 else 'serial',
            'worker_sitk_threads': 1,
            'remove_grid': bool(step1_record.get('params', {}).get('remove_grid', False)),
            'cut_edge_canvas_col': cut_column,
            'coarse_candidates': coarse_candidates,
            'refine_candidates': refine_candidates,
            'ultra_candidates': ultra_candidates,
        },
        'best_candidate_metrics': best,
        'selected_slice_path': str(selected_slice_path),
        'metrics_csv': str(metrics_csv),
        'slice_score_curve_png': str(slice_score_curve_png),
        'dense_weight_tif': str(dense_weight_tif),
        'grid_removed_fixed_path': step1_record.get('outputs', {}).get('degridded_canvas_path'),
        'grid_removal_diagnostics': step1_record.get('stats', {}).get('grid_removal_diagnostics'),
    }
    with open(record_json, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'Selected atlas slice saved to {selected_slice_path}')
    print(f'Dense weight tif saved to {dense_weight_tif}')
    print(f'Slice score curve saved to {slice_score_curve_png}')
    print(f'Step2 record saved to {record_json}')

    _cleanup_images(
        fixed_full,
        atlas,
        fixed_search,
        dense_weight_full,
        dense_weight_map,
        mask_search,
        mask_full if 'mask_full' in locals() else None,
        selected_canvas,
    )

    return selected_canvas, best_slice, record_json


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()

    try:
        selected_arr, selected_slice, info = select_slice(
            step1_record_json=args.step1_record_json,
            moving_path=args.moving_path,
            output_path=args.output_path,
            atlas_slice=args.atlas_slice,
            slice_search_radius=args.slice_search_radius,
            slice_search_step=args.slice_search_step,
            search_resize_max=args.search_resize_max,
            random_seed=args.random_seed,
            workers=args.workers,
            neighbor_smooth_sigma=args.neighbor_smooth_sigma,
        )
    finally:
        _cleanup_images()
