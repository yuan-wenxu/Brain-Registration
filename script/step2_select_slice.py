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
from skimage import io, transform, filters
import SimpleITK as sitk
from matplotlib import cm
from matplotlib import pyplot as plt
from skimage import color
from step1_preprocess import _extract_brain_mask, _save_masked_brain_on_canvas
from utils.registration import rigid_register, affine_register, bspline_register


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
        help='specify atlas slice number; if not provided, search automatically')
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


def _init_slice_worker(moving_path, fixed_search, dense_weight_map, mask_search):
    """Initialize immutable worker data once instead of sending it per slice."""
    global _PROCESS_WORKER_STATE
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    _PROCESS_WORKER_STATE = {
        'atlas': np.load(str(moving_path), mmap_mode='r'),
        'fixed_search': fixed_search,
        'dense_weight_map': dense_weight_map,
        'mask_search': mask_search,
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
    scores = _register_rigid_affine_bspline_for_score(
        state['fixed_search'],
        moving,
        idx,
        seed=seed,
        dense_weight_map=state['dense_weight_map'],
        score_mask=state['mask_search'],
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

    fixed_path = _resolve(outputs['masked_canvas_path'])
    mask_path = _resolve(outputs['mask_canvas_path'])
    for name, path in (('fixed image', fixed_path), ('mask', mask_path)):
        if not path.exists():
            raise FileNotFoundError(f'Step1 {name} does not exist: {path}')
    return fixed_path, mask_path, record


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


def _register_rigid_affine_bspline_for_score(fixed_arr, moving_arr, idx, seed=0, dense_weight_map=None, score_mask=None):
    fixed = _to_sitk(fixed_arr)
    moving_mask = _extract_brain_mask(moving_arr)
    moving, _ = _save_masked_brain_on_canvas(
        moving_arr,
        moving_mask,
        output_path=None,
        canvas_width=fixed_arr.shape[1],
        canvas_height=fixed_arr.shape[0],
    )
    moving = _to_sitk(moving)

    rigid = rigid_register(
        fixed, moving,
        sampling_seed=seed + 7,
        learning_rate=0.1,
        number_of_iterations=140,
        shrink_factors=(4, 2, 1),
        smoothing_sigmas=(2, 1, 0),
    )
    affine = affine_register(
        fixed, rigid['image'],
        center=rigid['transform'].GetCenter(),
        sampling_seed=seed + 13,
        learning_rate=0.15,
        number_of_iterations=220,
        shrink_factors=(4, 2, 1),
        smoothing_sigmas=(2, 1, 0),
    )
    bspline = bspline_register(
        fixed, affine['image'],
        mesh_size=6,
        sampling_strategy='random',
        sampling_percentage=0.2,
        sampling_seed=seed + 23,
        learning_rate=0.1,
        number_of_iterations=140,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1, 0),
    )
    moved_bspline_arr = sitk.GetArrayFromImage(bspline['image'])

    edge_ncc = _edge_ncc(fixed_arr, moved_bspline_arr, mask=score_mask)
    if dense_weight_map is None:
        dense_weight_map = _build_dense_focus_weight(fixed_arr, percentile=85.0, gamma=2.0)
    dense_ncc = _weighted_ncc(fixed_arr.astype(np.float32), moved_bspline_arr.astype(np.float32), dense_weight_map)

    score = bspline['metric'] - 0.12 * edge_ncc - 0.22 * dense_ncc

    return {
        'rigid_mi': rigid['metric'],
        'rigid_iterations': rigid['iterations'],
        'rigid_stop_reason': rigid['stop_reason'],
        'affine_mi': affine['metric'],
        'affine_iterations': affine['iterations'],
        'affine_stop_reason': affine['stop_reason'],
        'bspline_mi': bspline['metric'],
        'bspline_iterations': bspline['iterations'],
        'bspline_stop_reason': bspline['stop_reason'],
        'edge_ncc': edge_ncc,
        'dense_ncc': float(dense_ncc),
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

    sorted_slices = sorted(per_slice_best.keys())
    slice_scores = np.array([per_slice_best[idx] for idx in sorted_slices], dtype=np.float32)
    sigma = max(0.5, float(neighbor_smooth_sigma))
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / np.sum(kernel)
    padded = np.pad(slice_scores, (radius, radius), mode='edge')
    smoothed_arr = np.convolve(padded, kernel, mode='valid')
    smoothed = {idx: float(smoothed_arr[i]) for i, idx in enumerate(sorted_slices)}

    for row in records:
        idx = int(row['slice_idx'])
        slice_score_smoothed = float(smoothed[idx])
        row['slice_score_smoothed'] = slice_score_smoothed


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
    fixed_path, mask_path, _ = _load_step1_inputs(step1_record_json)
    workers = max(1, int(workers))
    _set_global_determinism(seed=random_seed, sitk_threads=1)

    fixed_full = _load_fixed_2d(fixed_path)
    atlas = np.load(moving_path, mmap_mode='r')
    if atlas.ndim != 3:
        raise ValueError('moving_path must be a 3D atlas npy file')

    total = atlas.shape[0]
    center = 650 if atlas_slice is None else int(atlas_slice)
    center = max(0, min(total - 1, center))

    fixed_search = transform.rescale(fixed_full, search_resize_max, 
                                     preserve_range=True, anti_aliasing=True).astype(np.float32)
    dense_weight_full = _build_dense_focus_weight(fixed_full, percentile=85.0, gamma=2.0)

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
        print('No tissue mask provided; using full-image weight map.')

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

    effective_radius = slice_search_radius if atlas_slice is None else 0
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
            initargs=(str(moving_path), fixed_search, dense_weight_map, mask_search),
        )

    def _evaluate_slice(idx, stage, seed_offset, scale):
        moving = _normalize_01(atlas[idx])
        if scale < 1.0:
            moving = transform.rescale(moving, scale, preserve_range=True, anti_aliasing=True).astype(np.float32)
        else:
            moving = moving.astype(np.float32)
        s = _register_rigid_affine_bspline_for_score(
            fixed_search,
            moving,
            idx,
            seed=int(random_seed) + seed_offset,
            dense_weight_map=dense_weight_map,
            score_mask=mask_search,
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
    _apply_neighbor_score_regularization(records, neighbor_smooth_sigma=neighbor_smooth_sigma)

    ranked_slices = _rank_slices_by_smoothed_score(records)
    topk_slices = [int(item[0]) for item in ranked_slices[:3]]

    refine_candidates = set()
    fine_step = max(1, slice_search_step // 6)
    fine_radius = max(10, int(slice_search_step * 0.75))
    for c in topk_slices:
        for i in range(max(coarse_left, c - fine_radius), min(coarse_right, c + fine_radius) + 1, fine_step):
            refine_candidates.add(i)
    refine_candidates = sorted(refine_candidates)
    print(f'Refine candidates around top-k: n={len(refine_candidates)}')

    _evaluate_batch(refine_candidates, 'refine', seed_offset=10000, scale=search_resize_max)
    _apply_neighbor_score_regularization(records, neighbor_smooth_sigma=neighbor_smooth_sigma)

    ranked_slices = _rank_slices_by_smoothed_score(records)
    ultra_centers = sorted([int(item[0]) for item in ranked_slices[:2]])
    ultra_radius = max(6, slice_search_step // 2)
    ultra_candidates = set()
    for c in ultra_centers:
        for i in range(max(coarse_left, c - ultra_radius), min(coarse_right, c + ultra_radius) + 1):
            ultra_candidates.add(i)
    ultra_candidates = sorted(ultra_candidates)
    print(f'Ultra-refine candidates around best-2: n={len(ultra_candidates)}')
    _evaluate_batch(ultra_candidates, 'ultra', seed_offset=20000, scale=search_resize_max)
    if process_pool is not None:
        process_pool.shutdown(wait=True)
    _apply_neighbor_score_regularization(records, neighbor_smooth_sigma=neighbor_smooth_sigma)

    ranked_slices = _rank_slices_by_smoothed_score(records)
    if len(ranked_slices) == 0:
        raise RuntimeError('No slice scores were computed in Step2.')
    best = ranked_slices[0][1][2]
    best_slice = int(best['slice_idx'])
    print(
        f'Best slice selected: {best_slice}, '
        f'score_smoothed={best["slice_score_smoothed"]:.6f}, '
        f'base_score={best["score"]:.6f}, bspline_mi={best["bspline_mi"]:.6f}'
    )

    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(step1_record_json).parent.parent / '02.select_slice'
    )
    output_path.mkdir(parents=True, exist_ok=True)
    moving_best_full = _normalize_01(atlas[best_slice])
    selected_canvas, _ = _save_masked_brain_on_canvas(
        moving_best_full,
        np.ones_like(moving_best_full, dtype=bool),
        output_path=None,
        canvas_width=fixed_full.shape[1],
        canvas_height=fixed_full.shape[0],
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
                'affine_mi', 'affine_iterations', 'affine_stop_reason',
                'bspline_mi', 'bspline_iterations', 'bspline_stop_reason',
                'edge_ncc', 'dense_ncc', 'score', 'slice_score_smoothed',
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
            'coarse_candidates': coarse_candidates,
            'refine_candidates': refine_candidates,
            'ultra_candidates': ultra_candidates,
        },
        'best_candidate_metrics': best,
        'selected_slice_path': str(selected_slice_path),
        'metrics_csv': str(metrics_csv),
        'slice_score_curve_png': str(slice_score_curve_png),
        'dense_weight_tif': str(dense_weight_tif),
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
