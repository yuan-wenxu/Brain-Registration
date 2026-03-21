import os
import argparse
import random
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from skimage import io, transform, filters
import SimpleITK as sitk
from matplotlib import cm
from matplotlib import pyplot as plt
from step1_preprocess import _extract_brain_mask, _save_masked_brain_on_canvas

def _set_global_determinism(seed=2026, sitk_threads=1):
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
        from skimage import color
        arr = color.rgb2gray(arr)
    return _normalize_01(arr)


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
        canvas_width=768,
        canvas_height=555,
    )
    moving = _to_sitk(moving)

    rigid_init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg1 = sitk.ImageRegistrationMethod()
    reg1.SetMetricAsMattesMutualInformation(50)
    reg1.SetMetricSamplingStrategy(reg1.RANDOM)
    reg1.SetMetricSamplingPercentage(0.25, seed=seed + 7)
    reg1.SetInterpolator(sitk.sitkLinear)
    reg1.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=140,
                                       convergenceMinimumValue=1e-8, convergenceWindowSize=20)
    reg1.SetOptimizerScalesFromPhysicalShift()
    reg1.SetShrinkFactorsPerLevel([4, 2, 1])
    reg1.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg1.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg1.SetInitialTransform(rigid_init, inPlace=False)
    rigid_tx = reg1.Execute(fixed, moving)
    rigid_tx = sitk.Euler2DTransform(rigid_tx.GetNthTransform(0))
    rigid_mi = float(reg1.GetMetricValue())
    rigid_iterations = int(reg1.GetOptimizerIteration())
    rigid_stop_reason = reg1.GetOptimizerStopConditionDescription()

    moving_after_rigid = sitk.Resample(moving, fixed, rigid_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())

    affine_tx = sitk.AffineTransform(2)
    affine_tx.SetCenter(rigid_tx.GetCenter())

    reg2 = sitk.ImageRegistrationMethod()
    reg2.SetMetricAsMattesMutualInformation(64)
    reg2.SetMetricSamplingStrategy(reg2.RANDOM)
    reg2.SetMetricSamplingPercentage(0.25, seed=seed + 13)
    reg2.SetInterpolator(sitk.sitkLinear)
    reg2.SetOptimizerAsGradientDescent(learningRate=0.15, numberOfIterations=220,
                                       convergenceMinimumValue=1e-8, convergenceWindowSize=20)
    reg2.SetOptimizerScalesFromPhysicalShift()
    reg2.SetShrinkFactorsPerLevel([4, 2, 1])
    reg2.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg2.SetInitialTransform(affine_tx, inPlace=True)
    affine_tx = reg2.Execute(fixed, moving_after_rigid)
    affine_mi = float(reg2.GetMetricValue())
    affine_iterations = int(reg2.GetOptimizerIteration())
    affine_stop_reason = reg2.GetOptimizerStopConditionDescription()
    moved_affine = sitk.Resample(moving_after_rigid, fixed, affine_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())

    bspline_init = sitk.BSplineTransformInitializer(fixed, [6, 6], order=3)
    reg3 = sitk.ImageRegistrationMethod()
    reg3.SetMetricAsMattesMutualInformation(64)
    reg3.SetMetricSamplingStrategy(reg3.RANDOM)
    reg3.SetMetricSamplingPercentage(0.2, seed=seed + 23)
    reg3.SetInterpolator(sitk.sitkLinear)
    reg3.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=140,
                                       convergenceMinimumValue=1e-8, convergenceWindowSize=20)
    reg3.SetOptimizerScalesFromPhysicalShift()
    reg3.SetShrinkFactorsPerLevel([2, 1])
    reg3.SetSmoothingSigmasPerLevel([1, 0])
    reg3.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg3.SetInitialTransform(bspline_init, inPlace=False)
    bspline_tx = reg3.Execute(fixed, moved_affine)
    bspline_mi = float(reg3.GetMetricValue())
    bspline_iterations = int(reg3.GetOptimizerIteration())
    bspline_stop_reason = reg3.GetOptimizerStopConditionDescription()

    moved_bspline = sitk.Resample(moved_affine, fixed, bspline_tx, sitk.sitkLinear, 0.0, moved_affine.GetPixelID())
    moved_bspline_arr = sitk.GetArrayFromImage(moved_bspline)

    edge_ncc = _edge_ncc(fixed_arr, moved_bspline_arr, mask=score_mask)
    if dense_weight_map is None:
        dense_weight_map = _build_dense_focus_weight(fixed_arr, percentile=85.0, gamma=2.0)
    dense_ncc = _weighted_ncc(fixed_arr.astype(np.float32), moved_bspline_arr.astype(np.float32), dense_weight_map)

    score = bspline_mi - 0.12 * edge_ncc - 0.22 * dense_ncc

    return {
        'rigid_mi': rigid_mi,
        'rigid_iterations': rigid_iterations,
        'rigid_stop_reason': rigid_stop_reason,
        'affine_mi': affine_mi,
        'affine_iterations': affine_iterations,
        'affine_stop_reason': affine_stop_reason,
        'bspline_mi': bspline_mi,
        'bspline_iterations': bspline_iterations,
        'bspline_stop_reason': bspline_stop_reason,
        'edge_ncc': edge_ncc,
        'dense_ncc': float(dense_ncc),
        'score': float(score),
    }


def _final_affine_fullres(fixed_arr, moving_arr, seed=2026, fixed_mask_arr=None):
    fixed = _to_sitk(fixed_arr)
    moving_mask = _extract_brain_mask(moving_arr)
    moving, _ = _save_masked_brain_on_canvas(
        moving_arr,
        moving_mask,
        output_path=None,
    )
    moving = _to_sitk(moving)

    rigid_init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg_rigid = sitk.ImageRegistrationMethod()
    reg_rigid.SetMetricAsMattesMutualInformation(50)
    reg_rigid.SetMetricSamplingStrategy(reg_rigid.RANDOM)
    reg_rigid.SetMetricSamplingPercentage(0.25, seed=int(seed) + 101)
    reg_rigid.SetInterpolator(sitk.sitkLinear)
    reg_rigid.SetOptimizerAsGradientDescent(learningRate=0.05, numberOfIterations=280,
                                            convergenceMinimumValue=1e-8, convergenceWindowSize=20)
    reg_rigid.SetOptimizerScalesFromPhysicalShift()
    reg_rigid.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    reg_rigid.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
    reg_rigid.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg_rigid.SetInitialTransform(rigid_init, inPlace=False)

    rigid_tx = reg_rigid.Execute(fixed, moving)
    rigid_tx = sitk.Euler2DTransform(rigid_tx.GetNthTransform(0))
    rigid_mi = float(reg_rigid.GetMetricValue())
    rigid_iterations = int(reg_rigid.GetOptimizerIteration())
    moving_after_rigid = sitk.Resample(moving, fixed, rigid_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())

    affine_tx = sitk.AffineTransform(2)
    affine_tx.SetCenter(rigid_tx.GetCenter())

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(64)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed=int(seed) + 131)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=0.05, numberOfIterations=700,
                                      convergenceMinimumValue=1e-8, convergenceWindowSize=20)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(affine_tx, inPlace=False)

    affine_tx = reg.Execute(fixed, moving_after_rigid)
    affine_mi = float(reg.GetMetricValue())
    affine_iterations = int(reg.GetOptimizerIteration())
    moved = sitk.Resample(moving_after_rigid, fixed, affine_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
    moved_arr = sitk.GetArrayFromImage(moved)

    return moved_arr, rigid_tx, affine_tx, {
        'rigid_metric_mi': rigid_mi,
        'rigid_iterations': rigid_iterations,
        'rigid_stop_reason': reg_rigid.GetOptimizerStopConditionDescription(),
        'affine_metric_mi': affine_mi,
        'affine_iterations': affine_iterations,
        'affine_stop_reason': reg.GetOptimizerStopConditionDescription(),
    }


def _gray_to_u8(arr):
    arr = arr.astype(np.float32)
    amin, amax = float(arr.min()), float(arr.max())
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _apply_colormap_rgb(arr, cmap_name='turbo'):
    u8 = _gray_to_u8(arr)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(u8 / 255.0)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def _save_overlay_rgb_tif(fixed_arr, moved_arr, output_path, alpha=0.45, cmap_name='turbo'):
    base = _gray_to_u8(fixed_arr)
    base_rgb = np.stack([base, base, base], axis=-1)
    allen_rgb = _apply_colormap_rgb(moved_arr, cmap_name=cmap_name)
    out = (base_rgb.astype(np.float32) * (1.0 - alpha) + allen_rgb.astype(np.float32) * alpha).astype(np.uint8)
    io.imsave(output_path, out)


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
    plt.figure(figsize=(10, 4.8), dpi=140)
    plt.plot(x, raw_y, color='#4f81bd', linewidth=1.4, alpha=0.75, label='raw_score')
    plt.plot(x, smooth_y, color='#c0504d', linewidth=2.0, alpha=0.95, label='smoothed_score')
    plt.xlabel('Atlas Slice Index')
    plt.ylabel('Score (lower is better)')
    plt.title('Step2 Slice Score Curve')
    plt.grid(True, alpha=0.25)
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(output_png_path, bbox_inches='tight')
    plt.close()


def affine_register(
    fixed_path,
    moving_path,
    output_path,
    mask_path=None,
    atlas_slice=None,
    slice_search_radius=200,
    slice_search_step=20,
    search_resize_max=768,
    random_seed=2026,
    sitk_threads=1,
    search_workers=1,
    neighbor_smooth_sigma=3.0,
):
    workers = max(1, int(search_workers))
    search_stage_threads = 1 if workers > 1 else int(sitk_threads)
    _set_global_determinism(seed=random_seed, sitk_threads=search_stage_threads)

    fixed_full = _load_fixed_2d(fixed_path)
    atlas = np.load(moving_path)
    if atlas.ndim != 3:
        raise ValueError('moving_path must be a 3D atlas npy file')

    total = atlas.shape[0]
    center =650

    scale = min(1.0, search_resize_max / max(fixed_full.shape))
    if scale < 1.0:
        fixed_search = transform.rescale(fixed_full, scale, preserve_range=True, anti_aliasing=True).astype(np.float32)
    else:
        fixed_search = fixed_full.copy().astype(np.float32)
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
        if scale < 1.0:
            mask_search = transform.rescale(mask_full, scale, order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
            mask_search = (mask_search > 0.5).astype(np.float32)
        else:
            mask_search = mask_full.copy().astype(np.float32)

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

    coarse_left = max(0, center - slice_search_radius)
    coarse_right = min(total - 1, center + slice_search_radius)
    coarse_candidates = list(range(coarse_left, coarse_right + 1, slice_search_step))
    if center not in coarse_candidates:
        coarse_candidates.append(center)
    coarse_candidates = sorted(set(coarse_candidates))

    print(f'Atlas shape: {atlas.shape}, center slice: {center}')
    print(f'Coarse range: [{coarse_left}, {coarse_right}], step={slice_search_step}, n={len(coarse_candidates)}')

    records = []
    evaluated = {}

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

        max_workers = min(workers, len(pending))
        print(f'Parallel evaluating {len(pending)} slices at stage={stage} with workers={max_workers}')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_evaluate_slice, idx, stage, seed_offset, scale): idx
                for idx in pending
            }
            for future in as_completed(futures):
                row = future.result()
                records.append(row)
                evaluated[row['slice_idx']] = row

    _evaluate_batch(coarse_candidates, 'coarse', seed_offset=0, scale=scale)
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

    _evaluate_batch(refine_candidates, 'refine', seed_offset=10000, scale=scale)
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
    _evaluate_batch(ultra_candidates, 'ultra', seed_offset=20000, scale=scale)
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

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(sitk_threads))

    moving_best_full = _normalize_01(atlas[best_slice])
    reg_arr, rigid_tx, affine_tx, final_info = _final_affine_fullres(
        fixed_full,
        moving_best_full,
        seed=int(random_seed),
        fixed_mask_arr=mask_full if mask_search is not None else None,
    )

    reg_u16 = (np.clip(reg_arr, 0, 1) * 65535).astype(np.uint16)
    io.imsave(output_path, reg_u16)

    merged_tx = sitk.CompositeTransform(2)
    merged_tx.AddTransform(rigid_tx)
    merged_tx.AddTransform(affine_tx)
    if hasattr(merged_tx, 'FlattenTransform'):
        merged_tx.FlattenTransform()
    output_path = str(output_path)
    merged_tx_path = output_path.replace('_affine.tif', '_rigid_affine.h5')
    sitk.WriteTransform(merged_tx, merged_tx_path)

    overlay_rgb_tif = output_path.replace('_affine.tif', '_overlay_on_step1_rgb.tif')
    _save_overlay_rgb_tif(fixed_full, reg_arr, overlay_rgb_tif, alpha=0.45, cmap_name='turbo')

    dense_weight_tif = output_path.replace('_affine.tif', '_dense_weight.tif')
    _save_weight_map_tif(dense_weight_full, dense_weight_tif)

    slice_score_curve_png = output_path.replace('_affine.tif', '_slice_score_curve.png')
    _save_slice_score_curve(records, slice_score_curve_png)

    metrics_csv = output_path.replace('_affine.tif', '_slice_search_metrics.csv')
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

    record_json = output_path.replace('_affine.tif', '_step2_record.json')
    payload = {
        'fixed_path': str(fixed_path),
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
            'sitk_threads': int(sitk_threads),
            'search_workers': int(workers),
            'coarse_candidates': coarse_candidates,
            'refine_candidates': refine_candidates,
            'ultra_candidates': ultra_candidates,
        },
        'best_candidate_metrics': best,
        'final_rigid_metric_mi': final_info['rigid_metric_mi'],
        'final_rigid_iterations': final_info['rigid_iterations'],
        'final_rigid_stop_reason': final_info['rigid_stop_reason'],
        'final_affine_metric_mi': final_info['affine_metric_mi'],
        'final_affine_iterations': final_info['affine_iterations'],
        'final_affine_stop_reason': final_info['affine_stop_reason'],
        'merged_rigid_affine_transform_path': merged_tx_path,
        'deformation_chain_forward': [
            {'step': 'step2_rigid_affine_merged', 'transform_path': merged_tx_path, 'type': 'CompositeTransform'},
        ],
        'metrics_csv': metrics_csv,
        'slice_score_curve_png': slice_score_curve_png,
        'overlay_rgb_tif': overlay_rgb_tif,
        'dense_weight_tif': dense_weight_tif,
    }
    with open(record_json, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'Affine registered image saved to {output_path}')
    print(f'Overlay RGB tif saved to {overlay_rgb_tif}')
    print(f'Dense weight tif saved to {dense_weight_tif}')
    print(f'Slice score curve saved to {slice_score_curve_png}')
    print(f'Merged rigid+affine transform saved to {merged_tx_path}')
    print(f'Step2 record saved to {record_json}')

    return reg_arr, affine_tx, record_json


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Step2: affine registration to select best matching atlas slice')
    parser.add_argument('-f', '--fixed-path', required=True, help='fixed image path (step1 resampled output)')
    parser.add_argument('-m', '--moving-path', required=True, help='3D atlas npy path (e.g., ara_nissl_10.npy)')
    parser.add_argument('-o', '--output-path', required=True, help='affine registration output tif path')
    parser.add_argument('--mask-path', default=None, help='tissue mask tif (step1 output, _mask.tif); constrains dense weight to tissue')
    parser.add_argument('-as', '--atlas-slice', type=int, default=None, help='specify atlas slice number; if not provided, search automatically')
    parser.add_argument('-sr', '--slice-search-radius', type=int, default=200, help='slice search radius for step2')
    parser.add_argument('-ss', '--slice-search-step', type=int, default=20, help='slice search step for step2')
    parser.add_argument('-sm', '--search-resize-max', type=int, default=768, help='maximum side length of images during search stage')
    parser.add_argument('-rs', '--random-seed', type=int, default=2026, help='random seed (for reproducible registration)')
    parser.add_argument('-st', '--sitk-threads', type=int, default=1, help='SimpleITK global thread count (recommend 1 for reproducibility)')
    parser.add_argument('-sw', '--search-workers', type=int, default=1, help='parallel workers for slice search (recommend 2-8)')
    parser.add_argument('--neighbor-smooth-sigma', type=float, default=3.0, help='sigma of Gaussian smoothing kernel used for neighbor slice score regularization')
    return parser


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    reg_arr, affine_tx, info = affine_register(
        fixed_path=args.fixed_path,
        moving_path=args.moving_path,
        output_path=args.output_path,
        mask_path=args.mask_path,
        atlas_slice=args.atlas_slice,
        slice_search_radius=args.slice_search_radius,
        slice_search_step=args.slice_search_step,
        search_resize_max=args.search_resize_max,
        random_seed=args.random_seed,
        sitk_threads=args.sitk_threads,
        search_workers=args.search_workers,
        neighbor_smooth_sigma=args.neighbor_smooth_sigma,
    )
