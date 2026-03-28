import os
import argparse
import random
import csv
import json
import numpy as np
import SimpleITK as sitk
from skimage import io, transform
from matplotlib import cm


def _set_global_determinism(seed=2026, sitk_threads=1):
    np.random.seed(int(seed))
    random.seed(int(seed))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(sitk_threads))


def _to_u16(arr):
    arr = arr.astype(np.float32)
    amin, amax = float(arr.min()), float(arr.max())
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    return (np.clip(arr, 0, 1) * 65535).astype(np.uint16)


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
    return (rgba[..., :3] * 255).astype(np.uint8)


def _save_overlay_rgb_tif(fixed_arr, moved_arr, out_path, alpha=0.45, cmap_name='turbo'):
    base = _gray_to_u8(fixed_arr)
    base_rgb = np.stack([base, base, base], axis=-1)
    allen_rgb = _apply_colormap_rgb(moved_arr, cmap_name=cmap_name)
    out = (base_rgb.astype(np.float32) * (1.0 - alpha) + allen_rgb.astype(np.float32) * alpha).astype(np.uint8)
    io.imsave(out_path, out)


def _to_sitk_with_like(arr, like_image):
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.CopyInformation(like_image)
    return img


def _to_mask_sitk_with_like(mask_arr, like_image):
    img = sitk.GetImageFromArray((mask_arr > 0).astype(np.uint8))
    img.CopyInformation(like_image)
    return img


def _load_mask(fixed_mask_path, target_shape):
    if fixed_mask_path is None or not os.path.exists(str(fixed_mask_path)):
        return None, None, None

    mask = io.imread(str(fixed_mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.float32)

    if mask.shape != target_shape:
        mask = transform.resize(
            mask,
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)

    mask = (mask > 0.5).astype(np.float32)
    coverage = float(np.mean(mask > 0))
    if coverage < 0.01:
        return None, str(fixed_mask_path), {'coverage': coverage, 'reason': 'too_sparse'}

    stats = {
        'coverage': coverage,
        'mask_min': float(mask.min()),
        'mask_max': float(mask.max()),
        'mask_mean': float(mask.mean()),
    }
    return mask.astype(np.float32), str(fixed_mask_path), stats


def _run_bspline_stage(
    fixed_img,
    moving_img,
    mesh_size,
    stage_name,
    metric_history,
    initial_transform=None,
    sampling_percentage=0.25,
    learning_rate=0.08,
    number_of_iterations=260,
    shrink_factors=None,
    smoothing_sigmas=None,
):
    if shrink_factors is None:
        shrink_factors = [8, 4, 2, 1]
    if smoothing_sigmas is None:
        smoothing_sigmas = [4, 2, 1, 0]

    if initial_transform is None:
        bspline_init = sitk.BSplineTransformInitializer(
            image1=fixed_img,
            transformDomainMeshSize=[int(mesh_size), int(mesh_size)],
            order=3,
        )
    else:
        bspline_init = sitk.BSplineTransform(initial_transform)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(reg.REGULAR)
    reg.SetMetricSamplingPercentage(float(sampling_percentage))

    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=float(learning_rate),
        numberOfIterations=int(number_of_iterations),
        convergenceMinimumValue=1e-8,
        convergenceWindowSize=20,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrink_factors)
    reg.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(bspline_init, inPlace=True)

    def _iter_cb():
        metric_history.append((reg.GetOptimizerIteration(), float(reg.GetMetricValue()), stage_name))

    reg.AddCommand(sitk.sitkIterationEvent, _iter_cb)
    stage_tx = reg.Execute(fixed_img, moving_img)
    return {
        'name': stage_name,
        'transform': stage_tx,
        'final_metric': float(reg.GetMetricValue()),
        'stop': reg.GetOptimizerStopConditionDescription(),
        'iterations': int(reg.GetOptimizerIteration()),
    }


def _load_dense_weight_from_step2(step2_record_json, target_shape):
    if step2_record_json is None or not os.path.exists(str(step2_record_json)):
        return None, None, None

    with open(step2_record_json, 'r') as f:
        s2 = json.load(f)

    dense_weight_tif = s2.get('dense_weight_tif')
    if dense_weight_tif is None or not os.path.exists(str(dense_weight_tif)):
        return None, dense_weight_tif, None

    weight = io.imread(str(dense_weight_tif)).astype(np.float32)
    if weight.ndim == 3:
        weight = weight[..., 0]

    wmax = float(weight.max())
    if wmax > 1e-8:
        weight = weight / wmax
    weight = np.clip(weight, 0.0, 1.0)

    if weight.shape != target_shape:
        weight = transform.resize(
            weight,
            target_shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
        weight = np.clip(weight, 0.0, 1.0)

    coverage = float(np.mean(weight > 1e-6))
    if coverage < 0.01:
        return None, dense_weight_tif, {'coverage': coverage, 'reason': 'too_sparse'}

    stats = {
        'coverage': coverage,
        'weight_min': float(weight.min()),
        'weight_max': float(weight.max()),
        'weight_mean': float(weight.mean()),
    }
    return weight.astype(np.float32), dense_weight_tif, stats


def nonlinear_register(
    fixed_path,
    moving_path,
    output_path,
    transform_path,
    step2_record_json=None,
    mesh_size=10,
    sampling_percentage=0.25,
    random_seed=2026,
    sitk_threads=1,
):
    _set_global_determinism(seed=random_seed, sitk_threads=sitk_threads)

    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    fixed_arr = sitk.GetArrayFromImage(fixed)

    metric_history = []

    dense_weight_arr = None
    dense_weight_tif_used = None
    dense_weight_stats = None
    metric_weight_tif = None

    dense_weight_arr, dense_weight_tif_used, dense_weight_stats = _load_dense_weight_from_step2(
        step2_record_json,
        target_shape=fixed_arr.shape,
    )
    if dense_weight_arr is not None:
        soft_weight = np.clip(dense_weight_arr, 0.0, 1.0)
        soft_weight = np.where(soft_weight > 0.1, soft_weight, 0.0)

        fixed_weighted_arr = fixed_arr * soft_weight
        fixed_weighted_arr = _to_sitk_with_like(fixed_weighted_arr, fixed)

        metric_weight_tif = str(output_path).replace('_nonlinear.tif', '_weight_from_step2_dense_weight.tif')
        io.imsave(metric_weight_tif, (np.clip(soft_weight, 0, 1) * 65535).astype(np.uint16))
        print(f'Using weighted metric in step3 stage1, coverage={dense_weight_stats["coverage"]*100:.1f}%')

    stage_summaries = []

    stage1 = _run_bspline_stage(
        fixed_img=fixed_weighted_arr,
        moving_img=moving,
        mesh_size=mesh_size,
        stage_name='bspline_coarse',
        metric_history=metric_history,
        sampling_percentage=sampling_percentage,
        learning_rate=0.005,
        number_of_iterations=200,
        shrink_factors=[8, 4, 2, 1],
        smoothing_sigmas=[4, 2, 1, 0],
    )
    stage_summaries.append({k: v for k, v in stage1.items() if k != 'transform'})
    
    stage2 = _run_bspline_stage(
        fixed_img=fixed,
        moving_img=moving,
        mesh_size=mesh_size,
        stage_name='bspline_refine',
        metric_history=metric_history,
        initial_transform=stage1['transform'],
        sampling_percentage=sampling_percentage,
        learning_rate=0.01,
        number_of_iterations=180,
        shrink_factors=[4, 2, 1],
        smoothing_sigmas=[2, 1, 0],
    )
    stage_summaries.append({k: v for k, v in stage2.items() if k != 'transform'})
    bspline_tx = stage2['transform']
    final_stage = stage2

    result = sitk.Resample(moving, fixed, bspline_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())

    result_arr = sitk.GetArrayFromImage(result)

    sitk.WriteTransform(bspline_tx, transform_path)
    output_path = str(output_path)
    io.imsave(output_path, _to_u16(result_arr))

    post_overlay_rgb_tif = output_path.replace('_nonlinear.tif', '_overlay_on_step1_rgb.tif')
    _save_overlay_rgb_tif(fixed_arr, result_arr, post_overlay_rgb_tif, alpha=0.45, cmap_name='turbo')

    metrics_csv = output_path.replace('_nonlinear.tif', '_metric_history.csv')
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'metric', 'stage'])
        writer.writerows(metric_history)

    record_json = output_path.replace('_nonlinear.tif', '_step3_record.json')
    payload = {
        'fixed_path': str(fixed_path),
        'moving_path': str(moving_path),
        'output_path': str(output_path),
        'transform_path': str(transform_path),
        'deformation_chain_forward': [
            {'step': 'step3_bspline', 'transform_path': str(transform_path), 'type': 'BSplineTransform'},
        ],
        'step2_record_json': str(step2_record_json),
        'random_seed': int(random_seed),
        'sitk_threads': int(sitk_threads),
        'dense_weight_tif_used': str(dense_weight_tif_used),
        'dense_weight_stats': dense_weight_stats,
        'metric_weight_tif': str(metric_weight_tif),
        'mesh_size': int(mesh_size),
        'sampling_percentage': float(sampling_percentage),
        'bspline_stages': stage_summaries,
        'bspline_final_metric': float(final_stage['final_metric']),
        'bspline_stop': final_stage['stop'],
        'bspline_iterations': int(final_stage['iterations']),
        'adjusted_allen_gray_tif': str(output_path),
        'overlay_rgb_tif': str(post_overlay_rgb_tif),
        'metric_history_csv': str(metrics_csv),
    }
    with open(record_json, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'Nonlinear registered image saved to {output_path}')
    print(f'BSpline transform saved to {transform_path}')
    print(f'Adjusted Allen gray tif saved to {output_path}')
    print(f'Overlay RGB tif saved to {post_overlay_rgb_tif}')
    print(f'Step3 record saved to {record_json}')
    return result, record_json


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Step3: nonlinear registration refinement based on step2 affine result')
    parser.add_argument('--fixed-path', required=True, help='fixed image path (step1 resampled output)')
    parser.add_argument('--moving-path', required=True, help='moving image path (step2 output)')
    parser.add_argument('--output-path', required=True, help='nonlinear registration result output tif path')
    parser.add_argument('--transform-path', required=True, help='B-spline transform output path (h5)')
    parser.add_argument('--fixed-mask-path', default=None, help='step1 mask path for metric masking')
    parser.add_argument('--use-metric-mask', type=int, default=1, choices=[0, 1], help='whether to use step1 mask in step3 metric (1=yes, 0=no)')
    parser.add_argument('--metric-mode', type=str, default='weighted', choices=['raw', 'weighted'], help='step3 stage1 metric mode')
    parser.add_argument('--weight-floor', type=float, default=0.25, help='weight floor for weighted stage1 metric to preserve internal structures')
    parser.add_argument('--step2-record-json', default=None, help='step2 record json path (optional)')
    parser.add_argument('--mesh-size', type=int, default=10, help='B-spline mesh size per dimension (smaller = stiffer transform)')
    parser.add_argument('--sampling-percentage', type=float, default=0.25, help='metric sampling percentage')
    parser.add_argument('--random-seed', type=int, default=2026, help='random seed (for reproducible registration)')
    parser.add_argument('--sitk-threads', type=int, default=1, help='SimpleITK global thread count (recommended 1 for reproducibility)')
    return parser


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tx_dir = os.path.dirname(args.transform_path)
    if tx_dir:
        os.makedirs(tx_dir, exist_ok=True)

    result, record_json = nonlinear_register(
        fixed_path=args.fixed_path,
        moving_path=args.moving_path,
        output_path=args.output_path,
        transform_path=args.transform_path,
        fixed_mask_path=args.fixed_mask_path,
        use_metric_mask=bool(args.use_metric_mask),
        metric_mode=args.metric_mode,
        weight_floor=args.weight_floor,
        step2_record_json=args.step2_record_json,
        mesh_size=args.mesh_size,
        sampling_percentage=args.sampling_percentage,
        random_seed=args.random_seed,
        sitk_threads=args.sitk_threads,
    )
