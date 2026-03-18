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
        return None, dense_weight_tif, {'coverage': coverage, 'threshold': 0.0}

    stats = {
        'coverage': coverage,
        'weight_min': float(weight.min()),
        'weight_max': float(weight.max()),
        'weight_mean': float(weight.mean()),
        'weight_style': 'direct dense_weight continuous weight',
    }
    return weight.astype(np.float32), dense_weight_tif, stats


def nonlinear_register(
    fixed_path,
    moving_path,
    output_path,
    transform_path,
    step2_record_json=None,
    random_seed=2026,
    sitk_threads=1,
):
    _set_global_determinism(seed=random_seed, sitk_threads=sitk_threads)

    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    fixed_arr = sitk.GetArrayFromImage(fixed)

    metric_history = []

    dense_weight_arr, dense_weight_tif_used, dense_weight_stats = _load_dense_weight_from_step2(
        step2_record_json,
        target_shape=fixed_arr.shape,
    )
    metric_weight_tif = None
    fixed_for_metric = fixed
    moving_for_metric = moving
    if dense_weight_arr is not None:
        weight_for_metric = np.clip(dense_weight_arr, 0.0, 1.0)
        weight_for_metric = 0.12 + 0.88 * np.power(weight_for_metric, 1.6)
        moving_arr = sitk.GetArrayFromImage(moving)
        fixed_weighted_arr = fixed_arr * weight_for_metric
        moving_weighted_arr = moving_arr * weight_for_metric
        fixed_for_metric = _to_sitk_with_like(fixed_weighted_arr, fixed)
        moving_for_metric = _to_sitk_with_like(moving_weighted_arr, moving)
        metric_weight_tif = str(output_path).replace('_nonlinear.tif', '_metric_weight_from_dense_weight.tif')
        io.imsave(metric_weight_tif, (np.clip(weight_for_metric, 0, 1) * 65535).astype(np.uint16))
        print(f'Using direct dense-weight as continuous metric weight for step3, coverage={dense_weight_stats["coverage"]*100:.1f}%')
    elif dense_weight_tif_used is not None:
        print(f'Dense-weight file found but invalid for weighting: {dense_weight_tif_used}')

    bspline_init = sitk.BSplineTransformInitializer(
        image1=fixed_for_metric,
        transformDomainMeshSize=[6, 6],
        order=3,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.3, seed=int(random_seed) + 17)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=0.1,
        numberOfIterations=480,
        convergenceMinimumValue=1e-8,
        convergenceWindowSize=20,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([8, 6, 4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([4, 3, 2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(bspline_init, inPlace=True)

    def _iter_cb():
        metric_history.append((reg.GetOptimizerIteration(), float(reg.GetMetricValue()), 'bspline'))

    reg.AddCommand(sitk.sitkIterationEvent, _iter_cb)
    bspline_tx = reg.Execute(fixed_for_metric, moving_for_metric)

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
        'metric_weight_tif': str(metric_weight_tif),
        'dense_weight_stats': dense_weight_stats,
        'bspline_final_metric': float(reg.GetMetricValue()),
        'bspline_stop': reg.GetOptimizerStopConditionDescription(),
        'bspline_iterations': int(reg.GetOptimizerIteration()),
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
    parser.add_argument('--step2-record-json', default=None, help='step2 record json path (optional)')
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
        step2_record_json=args.step2_record_json,
        random_seed=args.random_seed,
        sitk_threads=args.sitk_threads,
    )
