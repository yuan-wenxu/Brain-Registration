import argparse
import random
import csv
import json
import gc
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from skimage import io, transform
from matplotlib import colormaps
from utils.registration import (
    rigid_register,
    affine_register,
    bspline_register,
    compose_transforms,
)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Step3: rigid, affine, and nonlinear registration of the Step2 selected slice')
    parser.add_argument('--step2-record-json', required=True, help='Step2 record JSON; image paths are read automatically')
    parser.add_argument('--output-path', default=None, help='output directory; default: <step2-json>.parent.parent/03.registration')
    parser.add_argument('--metric-mode', type=str, default='weighted', choices=['raw', 'weighted'], help='step3 stage1 metric mode')
    parser.add_argument('--weight-floor', type=float, default=0.25, help='weight floor for weighted stage1 metric to preserve internal structures')
    parser.add_argument('--mesh-size', type=int, default=8, help='B-spline mesh size per dimension (smaller = stiffer transform)')
    parser.add_argument('--sampling-percentage', type=float, default=0.25, help='metric sampling percentage')
    parser.add_argument('--random-seed', type=int, default=42, help='random seed (for reproducible registration)')
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


def _set_global_determinism(seed=42):
    np.random.seed(int(seed))
    random.seed(int(seed))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


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
    cmap = colormaps[cmap_name]
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
    if step2_record_json is None or not Path(step2_record_json).exists():
        return None, None, None

    with open(step2_record_json, 'r') as f:
        s2 = json.load(f)

    dense_weight_tif = s2.get('dense_weight_tif')
    if dense_weight_tif is None or not Path(dense_weight_tif).exists():
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


def _load_step2_inputs(step2_record_json, output_path=None):
    record_path = Path(step2_record_json)
    if not record_path.exists():
        raise FileNotFoundError(f'Step2 record JSON does not exist: {record_path}')
    with open(record_path, 'r') as f:
        step2 = json.load(f)

    def _resolve(path_value, label):
        if not path_value:
            raise ValueError(f'Step2 record JSON is missing {label}')
        path = Path(path_value)
        if not path.is_absolute() and not path.exists():
            path = record_path.parent / path
        if not path.exists():
            raise FileNotFoundError(f'{label} does not exist: {path}')
        return path

    fixed_path = _resolve(step2.get('fixed_path'), 'fixed_path')
    moving_path = _resolve(step2.get('selected_slice_path'), 'selected_slice_path')

    step1_record_path = _resolve(step2.get('step1_record_json'), 'step1_record_json')
    with open(step1_record_path, 'r') as f:
        step1 = json.load(f)
    input_path = step1.get('input_path')
    sample_name = Path(input_path).stem if input_path else fixed_path.stem

    output_dir = (
        Path(output_path)
        if output_path is not None
        else record_path.parent.parent / '03.registration'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        'fixed_path': fixed_path,
        'moving_path': moving_path,
        'output_path': output_dir / f'{sample_name}_registration.tif',
        'transform_path': output_dir / f'{sample_name}_registration.h5',
    }


def register(
    step2_record_json,
    output_path=None,
    metric_mode='weighted',
    weight_floor=0.25,
    mesh_size=8,
    sampling_percentage=0.25,
    random_seed=42,
):
    paths = _load_step2_inputs(step2_record_json, output_path=output_path)
    fixed_path = str(paths['fixed_path'])
    moving_path = str(paths['moving_path'])
    output_path = paths['output_path']
    transform_path = str(paths['transform_path'])
    _set_global_determinism(seed=random_seed)

    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    fixed_arr = sitk.GetArrayFromImage(fixed)
    output_path = str(output_path)

    metric_history = []
    dense_weight_arr = None
    dense_weight_tif_used = None
    dense_weight_stats = None
    metric_weight_tif = None
    fixed_weighted_arr = fixed

    if metric_mode == 'weighted':
        dense_weight_arr, dense_weight_tif_used, dense_weight_stats = _load_dense_weight_from_step2(
            step2_record_json,
            target_shape=fixed_arr.shape,
        )
    if dense_weight_arr is not None:
        soft_weight = np.clip(dense_weight_arr, 0.0, 1.0)
        soft_weight = np.where(soft_weight > 0.1, np.maximum(soft_weight, weight_floor), 0.0)
        fixed_weighted_arr = _to_sitk_with_like(fixed_arr * soft_weight, fixed)
        metric_weight_tif = output_path.replace(
            '_registration.tif', '_weight_from_step2_dense_weight.tif'
        )
        io.imsave(metric_weight_tif, (np.clip(soft_weight, 0, 1) * 65535).astype(np.uint16))
        print(f'Using weighted metric in step3 stage1, coverage={dense_weight_stats["coverage"]*100:.1f}%')

    rigid = rigid_register(
        fixed, moving,
        sampling_seed=int(random_seed) + 101,
        learning_rate=0.05,
        number_of_iterations=280,
        shrink_factors=(8, 4, 2, 1),
        smoothing_sigmas=(4, 2, 1, 0),
        metric_history=metric_history,
        stage_name='rigid',
    )
    affine = affine_register(
        fixed, rigid['image'],
        center=rigid['transform'].GetCenter(),
        sampling_seed=int(random_seed) + 131,
        learning_rate=0.05,
        number_of_iterations=700,
        shrink_factors=(8, 4, 2, 1),
        smoothing_sigmas=(4, 2, 1, 0),
        metric_history=metric_history,
        stage_name='affine',
    )
    affine_result = affine['image']
    rigid_output_tif = output_path.replace('_registration.tif', '_rigid.tif')
    affine_output_tif = output_path.replace('_registration.tif', '_affine.tif')
    io.imsave(rigid_output_tif, _to_u16(sitk.GetArrayFromImage(rigid['image'])))
    io.imsave(affine_output_tif, _to_u16(sitk.GetArrayFromImage(affine_result)))

    stage1 = bspline_register(
        fixed_weighted_arr,
        affine_result,
        mesh_size=(mesh_size - 2),
        stage_name='bspline_coarse',
        metric_history=metric_history,
        sampling_percentage=sampling_percentage,
        learning_rate=0.005,
        number_of_iterations=200,
        shrink_factors=[4, 2, 1],
        smoothing_sigmas=[2, 1, 0],
    )
    
    stage2 = bspline_register(
        fixed,
        affine_result,
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

    bspline_tx = stage2['transform']

    complete_transform = compose_transforms(
        rigid['transform'],
        affine['transform'],
        bspline_tx,
    )
    result = sitk.Resample(
        moving, fixed, complete_transform, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )

    result_arr = sitk.GetArrayFromImage(result)

    sitk.WriteTransform(complete_transform, transform_path)
    io.imsave(output_path, _to_u16(result_arr))

    post_overlay_rgb_tif = output_path.replace('_registration.tif', '_overlay_on_step1_rgb.tif')
    _save_overlay_rgb_tif(fixed_arr, result_arr, post_overlay_rgb_tif, alpha=0.45, cmap_name='turbo')

    metrics_csv = output_path.replace('_registration.tif', '_metric_history.csv')
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'metric', 'stage'])
        writer.writerows(metric_history)

    record_json = output_path.replace('_registration.tif', '_step3_record.json')
    payload = {
        'fixed_path': str(fixed_path),
        'moving_path': str(moving_path),
        'output_path': str(output_path),
        'transform_path': str(transform_path),
        'deformation_chain_forward': [
            {
                'step': 'step3_rigid_affine_bspline',
                'transform_path': str(transform_path),
                'type': 'CompositeTransform',
            },
        ],
        'step2_record_json': str(step2_record_json),
        'random_seed': int(random_seed),
        'dense_weight_tif_used': str(dense_weight_tif_used),
        'dense_weight_stats': dense_weight_stats,
        'metric_weight_tif': str(metric_weight_tif),
        'mesh_size': int(mesh_size),
        'sampling_percentage': float(sampling_percentage),
        'metric_mode': str(metric_mode),
        'weight_floor': float(weight_floor),
        'rigid_output_tif': str(rigid_output_tif),
        'affine_output_tif': str(affine_output_tif),
        'adjusted_allen_gray_tif': str(output_path),
        'overlay_rgb_tif': str(post_overlay_rgb_tif),
        'metric_history_csv': str(metrics_csv),
    }
    with open(record_json, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'Rigid registered image saved to {rigid_output_tif}')
    print(f'Affine registered image saved to {affine_output_tif}')
    print(f'Rigid + affine + nonlinear registered image saved to {output_path}')
    print(f'Complete composite transform saved to {transform_path}')
    print(f'Adjusted Allen gray tif saved to {output_path}')
    print(f'Overlay RGB tif saved to {post_overlay_rgb_tif}')
    print(f'Step3 record saved to {record_json}')

    _cleanup_images(
        fixed,
        moving,
        rigid['image'],
        affine_result,
        complete_transform,
        result,
        fixed_arr,
        result_arr,
        dense_weight_arr,
        soft_weight if 'soft_weight' in locals() else None,
        fixed_weighted_arr if 'fixed_weighted_arr' in locals() else None,
    )
    return result, record_json


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()

    try:
        result, record_json = register(
            step2_record_json=args.step2_record_json,
            output_path=args.output_path,
            metric_mode=args.metric_mode,
            weight_floor=args.weight_floor,
            mesh_size=args.mesh_size,
            sampling_percentage=args.sampling_percentage,
            random_seed=args.random_seed,
        )
    finally:
        _cleanup_images()
