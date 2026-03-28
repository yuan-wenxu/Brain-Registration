import os
import argparse
from pathlib import Path
import json
from step1_preprocess import preprocess_image
from step2_affine import affine_register
from step3_nonlinear import nonlinear_register
from step4_apply_label import apply_transform_to_label


def build_arg_parser():
    parser = argparse.ArgumentParser(description='4 step registration pipeline: preprocess -> affine -> nonlinear -> apply label transform')

    parser.add_argument('--data-path', help='image path')

    parser.add_argument('--atlas-nissl', help='atlas nissl path (relative to data-dir)')
    parser.add_argument('--atlas-annotation', help='atlas annotation path (relative to data-dir)')

    parser.add_argument('--input-res', type=float, default=0.294, help='step1 input resolution (um/px)')
    parser.add_argument('--target-res', type=float, default=10.0, help='step1 target resolution (um/px)')

    parser.add_argument('--atlas-slice', type=int, default=None, help='step2 specify slice; if not filled, search automatically')
    parser.add_argument('--slice-search-radius', type=int, default=200, help='step2 search radius')
    parser.add_argument('--slice-search-step', type=int, default=20, help='step2 search step')
    parser.add_argument('--search-resize-max', type=int, default=768, help='step2 maximum边 during search')
    parser.add_argument('--random-seed', type=int, default=2026, help='random seed (step2/step3)')
    parser.add_argument('--sitk-threads', type=int, default=1, help='SimpleITK global thread count (recommended 1 for reproducibility)')
    parser.add_argument('--search-workers', type=int, default=1, help='step2 parallel worker count for slice search (recommended 2-8)')
    parser.add_argument('--neighbor-smooth-sigma', type=float, default=3.0, help='step2 neighbor smooth sigma for better slice selection (in pixel, recommend 1-3)')
    return parser


def main():
    args = build_arg_parser().parse_args()
    serial_input_path = args.data_path
    img_name = os.path.basename(serial_input_path).split('.')[0]
    output_root = Path(serial_input_path).parent
    output_root = output_root / f'{img_name}_registration_result'
    output_root.mkdir(exist_ok=True)
    atlas_nissl_path = args.atlas_nissl
    atlas_annotation_path = args.atlas_annotation

    # Step 1
    step1_dir = output_root / '01.preprocess'
    step1_dir.mkdir(exist_ok=True)
    gray_path = step1_dir / f'{img_name}_gray.tif'
    resampled_path = step1_dir / f'{img_name}_resampled.tif'
    mask_path = step1_dir / f'{img_name}_mask.tif'
    step1_kwargs = {
        'input_path': serial_input_path,
        'gray_output_path': gray_path,
        'resampled_output_path': resampled_path,
        'mask_output_path': mask_path,
        'input_res': args.input_res,
        'target_res': args.target_res,
    }
    if not gray_path.exists() or not resampled_path.exists() or not mask_path.exists():
        step1_record_path = preprocess_image(**step1_kwargs)
    else:
        print(f'Step1 outputs already exist, skipping preprocessing')
        step1_record_path = step1_dir / f'{img_name}_step1_record.json'
    with open(step1_record_path, 'r') as f:
        step1_info = json.load(f)

    # Step 2
    step2_dir = output_root / '02.affine'
    step2_dir.mkdir(exist_ok=True)
    affine_path = step2_dir / f'{img_name}_affine.tif'
    if not affine_path.exists():
        _, _, affine_info_path = affine_register(
            fixed_path=step1_info['outputs']['masked_canvas_path'],
            moving_path=atlas_nissl_path,
            output_path=affine_path,
            mask_path=step1_info['outputs']['mask_canvas_path'],
            atlas_slice=args.atlas_slice,
            slice_search_radius=args.slice_search_radius,
            slice_search_step=args.slice_search_step,
            search_resize_max=args.search_resize_max,
            random_seed=args.random_seed,
            sitk_threads=args.sitk_threads,
            search_workers=args.search_workers,
            neighbor_smooth_sigma=args.neighbor_smooth_sigma,
        )
    else:
        print(f'Step2 affine output already exists, skipping affine registration')
        affine_info_path = step2_dir / f'{img_name}_step2_record.json'
    with open(affine_info_path, 'r') as f:
        affine_info = json.load(f)

    best_slice = affine_info['selected_slice']
    step2_record_json = affine_info_path
    print(f'Best atlas slice selected in step2: {best_slice}')

    # Step 3
    step3_dir = output_root / '03.nonlinear'
    step3_dir.mkdir(exist_ok=True)
    nonlinear_path = step3_dir / f'{img_name}_nonlinear.tif'
    transform_path = step3_dir / f'{img_name}_nonlinear.h5'
    if not nonlinear_path.exists() or not transform_path.exists():
        _, step3_record_json = nonlinear_register(
            fixed_path=step1_info['outputs']['masked_canvas_path'],
            moving_path=affine_path,
            output_path=nonlinear_path,
            transform_path=transform_path,
            step2_record_json=step2_record_json,
            random_seed=args.random_seed,
            sitk_threads=args.sitk_threads,
        )
    else:
        print(f'Step3 nonlinear output already exists, skipping nonlinear registration')
        step3_record_json = step3_dir / f'{img_name}_step3_record.json'
    with open(step3_record_json, 'r') as f:
        step3_info = json.load(f)

    # Step 4
    step4_dir = output_root / '04.apply_label'
    step4_dir.mkdir(exist_ok=True)
    warped_label_path = step4_dir / f'{img_name}_label.tif'
    apply_transform_to_label(
        label_path=atlas_annotation_path,
        reference_path=step1_info['outputs']['masked_canvas_path'],
        output_path=warped_label_path,
        step2_record_json=step2_record_json,
        step3_record_json=step3_record_json,
        nissl_path=step3_info['adjusted_allen_gray_tif'],  # optional, for better visualization
    )

    print('Registration pipeline finished!')


if __name__ == '__main__':
    main()
