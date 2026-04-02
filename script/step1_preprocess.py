import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

import os
import argparse
import json
import gc
import numpy as np
from skimage import io, transform, color, measure
from skimage.filters import threshold_otsu, gaussian
from skimage import morphology
from scipy.ndimage import binary_fill_holes


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


def _save_masked_brain_on_canvas(
    gray_resampled,
    tissue_mask,
    output_path,
    canvas_width=1100,
    canvas_height=800,
    mask_canvas_output_path=None,
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
    scale = min(1.0, float(canvas_height) / max(ch, 1), float(canvas_width) / max(cw, 1))
    if scale < 1.0:
        crop = transform.rescale(crop, scale, preserve_range=True, anti_aliasing=True).astype(np.float32)
        mask_crop = transform.rescale(
            mask_crop,
            scale,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)
        ch, cw = crop.shape
    mask_crop = (mask_crop > 0.5).astype(np.uint8)

    top = max(0, (canvas_height - ch) // 2)
    left = max(0, (canvas_width - cw) // 2)
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
    canvas_width=1100,
    canvas_height=800,
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
    scale = min(1.0, float(fullres_canvas_h) / max(ch, 1), float(fullres_canvas_w) / max(cw, 1))
    if scale < 1.0:
        crop = transform.rescale(crop, scale, preserve_range=True, anti_aliasing=True).astype(np.float32)
        ch, cw = crop.shape
    
    top = max(0, (fullres_canvas_h - ch) // 2)
    left = max(0, (fullres_canvas_w - cw) // 2)
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
):
    """
    Step1: gray image + resample
    1) gray_output
    2) resampled_output
    """
    print(f'Loading {input_path} ...')
    img_arr = io.imread(input_path)
    print(f'Shape: {img_arr.shape}, dtype: {img_arr.dtype}')

    img_arr, replaced_pixels = _replace_background_values(img_arr)
    if replaced_pixels > 0:
        print(f'Background cleanup: replaced {replaced_pixels} pixels of (255,255,255)/(204,204,204) to black')

    if img_arr.ndim == 3:
        gray = color.rgb2gray(img_arr).astype(np.float32)
    else:
        gray = img_arr.astype(np.float32)
        maxv = float(gray.max())
        if maxv > 1.0:
            gray = gray / maxv

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

    canvas_output_path = str(mask_output_path).replace('_mask.tif', '_masked_on_1100x800_black.tif')
    mask_canvas_output_path = str(mask_output_path).replace('_mask.tif', '_mask_on_1100x800_black.tif')
    _, _ = _save_masked_brain_on_canvas(
        gray_resampled,
        tissue_mask,
        canvas_output_path,
        canvas_width=1100,
        canvas_height=800,
        mask_canvas_output_path=mask_canvas_output_path,
    )
    
    # Generate high-resolution canvas at original image resolution
    fullres_canvas_output_path = str(mask_output_path).replace('_mask.tif', '_masked_on_fullres_black.tif')
    _, fullres_canvas_shape = _save_fullres_canvas(
        gray,
        tissue_mask,
        gray_resampled.shape,
        fullres_canvas_output_path,
        canvas_width=1100,
        canvas_height=800,
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
            'resample_scale': float(scale),
            'canvas_width': 1100,
            'canvas_height': 800,
            'fullres_canvas_shape': [int(fullres_canvas_shape[0]), int(fullres_canvas_shape[1])],
        },
        'stats': {
            'input_shape': [int(v) for v in img_arr.shape],
            'background_replaced_pixels': int(replaced_pixels),
            'gray_shape': [int(v) for v in gray.shape],
            'resampled_shape': [int(v) for v in gray_resampled.shape],
            'fullres_canvas_shape': [int(fullres_canvas_shape[0]), int(fullres_canvas_shape[1])],
            'mask_coverage': float(tissue_mask.mean()),
            'mask_bbox_r0c0r1c1': mask_bbox,
        },
    }
    with open(record_json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'Tissue mask saved to {mask_output_path}')
    print(f'Mask coverage: {tissue_mask.mean() * 100:.2f}%')
    print(f'Masked brain on 1100x800 black canvas saved to {canvas_output_path}')
    print(f'Masked brain on fullres black canvas saved to {fullres_canvas_output_path}')
    print(f'Mask on 1100x800 black canvas saved to {mask_canvas_output_path}')
    print(f'Step1 record json saved to {record_json_path}')

    _cleanup_images(img_arr, gray, gray_resampled, tissue_mask)

    return record_json_path


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Step1: gray and resample')
    parser.add_argument('-i', '--input-path', required=True, help='input image path')
    parser.add_argument('-g', '--gray-output-path', required=True, help='gray image output path')
    parser.add_argument('-r', '--resampled-output-path', required=True, help='resampled gray image output path')
    parser.add_argument('-m', '--mask-output-path', required=True, help='tissue mask output path ')
    parser.add_argument('--input-res', type=float, default=0.294, help='input resolution (um/px)')
    parser.add_argument('--target-res', type=float, default=10.0, help='target resolution (um/px)')
    return parser


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    out_dir = os.path.dirname(args.gray_output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_dir2 = os.path.dirname(args.resampled_output_path)
    if out_dir2:
        os.makedirs(out_dir2, exist_ok=True)

    try:
        record_json_path = preprocess_image(
            input_path=args.input_path,
            gray_output_path=args.gray_output_path,
            resampled_output_path=args.resampled_output_path,
            mask_output_path=args.mask_output_path,
            input_res=args.input_res,
            target_res=args.target_res,
        )
    finally:
        _cleanup_images()
