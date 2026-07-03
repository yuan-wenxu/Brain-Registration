# Brain-Registration Technical Documentation

This document provides a detailed technical description of the 5-step registration pipeline in this project.

- Pipeline entry: `run_registration.sh` (installed as the `br` command via `install-cli.sh`)
- Data preparation:
  - `script/convert_nrrd_to_npy.py` (NRRD to NPY atlas conversion)
- Step scripts:
  - `script/step1_preprocess.py`
  - `script/step2_select_slice.py`
  - `script/step3_registration.py`
  - `script/step4_apply_label.py`
  - `script/step5_roi_mask.py`
  - `script/utils/registration.py` (shared rigid, affine, and B-spline functions)

---

## 0. Pipeline Overview

The pipeline performs Allen atlas registration for 2D brain slice images through five stages:

1. **Step1 Preprocess**: grayscale conversion, resolution normalization, tissue mask extraction, fixed-size canvas generation, and high-resolution fullres canvas generation.
2. **Step2 Slice Selection**: search for and export the best matching atlas slice.
3. **Step3 Registration**: rigid + affine registration followed by B-spline refinement.
4. **Step4 Apply Label**: apply the Step3 transform chain to atlas annotation and export region statistics.
5. **Step5 ROI Mask**: extract individual ROI masks from warped annotation and generate high-resolution grayscale masks.

### Data flow

- Input image: `*.tif`
- Atlas inputs (convert from `.nrrd` to `.npy` using `script/convert_nrrd_to_npy.py`):
  - Nissl: `ara_nissl_10.npy` (3D stack)
  - Annotation: `annotation_10.npy` (3D stack)
- Output root:
  - `<input_name>_registration_result/`

---

## 0.5 Data Preparation: NRRD to NPY Conversion (`convert_nrrd_to_npy.py`)

### Goal

Convert Allen Brain Atlas `.nrrd` files to NumPy `.npy` format for efficient downstream loading. NPY supports memory-mapped access (`mmap_mode='r'`), which is used in Step2 for fast random slice access from the 3D atlas stack.

### Inputs

- `--input`: a single `.nrrd` file or a directory containing `.nrrd` files
- `--output` (optional): output `.npy` file or directory; defaults to the same name/location as input
- `--save-header` (optional): also save the NRRD header (spacing, origin, direction) as a `.meta.json` sidecar file

### Typical usage

```bash
# single file
pixi run python script/convert_nrrd_to_npy.py \
  --input /path/to/ara_nissl_10.nrrd \
  --output /path/to/ara_nissl_10.npy

# batch convert all .nrrd in a directory
pixi run python script/convert_nrrd_to_npy.py \
  --input /path/to/AllenCCF/

# preserve spatial metadata
pixi run python script/convert_nrrd_to_npy.py \
  --input /path/to/ara_nissl_10.nrrd \
  --output /path/to/ara_nissl_10.npy \
  --save-header
```

### Implementation notes

- Uses `pynrrd` (preferred) or `nibabel` as fallback for reading NRRD files.
- The conversion preserves raw voxel data as-is (no resampling or reorientation).
- Spatial metadata (spacing, origin, direction) is **not** embedded in the `.npy` file. Use `--save-header` to export it as JSON if needed for downstream tools.
- The pipeline does not require spatial metadata from the atlas NRRD: Step2 selects 2D slices by index and Step3 performs its own spatial alignment via registration.

---

## 1. Step1: Preprocess (`step1_preprocess.py`)

### Goal

Convert the raw image into a robust fixed reference image for registration by standardizing intensity, geometry, and background.

### Inputs

- `input_path`: raw image path (`tif`)
- `input_res` (um/px): source resolution (default `0.294`)
- `target_res` (um/px): target resolution (default `10.0`)

### Core processing

1. **Background cleanup for uint8 images**
   - Replace pure white/gray background values (`255`, `204`) with black.
2. **Grayscale conversion**
   - RGB -> grayscale (`skimage.color.rgb2gray`) or normalized single-channel.
3. **Resampling**
   - Scale factor: `input_res / target_res`.
4. **Tissue mask extraction**
   - Border-median background estimation + contrast map (`abs(image - border_median)`).
   - Otsu thresholding + morphology cleanup + largest connected component selection.
   - Adaptive fallback for over-coverage and under-coverage cases.
5. **Canvas normalization**
   - Crop by tissue bbox and center it on a fixed black canvas (`1140x800`) without changing pixel resolution. Oversized content raises an error instead of being scaled or cropped.
6. **High-resolution fullres canvas generation**
   - Generate a canvas at original image resolution for Step5 ROI mask generation.
   - Dimensions are recorded in `step1_record.json` under `params.fullres_canvas_shape` to enable memory-safe access in Step5.

### Outputs

Under `01.preprocess/`:

- `<name>_gray.tif` - Grayscale image converted from input (uint16)
- `<name>_resampled.tif` - Resampled to target resolution (uint16)
- `<name>_mask.tif` - Binary tissue mask (uint8: 0/255)
- `<name>_masked_on_1140x800_black.tif` - Fixed canvas at standard resolution (1140×800 px, uint16)
- `<name>_mask_on_1140x800_black.tif` - Tissue mask on fixed canvas (1140×800 px, uint8: 0/255)
- `<name>_masked_on_fullres_black.tif` - High-resolution canvas at original image resolution (uint16)
- `<name>_step1_record.json` - Metadata including fullres_canvas_shape for downstream steps

The key output for downstream registration is the **fixed canvas** (`*_masked_on_1140x800_black.tif`). The **fullres canvas** (`*_masked_on_fullres_black.tif`) is generated for high-resolution ROI mask generation in Step5, with dimensions recorded in the JSON record to avoid OOM during Step5 execution.

### Practical notes

- Step1 strongly impacts downstream stability. If mask coverage is poor, Step2 search quality usually drops.
- Canvas standardization is critical to keep all later transforms in a consistent image space.

---

## 2. Step2: Slice Selection (`step2_select_slice.py`)

### Goal

Find and export the best matching atlas slice. Candidate scoring uses temporary low-resolution registrations, but Step2 does not produce or persist a final registration transform.

### Inputs

- `step1_record_json`: Step1 record JSON; the fixed canvas and mask paths are parsed from its `outputs` section
- `moving_path`: atlas Nissl stack (`.npy`, 3D)
- Search hyperparameters:
  - `atlas_slice` (optional center)
  - `slice_search_radius`
  - `slice_search_step`
  - `search_resize_max`
  - `neighbor_smooth_sigma`
- Runtime:
  - `workers`
  - `random_seed`

### Search strategy

#### Step2 evaluates candidate slices in 3 stages:

1. **Coarse search** around center with coarse step.
2. **Refine search** around top-ranked coarse slices with finer step.
3. **Ultra-refine** around best slices densely.

### Per-slice scoring method

#### For each candidate slice:

1. Run **rigid** registration (MI metric).
2. Run **affine** registration (MI metric).
3. Run temporary **B-spline** registration only for scoring robustness.
4. Compute scoring terms:
   - `bspline_mi`
   - edge similarity (`edge_ncc`)
   - dense-focus weighted NCC (`dense_ncc`)

The edge focuses on gradient changes in the image.  
The dense of background is 0.  
Only images within the mask have dense.  

Then slice-level best scores are smoothed along slice index using Gaussian kernel (`neighbor_smooth_sigma`) and ranked.

#### Scoring function:
1. score = bspline_mi - 0.12 * edge_ncc - 0.22 * dense_ncc
2. smoothed

### Selected-slice export

After selection, save the unregistered atlas slice on the Step1-sized canvas. Final rigid and affine registration is performed in Step3.

### Outputs

Under `02.select_slice/`:

- `<name>_selected_slice.tif`
- `<name>_dense_weight.tif`
- `<name>_slice_search_metrics.csv`
- `<name>_slice_score_curve.png`
- `<name>_step2_record.json`

### Practical notes

- `workers` runs candidate scoring in separate processes. Each worker uses one SimpleITK thread to avoid CPU oversubscription; `2~8` workers is a practical starting range.
- Each worker always uses one SimpleITK thread to avoid CPU oversubscription.
- If alignment is unstable, tune search radius/step and smoothing sigma first.
- `<name>_dense_weight.tif` is not used for dense scoring. Its background is 0.
---

## 3. Step3: Rigid, Affine, and Nonlinear Registration (`step3_registration.py`)

### Goal

Register the Step2-selected slice using rigid, affine, and B-spline transforms.

### Inputs

- `step2_record_json`: Step2 record JSON; Step1 fixed canvas, mask, selected slice, and dense-weight paths are parsed automatically
- `output_path` (optional): output directory; defaults to the sibling `03.registration/` directory

### Core processing

1. Load fixed image and the selected atlas slice.
2. Run full-resolution rigid and affine registration and save the merged transform.
3. Optionally read Step2 `dense_weight.tif` and build a continuous metric-weight map.
4. Initialize B-spline transform (mesh `[8, 8]`, order `3`).
4. Weighted image registration and original image registration.
5. Optimize MI metric with multi-resolution pyramid.
6. Compose rigid, affine, and B-spline transforms and resample the original moving image once.

### Outputs

Under `03.registration/`:

- `<name>_registration.tif`
- `<name>_registration.h5`
- `<name>_overlay_on_step1_rgb.tif`
- `<name>_metric_history.csv`
- `<name>_metric_weight_from_dense_weight.tif` (if dense-weight is used)
- `<name>_step3_record.json`

### Practical notes

- Step3 improves local alignment details but can overfit noisy regions if input quality is poor.
- Dense-weight guidance helps prioritize biologically relevant/high-confidence regions.

---

## 4. Step4: Apply Label (`step4_apply_label.py`)

### Goal

Apply Step2 and Step3 deformation chain to annotation labels and export interpretable statistics.

### Inputs

- `label_path`: annotation atlas (`.npy` or image)
- `reference_path`: Step1 fixed image
- `step2_record_json`: Step2 selected-slice metadata
- `step3_record_json`: Step3 transform metadata
- `atlas_slice` (optional; inferred from Step2 when omitted)
- `nissl_path` (optional, for visualization overlay)

### Core processing

1. Read the selected slice from Step2 metadata and build the deformation chain from Step3 metadata.
2. Load annotation label for selected slice.
3. Compose transform chain and apply **single composite resampling** with nearest-neighbor interpolation.
4. Export colorized label image and overlay.
5. Compute region-level area/centroid statistics.

### Outputs

Under `04.apply_label/`:

- `<name>_label.tif` - Colorized label image (RGB, uint8) showing warped annotation regions
- `<name>_overlay.tif` - Label overlay on fixed reference image for visual verification
- `<name>_annotation_nissl_merge.tif` - Label overlay on Nissl reference (if `nissl_path` provided)
- `<name>_region_distribution.csv` - Region-level statistics (pixel_count, area_ratio, centroid coordinates)
- `<name>_step4_record.json` - Complete deformation chain metadata and registration parameters

### Practical notes

- Nearest-neighbor interpolation is required for label integrity.
- `atlas_slice` must match Step2 selected slice (script validates this).

---

## 5. Step5: ROI Mask Generation (`step5_roi_mask.py`)

### Goal

Extract individual region of interest (ROI) masks from Step4's warped annotation. Generate both downsampled masks (aligned with Step1's 1140×800 canvas) and high-resolution grayscale masks (aligned with original image resolution).

### Inputs

- `step4_record_json`: Step4 output record containing deformation chain and warped label path
- `roi_txt_path`: Text file with one ROI acronym/name per line (e.g., "CA1", "DG", "PFC")
- `structure_tree_csv`: Allen structure tree CSV file mapping ROI names to numeric IDs and hierarchies
- Optional: Step1 record (automatically discovered) containing `fullres_canvas_shape` metadata

### Core processing

1. **Load warped annotation** from Step4 output.
2. **Resolve ROI names** to numeric IDs using structure tree:
   - Support both exact acronym match and name prefix match.
   - Collect all descendant IDs for hierarchical regions.
3. **Memory-efficient fullres sizing**:
   - Read `fullres_canvas_shape` from Step1 JSON metadata (avoids loading large fullres image).
   - Fallback to metadata-only read via SimpleITK if Step1 record unavailable.
4. **Early ROI skip optimization**:
   - Pre-check if any label IDs exist in warped annotation before mask computation.
   - Skip empty/absent ROIs to avoid wasteful processing.
5. **Mask generation and scaling**:
   - Create binary mask at downsampled resolution (1140×800).
   - Scale mask to fullres using Step1 metadata.
   - Convert to single-channel grayscale PNG (uint8: 0/255) for high-resolution output.
6. **Generate report** with pixel counts, area ratios, and matched descendant IDs.

### Outputs

Under `05.roi_mask/`:

- `<roi_name>_mask.tif` - Downsampled mask at standard canvas resolution (1140×800 px, uint8: 0/255)
- `<roi_name>_mask_fullres_gray.png` - High-resolution mask at original image resolution (single-channel grayscale, uint8: 0/255)
- `roi_mask_report.csv` - Comprehensive ROI summary:
  - `roi`: ROI acronym/name
  - `match_mode`: "exact" or "prefix"
  - `matched_root_ids`: Comma-separated IDs of matched root regions
  - `descendant_inclusive_id_count`: Total descendant count including matched roots
  - `pixel_count`: Number of pixels in mask at downsampled resolution
  - `area_ratio`: Fraction of total canvas area occupied by ROI
  - `fullres_gray_mask_path`: Path to high-resolution grayscale mask
- `step5_record.json` - Metadata including ROI list, applied deformation chain, and fullres mask availability status

### Practical notes

- **Memory efficiency**: Step5 reads fullres canvas dimensions from Step1 JSON only; the actual image file is not loaded.
- **Format choice**: High-resolution masks are saved as **single-channel grayscale PNG** (not RGBA) to reduce file size by 75%.
- **Hierarchical ROI support**: Structure tree enables automatic inclusion of sub-regions (e.g., requesting "Hippocampus" includes "CA1", "CA3", "DG").
- **Early filtering**: ROIs with no matching label IDs in the warped annotation are skipped automatically to save computation.
- **Backward compatibility**: Older Step1 records lacking `fullres_canvas_shape` can still be processed via SimpleITK metadata reads.

---

## 6. Main Entry Integration (`run_registration.sh` / `br`)

The shell script `run_registration.sh` orchestrates Step1 → Step5 end-to-end. It is installed as the user-level `br` command by `install-cli.sh`.

### Configuration

All parameters are read from a configuration file (`config/registration.conf`):

```bash
cp config/registration.conf.example config/registration.conf
# edit config/registration.conf to set atlas paths, resolutions, etc.
```

### Typical command

```bash
br rgb --input-path /path/to/input.tif
br nissl --input-path /path/to/input.tif --rotation 90
br rgb --input-path /path/to/input.tif --config /path/to/other.conf
```

- First argument: `rgb` or `nissl` (grayscale conversion mode)
- `--input-path`: brain section image (required)
- `--config`: override default configuration file (optional)
- `--rotation`: counterclockwise degrees, `0`/`90`/`180`/`270` (default: `0`)

### Pipeline behavior

- Creates output directories `01.preprocess/`, `02.select_slice/`, `03.registration/`, `04.apply_label/`, `05.roi_mask/`
- Executes each step in sequence via `pixi run python`
- Reads/writes step record JSON files for cross-step parameter/transform propagation
- Skips Step5 when `ROI_TXT_PATH` is empty in the configuration

---

## 7. Reproducibility and Debugging

### Recommended for reproducibility

- `--random-seed 42` (or fixed custom value)
- Step3 fixes the SimpleITK thread count at `1`
- Keep Step2/Step3 record JSON files

### Common failure points

1. Incorrect atlas path (`--atlas-nissl`, `--atlas-annotation`)
2. Missing or invalid Step1 mask/canvas files
3. Shape mismatch from external manual edits of intermediate outputs
4. Resource pressure when `workers` is too high

### Minimal debug checklist

1. Validate Step1 outputs exist and look correct.
2. Inspect `02.select_slice/*_step2_record.json` for selected slice and score terms.
3. Inspect `03.registration/*_metric_history.csv` for convergence behavior.
4. Confirm Step4 chain files exist in `*_step4_record.json`.

---

## 8. File Contract Summary

- Data prep (`convert_nrrd_to_npy.py`) -> Step2:
  - `ara_nissl_10.npy`, `annotation_10.npy` (prerequisite atlas files)
- Step1 -> Step2:
  - `masked_canvas_path`, `mask_canvas_path`
- Step2 -> Step3:
  - `*_selected_slice.tif`, `*_step2_record.json`, `dense_weight.tif`
- Step3 -> Step4:
  - `*_registration.h5` (complete rigid + affine + B-spline transform), `*_step3_record.json`
- Step4 -> Step5:
  - `*_label.tif` (warped annotation from Step4)
  - `*_step4_record.json` (contains deformation chain)
  - Step1 record JSON (discovered automatically for `fullres_canvas_shape` metadata)
- Step5 final:
  - ROI masks at both downsampled (1140×800) and fullres resolutions
  - ROI mask report and statistics

This contract should be preserved when modifying scripts to maintain pipeline compatibility.

---

## 9. Memory Management

All five steps implement explicit resource cleanup to prevent OOM (out-of-memory) failures with large images:

- `_cleanup_images()` helper function defined in each step script
- `gc.collect()` invoked after large array processing
- `.close()` called on image objects (SimpleITK, PIL, etc.) where applicable
- try/finally blocks in CLI entry points guarantee cleanup execution

### Memory-critical operations

- **Step1**: High-resolution canvas generation (original image res ~28k×39k)
- **Step2**: Atlas stack search (3D array with many candidate slices)
- **Step3**: BSpline transform computation and resampling
- **Step4**: Label rasterization and region statistics
- **Step5**: Deferred fullres sizing via metadata-only reads (avoids loading large images)
