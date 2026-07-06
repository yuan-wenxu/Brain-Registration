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
  - `script/utils/image_processing.py` (shared grid-removal functions)

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
- Output root: `OUTPUT_PATH` from the configuration. When empty, the five
  stage directories are created beside the input image.

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
- `grayscale_mode`: `rgb` luminance or Nissl optical-density conversion
- `rotation`: counterclockwise `0`, `90`, `180`, or `270`
- `brain_layout`: `auto`, `whole`, `left`, or `right`
- `remove_grid`: optionally generate a corrected downstream canvas

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
7. **Optional grid suppression**
   - Detect row/column stripe profiles and periodic Fourier-axis peaks.
   - Preserve the original fixed canvas and save a separate corrected canvas.
   - Store the selected downstream image in `outputs.downstream_canvas_path`.
8. **Partial-brain geometry metadata**
   - Detect left/right layout and record the persistent straight tissue cut
     edge as `stats.cut_edge_canvas_col` for Step2 and Step3.
   - If no sufficiently persistent straight edge is detected, record `null`;
     partial-brain processing then falls back to the canvas center column.

### Outputs

Under `01.preprocess/`:

- `<name>_gray.tif` - Grayscale image converted from input (uint16)
- `<name>_resampled.tif` - Resampled to target resolution (uint16)
- `<name>_mask.tif` - Binary tissue mask (uint8: 0/255)
- `<name>_masked_on_1140x800_black.tif` - Fixed canvas at standard resolution (1140×800 px, uint16)
- `<name>_masked_degridded_on_1140x800_black.tif` - Optional corrected fixed canvas used by Step2 and Step3
- `<name>_mask_on_1140x800_black.tif` - Tissue mask on fixed canvas (1140×800 px, uint8: 0/255)
- `<name>_masked_on_fullres_black.tif` - High-resolution canvas at original image resolution (uint16)
- `<name>_step1_record.json` - Metadata including fullres_canvas_shape for downstream steps

The key output for downstream registration is `outputs.downstream_canvas_path`.
It points to the corrected canvas when grid removal is enabled and otherwise
points to the original fixed canvas. Older records without this field fall back
to `masked_canvas_path`. The fullres canvas is generated for high-resolution
ROI mask generation in Step5.

### Practical notes

- Step1 strongly impacts downstream stability. If mask coverage is poor, Step2 search quality usually drops.
- Canvas standardization is critical to keep all later transforms in a consistent image space.

---

## 2. Step2: Slice Selection (`step2_select_slice.py`)

### Goal

Find and export the best matching atlas slice. Candidate scoring uses temporary
low-resolution rigid and affine registrations, but Step2 does not produce or
persist a final registration transform. Nonlinear B-spline registration is
reserved for Step3 so it cannot make an anatomically incorrect slice appear to
be a good candidate through excessive deformation.

### Inputs

- `step1_record_json`: Step1 record JSON; the fixed canvas and mask paths are parsed from its `outputs` section
- `moving_path`: atlas Nissl stack (`.npy`, 3D)
- Search hyperparameters:
  - `atlas_slice` (optional expected slice used as the search center)
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

When grid removal is enabled, Step1 corrects robust row/column intensity
profiles to suppress non-stationary stripes, then applies soft Gaussian notches
at automatically detected Fourier-axis peaks. It preserves the original canvas
and saves a separate `*_masked_degridded_on_1140x800_black.tif`. The Step1
record exposes this as `downstream_canvas_path`, so Step2 and Step3 use the same
corrected image. Detection diagnostics are stored in the Step1 record.

#### For each candidate slice:

1. For a whole brain, place the atlas slice centered on a canvas matching the
   fixed image dimensions.
2. **Partial-brain handling**: Step1 detects the specimen's persistent straight
   cut edge from the placed tissue mask. Allen is cropped at that same canvas
   column before registration, rather than being divided at its center. This
   supports sections containing more or less than exactly one hemisphere. The
   full superior/inferior extent is retained. If the recorded edge is `null`,
   the canvas center is used as a backward-compatible fallback.
3. Run **rigid** registration (MI metric) on the (possibly cropped) region.
4. Run **affine** registration (MI metric).
5. Resample the atlas tissue mask with the rigid + affine transform.
6. Compute deterministic scoring terms on the same pixels and mask:
   - modality-independent local self-similarity (`mind_ncc`)
   - normalized mutual information (`normalized_mi`)
   - Gaussian low-frequency NCC (`low_frequency_ncc`)
   - tissue-mask Dice (`tissue_dice`)
   - tissue-boundary similarity (`boundary_similarity`)
   - affine scale, anisotropy, and shear penalty (`affine_deformation_penalty`)

The edge focuses on gradient changes in the image.  
The dense of background is 0.  
Only images within the mask have dense.  

Then slice-level best scores are smoothed with a Gaussian kernel over the actual
atlas-index distances (`neighbor_smooth_sigma`) and ranked. Each candidate uses
independently normalized weights, so sparse sampling and search boundaries do
not duplicate edge scores. Refine stages include the best candidates from both
the smoothed ranking and the raw-score ranking to retain strong isolated matches.

#### Scoring function

The score uses fixed absolute metric weights, so the same candidate receives the
same score regardless of the configured search center, radius, or refine-sample
density. Local self-similarity descriptors receive the largest weight because
they preserve internal anatomy across staining modalities:

```text
score = -(
    0.53 * mind_ncc
  + 0.30 * rigid_normalized_mi
  + 0.08 * rigid_multiscale_ncc
  + 0.06 * boundary_similarity
  - 0.02 * affine_deformation_penalty
  + 0.01 * tissue_dice
)
```

No slice-index location prior is included in the score.

For partial brains, rigid initialization first aligns signed tissue-mask
distance maps. If a candidate has insufficient overlap for that optimization,
Step2 falls back to intensity-based rigid initialization instead of aborting the
whole search. If affine optimization still fails for an isolated candidate,
Step2 retains and scores the rigid result instead of aborting the batch.

Higher similarity produces a more negative score, so lower is better. All
search stages use identical registration sampling seeds. The final ranking uses
the atlas-index-aware smoothed score.

### Selected-slice export

After selection, save the unregistered atlas slice on the Step1-sized canvas. Final rigid and affine registration is performed in Step3.

### Outputs

Under `02.select_slice/`:

- `selected_slice.tif`
- `dense_weight.tif`
- `slice_search_metrics.csv`
- `slice_score_curve.png`
- `step2_record.json`

### Practical notes

- `workers` runs candidate scoring in separate processes. Each worker uses one SimpleITK thread to avoid CPU oversubscription; `2~8` workers is a practical starting range.
- Each worker always uses one SimpleITK thread to avoid CPU oversubscription.
- If alignment is unstable, tune search radius/step and smoothing sigma first.
- `dense_weight.tif` is not used as a standalone score. It guides weighted
  registration and has background value 0.
- **Partial-brain support**: Step1 automatically detects left/right layout and
  records the actual straight cut-edge column. Step2 uses the corresponding
  full-height region for scoring. The exported `selected_slice.tif` is already
  cropped at that specimen-derived edge, so Step3 receives the same geometry.
---

## 3. Step3: Rigid, Affine, and Nonlinear Registration (`step3_registration.py`)

### Goal

Register the Step2-selected slice using rigid, affine, and B-spline transforms.

### Inputs

- `step2_record_json`: Step2 record JSON; Step1 fixed canvas, mask, selected slice, and dense-weight paths are parsed automatically
- `output_path` (optional): output directory; defaults to the sibling `03.registration/` directory

### Core processing

1. Load fixed image and the selected atlas slice. Read `brain_layout` from the Step1 record.
2. **Partial-brain handling**: when `brain_layout` is `left` or `right`, build a
   full-height crop split vertically at Step1's detected specimen cut edge,
   then crop both images with adjusted physical origins. This preserves Allen
   superior and inferior boundaries without assuming an exact hemisphere.
3. Run rigid registration on the (possibly cropped) region.
4. Run affine registration on the (possibly cropped) region.
5. For half-brain, resample the full moving image with the composed rigid+affine transform for bspline input.
6. Optionally read Step2 `dense_weight.tif` and build a continuous metric-weight map.
7. Run B-spline registration on the full canvas using weighted metric (coarse) and unweighted metric (refine).
8. Compose rigid, affine, and B-spline transforms and resample the original moving image once.

### Outputs

Under `03.registration/`:

- `<name>_registration.tif`
- `<name>_registration.h5`
- `<name>_rigid.tif`
- `<name>_affine.tif`
- `<name>_overlay_on_step1_rgb.tif`
- `<name>_metric_history.csv`
- `<name>_weight_from_step2_dense_weight.tif` (if dense-weight is used)
- `<name>_step3_record.json`

### Practical notes

- Step3 improves local alignment details but can overfit noisy regions if input quality is poor.
- Dense-weight guidance helps prioritize biologically relevant/high-confidence regions.
- **Partial-brain support**: rigid and affine stages operate on the full-height
  region retained by the specimen-derived cut edge. B-spline stages use the
  full canvas with weighted metric. Transforms remain in the full-canvas
  physical coordinate system.

---

## 4. Step4: Apply Label (`step4_apply_label.py`)

### Goal

Apply the composite transform produced by Step3 to the selected Allen
annotation slice and export interpretable statistics.

### Inputs

- `label_path`: annotation atlas (`.npy` or image)
- `step3_record_json`: Step3 transform metadata; Step2 record, selected slice,
  fixed reference, transformed Nissl image, and atlas index are resolved from it
- `output_path` (optional): output directory

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
2. **Resolve ROI tokens** to numeric IDs using the structure tree:
   - Support case-insensitive exact acronym matching.
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
  - `match_mode`: `acronym` or `not_found`
  - `matched_root_ids`: Comma-separated IDs of matched root regions
  - `descendant_inclusive_id_count`: Total descendant count including matched roots
  - `pixel_count`: Number of pixels in mask at downsampled resolution
  - `area_ratio`: Fraction of total canvas area occupied by ROI
  - `fullres_gray_mask_path`: Path to high-resolution grayscale mask
- `step5_record.json` - Metadata including ROI list, applied deformation chain, and fullres mask availability status

### Practical notes

- **Memory efficiency**: Step5 reads fullres canvas dimensions from Step1 JSON only; the actual image file is not loaded.
- **Format choice**: High-resolution masks are saved as **single-channel grayscale PNG** (not RGBA) to reduce file size by 75%.
- **Hierarchical ROI support**: after an exact acronym match, all descendants
  in `structure_id_path` are included automatically.
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
br rgb --input /path/to/input.tif
br rgb --input /path/to/input.tif --output /path/to/result
br nissl --input /path/to/input.tif --rotation 90 --input-res 0.294
br nissl --input /path/to/input.tif --rotation 90 --remove-grid
br rgb --input /path/to/input.tif --config /path/to/other.conf
```

- First argument: `rgb` or `nissl` (grayscale conversion mode)
- `--input`: brain section image (required)
- `--output`: per-run output root override; takes precedence over `OUTPUT_PATH`
- `--config`: override default configuration file (optional)
- `--rotation`: counterclockwise degrees, `0`/`90`/`180`/`270` (default: `0`)
- `--input-res`: per-image source resolution override in `um/px`; takes
  precedence over `INPUT_RES` in the configuration
- `--remove-grid`: make Step1 generate a corrected downstream canvas used by
  both Step2 slice selection and Step3 registration

### Pipeline behavior

- Creates `01.preprocess/` through `04.apply_label/`; creates `05.roi_mask/`
  only when `ROI_TXT_PATH` is configured
- Executes each step in sequence via `pixi run python`
- Reads/writes step record JSON files for cross-step parameter/transform propagation
- Skips Step5 when `ROI_TXT_PATH` is empty in the configuration

---

## 7. Reproducibility and Debugging

### Recommended for reproducibility

- Set `RANDOM_SEED="42"` (or another fixed value) in the configuration. Direct
  Step2 and Step3 invocations also accept `--random-seed`.
- Step3 fixes the SimpleITK thread count at `1`
- Keep Step2/Step3 record JSON files

### Common failure points

1. Incorrect `ATLAS_NISSL` or `ATLAS_ANNOTATION` path in the configuration
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
  - `downstream_canvas_path` (falls back to `masked_canvas_path`),
    `masked_canvas_path`, `mask_canvas_path`
- Step2 -> Step3:
  - `selected_slice.tif`, `step2_record.json`, `dense_weight.tif`
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
