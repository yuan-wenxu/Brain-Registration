# Brain-Registration

Register a mouse brain section to the Allen Mouse Brain Atlas, transform the Allen annotation, and optionally export ROI masks.

## Install

Install [Pixi](https://pixi.prefix.dev/) first, then run:

```bash
cd /path/to/Brain-Registration
pixi run init
source ~/.bashrc
br --help
```

`pixi run init` installs the project environment and the user-level `br` command.

## Configuration

Create the local configuration from the tracked template:

```bash
cp config/registration.conf.example config/registration.conf
```

`config/registration.conf` is the default configuration used by `br`. Edit the configuration to set the paths and parameters for your data.

The configuration contains:

- Allen Nissl atlas path
- Allen annotation path
- structure tree path
- input and Allen resolutions
- slice-selection parameters
- registration parameters
- optional ROI list path

Set the ROI path to an empty string to skip Step5:

```bash
ROI_TXT_PATH=""
```

Set `ATLAS_SLICE` when the approximate anatomical level is known. It defines
the search center; leave it empty for fully automatic image-only selection.

`config/ROI.txt.example` is the ROI template.
## Usage

RGB grayscale conversion:

```bash
br rgb \
  --input /path/to/brain.tif \
  --rotation 0
```

Nissl stain grayscale conversion:

```bash
br nissl \
  --input /path/to/brain.tif \
  --rotation 90 \
  --input-res 0.294
```

`--rotation` is counterclockwise and accepts `0`, `90`, `180`, or `270`. Use `--config /path/to/another.conf` to override the default configuration.
Use `--output /path/to/result` to keep complete results separate when multiple
input images share one directory.
Use `--input-res UM_PER_PX` for images whose source resolution differs from the
configuration default. The command-line value applies only to that run and
takes precedence over `INPUT_RES` in the configuration file.
Use `--remove-grid` only when the image contains periodic grid artifacts.

## Inputs

- Brain section image: TIFF or another image format supported by scikit-image.
- Configuration file: local `config/registration.conf`, or another file supplied with `--config`.
- Allen Nissl atlas: 3D `.npy` array converted from the official
  [`ara_nissl_10.nrrd`](https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_10.nrrd).
- Allen CCF 2022 annotation: 3D `.npy` array converted from the official
  [`annotation_10.nrrd`](https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2022/annotation_10.nrrd).

Convert NRRD to NPY with the bundled script:

```bash
# single file
pixi run python script/convert_nrrd_to_npy.py \
  --input /path/to/ara_nissl_10.nrrd \
  --output /path/to/ara_nissl_10.npy
```
- ROI text file: optional, one ROI acronym or name per line.
- Allen 2017 structure tree CSV: required only when an ROI file is configured; [download from OSF](https://osf.io/fv7ed/files/osfstorage). The 2017 structure hierarchy is used with the 2020 atlas volumes.

## Outputs

When `OUTPUT_PATH` is empty, output directories are created beside the input image:

```text
01.preprocess/
02.select_slice/
03.registration/
04.apply_label/
05.roi_mask/        # only when ROI_TXT_PATH is configured
```

Main outputs:

```text
01.preprocess/
  *_gray.tif
  *_resampled.tif
  *_mask.tif
  *_mask_on_1140x800_black.tif
  *_masked_on_1140x800_black.tif
  *_masked_degridded_on_1140x800_black.tif  # only with --remove-grid
  *_masked_on_fullres_black.tif
  *_step1_record.json

02.select_slice/
  selected_slice.tif
  slice_search_metrics.csv
  slice_score_curve.png
  dense_weight.tif
  step2_record.json

03.registration/
  *_rigid.tif
  *_affine.tif
  *_registration.tif
  *_overlay_on_step1_rgb.tif
  *_weight_from_step2_dense_weight.tif
  *_registration.h5
  *_metric_history.csv
  *_step3_record.json

04.apply_label/
  *_label.tif
  *_overlay.tif
  *_region_distribution.csv
  *_annotation_nissl_merge.tif
  *_step4_record.json

05.roi_mask/
  *_mask.tif
  *_mask_fullres_gray.png
  roi_mask_report.csv
  step5_record.json
```

Each step writes a JSON record used automatically by the next step.

## Citation

This project uses the Allen Mouse Brain Common Coordinate Framework. Please cite:

Wang, Q., et al. (2020). *The Allen Mouse Brain Common Coordinate Framework: A 3D Reference Atlas*. Cell, 181(4), 936–953.e20.
