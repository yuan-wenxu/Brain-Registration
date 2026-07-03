# Brain-Registration Project Guidelines

## Project Purpose

This project registers 2D mouse brain sections to standard sections from the Allen Mouse Brain Atlas, applies Allen annotations to the experimental sections, and optionally generates masks for specified ROIs.

The environment is managed exclusively with Pixi. The user-level CLI command is `br`, whose entry point is `run_registration.sh`.

## Processing Pipeline

1. `script/step1_preprocess.py`: Convert to grayscale, normalize resolution, extract the tissue mask, and place the tissue on a fixed canvas.
2. `script/step2_select_slice.py`: Search the 3D Allen Nissl atlas for the best-matching 2D section. Candidate scoring temporarily runs rigid, affine, and B-spline registration, but does not persist a final transform.
3. `script/step3_registration.py`: Perform rigid, affine, and B-spline refinement on the selected atlas section and save the composite transform.
4. `script/step4_apply_label.py`: Apply the Step3 transform chain to the Allen annotation with nearest-neighbor resampling and export region statistics.
5. `script/step5_roi_mask.py`: Optionally generate individual and high-resolution masks from an Allen structure tree and an ROI list.

Pipeline stages pass paths, parameters, and spatial metadata through `*_record.json` files. When changing an output field, inspect and update every downstream consumer.

## Key Files

- `run_registration.sh`: Full pipeline entry point and configuration defaults. Check this file first when changing the CLI or configuration options.
- `config/registration.conf.example`: Tracked configuration template.
- `config/registration.conf`: Local path configuration. Never add machine-specific absolute paths to the template or source code.
- `script/utils/registration.py`: Shared rigid, affine, and B-spline registration implementation.
- `script/convert_nrrd_to_npy.py`: Converts official Allen NRRD data to the NPY format consumed by the pipeline.
- `docs/TECHNICAL_DOCUMENTATION.md`: Detailed algorithm, data-flow, and output documentation.

## Environment and Execution

Install the environment and CLI:

```bash
pixi run init
source ~/.bashrc
br --help
```

Create the local configuration:

```bash
cp config/registration.conf.example config/registration.conf
```

Run the full pipeline:

```bash
br rgb --input /path/to/brain.tif --rotation 0
br nissl --input /path/to/brain.tif --rotation 90
```

Run all Python commands inside the Pixi environment, for example:

```bash
pixi run python script/step1_preprocess.py --help
```

Do not assume that the system Python installation contains the project dependencies.

## Data and Spatial Constraints

- Allen Nissl and annotation inputs are both 3D `.npy` arrays. Annotation data must always use label-safe nearest-neighbor interpolation.
- The Step1 standard canvas has a width of `1140` and a height of `800`, giving a NumPy array shape of `(800, 1140)`. Do not confuse image width/height with NumPy row/column order.
- `INPUT_RES` and `TARGET_RES` are both expressed in `um/px`.
- Tissue layouts `whole`, `left`, `right`, and `auto` are supported. Changes to cropping or transform logic must cover half-brain cases.
- Images can be large. Prefer memory mapping, release intermediate objects promptly, and avoid unnecessary full-array copies.
- Intensity images, masks, and label images have different semantics. Explicitly preserve the correct dtype, interpolation method, and background value when saving or resampling them.
- Keep `RANDOM_SEED` configurable and propagated so slice selection and registration remain reproducible.

## Change Guidelines

- Keep each step script usable both through `br` and directly from the command line.
- When adding a pipeline parameter, update `run_registration.sh`, `registration.conf.example`, the README, and the technical documentation together.
- Prefer backward-compatible additions when changing record JSON structures. Add fields instead of renaming or removing existing fields.
- Path handling must support absolute paths and paths relative to the record JSON file.
- Do not commit atlas data, raw sections, generated results, or local configuration files.
- Do not overwrite unrelated changes already present in the working tree.

## Validation

The repository currently has no automated test suite. At minimum, run the checks relevant to the changed code:

```bash
pixi run python -m compileall script
pixi run python script/step1_preprocess.py --help
pixi run python script/step2_select_slice.py --help
pixi run python script/step3_registration.py --help
pixi run python script/step4_apply_label.py --help
pixi run python script/step5_roi_mask.py --help
bash -n run_registration.sh install-cli.sh
```

Changes involving algorithms or spatial transforms also require an end-to-end run on a small real sample. Inspect the tissue mask, slice-score curve, registration overlay, annotation boundaries, and ROI masks. A successful process exit alone is not sufficient evidence of correctness.
