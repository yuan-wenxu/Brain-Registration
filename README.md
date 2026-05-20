# Brain-Registration

Five-step brain slice registration pipeline (Step1~Step5, Step5 optional):

1. Preprocessing (grayscale, resampling, tissue mask)
2. Slice search + affine registration (to Allen Nissl)
3. Nonlinear refinement (B-spline)
4. Apply transforms to annotation (Allen annotation)
5. Generate ROI masks from Step4 result (using ROI list + structure tree)

Script directory: `script/`

Detailed technical documentation: [TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)

- `run_registration_pipeline.py`: main entry (recommended)
- `step1_preprocess.py`
- `step2_affine.py`
- `step3_nonlinear.py`
- `step4_apply_label.py`
- `step5_roi_mask.py` (optional)

---

## 1. Environment

Recommended: Python 3.9+ with the following dependencies:

- `numpy`
- `scikit-image`
- `SimpleITK`
- `matplotlib`

Example install command (adapt to your environment manager):

```bash
pip install numpy scikit-image SimpleITK matplotlib
```

Pixi environment configuration files are also provided here.

---

## 2. Quick Start (Recommended)

Run from the project root:

```bash
python ./script/run_registration_pipeline.py \
	--data-path /path/to/your/whole_brain.tif \
    --atlas-nissl /path/to/Allen_nissl_atlas \
    --atlas-annotation  /path/to/Allen_annotation_atlas \
    --input-res resolution (whole_brain.tif) \
    --target-res resolution (Allen)
```

Optional arguments (main entry):

- `--atlas-nissl`: Allen nissl atlas path
- `--atlas-annotation`: Allen annotation atlas path
- `--input-res`: input resolution
- `--target-res`: target resolution (default: `10.0` um/px)
- `--atlas-slice`: manually specify atlas slice (auto-search if omitted)
- `--slice-search-radius`, `--slice-search-step`, `--search-resize-max`: Step2 search range/speed-accuracy controls
- `--search-workers`: Step2 parallel workers (recommended `2~8`)
- `--sitk-threads`: SimpleITK thread count (use `1` for better reproducibility)
- `--neighbor-smooth-sigma`: Step2 slice-score smoothing parameter
- `--roi-txt-path`: Step5 ROI txt path (default uses `docs/ROI.txt` if present)
- `--structure-tree-csv`: Step5 structure tree csv path (`structure_tree_safe_2017.csv`)

---

## 3. Output Directory Structure

For input image `xxx.tif`, outputs are generated next to it as `xxx_registration_result/`:

```text
xxx_registration_result/
├── 01.preprocess/
├── 02.affine/
├── 03.nonlinear/
├── 04.apply_label/
└── 05.roi_mask/   (optional)
```

Key outputs:

- `02.affine/*_affine.tif`: Step2 affine result
- `02.affine/*_step2_record.json`: Step2 parameters and metrics
- `03.nonlinear/*_nonlinear.tif`: Step3 nonlinear result
- `03.nonlinear/*_step3_record.json`: Step3 parameters and metrics
- `04.apply_label/*_label.tif`: final warped annotation result
- `05.roi_mask/*_mask.tif`: one binary mask tif per ROI (optional)
- `05.roi_mask/roi_mask_report.csv`: ROI->id mapping and pixel statistics


<p align="center">
    <a href="docs/image/03.nonlinear/test.png">
        <img src="docs/image/03.nonlinear/test.png" alt="Nissl registration" width="49%" />
    </a>
    <a href="docs/image/04.apply_label/test.png">
        <img src="docs/image/04.apply_label/test.png" alt="Annotation registration" width="49%" />
    </a>
</p>

<p align="center">
    <sub>Left: Nissl registration | Right: Annotation registration (click images for full size)</sub>
</p>

## Atribution

This project utilizes data from the **Allen Brain Atlas**. For brain registration, we employed the Allen Mouse Brain Common Coordinate Framework (CCFv3) at a isotropic resolution of $10\,\mu m$ (Wang et al., 2020).
- If you use the brain mapping registration function provided by this project in your research, please be sure to cite the following core references and statements in accordance with the official requirements of the Allen Institute for Brain Science.
- Wang, Q., et al. (2020). The Allen Mouse Brain Common Coordinate Framework: A 3D Reference Atlas. Cell, 181(4), 936–953.e20. ([link](https://www.cell.com/cell/fulltext/S0092-8674(20)30402-5))