## Lumbar Ultrasound Bone Segmentation

This repository contains a full pipeline for segmenting lumbar bone in ultrasound frames using U-Net–style models, plus utilities for data selection, sanity checks, pseudo‑labeling, and visualization.

The code is written in Python with PyTorch and is designed to be easy to reproduce and extend.

---

## 1. Project structure

At the top level:

- **`scripts/`**: all training, inference, and utility scripts.
- **`data/`**: images, masks, splits, pseudo‑labels, etc. (you create/populate this).
- **`models/`**: saved model weights (`.pth`).
- **`notebooks/`**: optional Jupyter notebooks (currently empty in this repo).

Key subfolders expected under `data/`:

- **`data/images/`**: grayscale ultrasound frames (`.jpg`).
- **`data/masks_manual/`**: manually drawn binary masks (`.png`) for labeled images.
- **`data/splits/`**: text files with train/val/test image filenames.
- **`data/raw_unlabelled/`**: raw frames without manual masks.
- **`data/to_label/`**: small subset of `raw_unlabelled` recommended for manual annotation.
- **`data/masks_pred/`**, **`data/overlays_pred/`**: predictions & overlays using the big U‑Net model.
- **`data/masks_pseudo_small/`**, **`data/overlays_pseudo_small/`**: predictions & overlays from the small U‑Net (pseudo‑labels).

---

## 2. Environment setup

The repo **does not** track any virtual environment. You should create your own:

```bash
cd lumbar_model

python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate   # on Windows PowerShell
```

Install dependencies (minimal set, adjust as needed):

```bash
pip install \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
  segmentation-models-pytorch \
  numpy \
  pillow \
  matplotlib \
  opencv-python \
  tqdm \
  scipy
```

If you have a CUDA GPU, install the CUDA‑enabled PyTorch wheel instead (see the official PyTorch install page).

---

## 3. Data layout and naming conventions

The training code assumes:

- All **images** live in `data/images/` with names like:
  - `1.jpg`, `2.jpg`, ..., or generally `<id>.jpg`
- All **manual masks** live in `data/masks_manual/` with the **same stem** but `.png`:
  - `1.png` corresponds to `data/images/1.jpg`
- Image format:
  - Scripts load images as **grayscale**.
  - Internally everything is resized to **256×256** before feeding into the networks.
- Masks:
  - Binary masks, stored as single‑channel `.png` with foreground > 127.

If your data is in a different folder or file naming scheme, you can either:

- Rename/move the files to match this layout, or
- Modify the configuration constants at the top of the scripts (e.g. `IMG_DIR`, `MASK_DIR`, `DATA_ROOT`).

---

## 4. Overview of the pipeline

The repository is organized around two U‑Net models and accompanying utilities:

- **`train_unet.py`**: trains a **ResNet‑34 U‑Net** (from `segmentation_models_pytorch`) on manually labeled data and saves `models/unet_best.pth`.
- **`predict_all_raw.py`**: uses `unet_best.pth` to generate full‑resolution masks and overlay images for **unlabelled** frames in `data/raw_unlabelled/`.
- **`simple_unet.py` + train_unet_small.py`**: defines and trains a lightweight U‑Net (`SimpleUNet`) with a **BCEDice** loss.
- **`predict_unet_small.py`**: uses the small model to generate pseudo‑labels on unlabelled data, writing `data/masks_pseudo_small/` and `data/overlays_pseudo_small/` plus a CSV of statistics.

Supporting utilities:

- **`make_splits.py`**: creates train/val/test splits from the labeled images.
- **`dataset.py`**: implements `UltrasoundBoneDataset`, used by the training scripts.
- **`select_frames_for_manual.py`**: scores unlabelled frames by brightness in a plausible depth band and copies the top‑N to `data/to_label/` for manual annotation.
- **`move_annotated.py`**: separates images with masks from raw images.
- **`check_pairs.py`**: sanity‑checks image ↔ mask pairs and reports any mismatches/bad masks.
- **`show_pairs.py` / `show_samples.py` / `predict_visualize.py`**: visualization utilities for data sanity and qualitative model inspection.
- **`analyze_masks.py`**: computes statistics (mask height range, area fraction) over existing masks to inform reasonable priors and thresholds.
- **`test_dataloader.py`**: quick smoke test for the dataset and dataloader shapes.

The typical workflow is:

1. Prepare raw ultrasound frames.
2. Use `move_annotated.py` and/or `select_frames_for_manual.py` to define labeled vs unlabelled sets.
3. Run `make_splits.py` to generate train/val/test lists.
4. Run `train_unet.py` (or `train_unet_small.py`) to train a model.
5. Use `predict_visualize.py` and `show_*` scripts to inspect model performance.
6. Use `predict_all_raw.py` or `predict_unet_small.py` to generate masks for the raw, unlabelled frames (pseudo‑labelling).

The sections below describe each major script and how to run it.

---

## 5. Datasets and splits

### 5.1 `dataset.UltrasoundBoneDataset`

- Defined in `scripts/dataset.py`.
- Expects:
  - `img_dir`: directory containing `.jpg` images.
  - `mask_dir`: directory containing `.png` masks with matching stem.
  - `file_list_path`: path to a `.txt` file where each line is an image filename (e.g. `1.jpg`).
- Behavior:
  - Loads image and mask as grayscale.
  - Resizes both to `img_size` (default 256).
  - Normalizes image to \([0, 1]\).
  - Binarizes mask to `{0, 1}`.
  - Returns `img` and `mask` as tensors of shape `[1, H, W]`.

### 5.2 Creating train/val/test splits: `scripts/make_splits.py`

Once you have `data/images/` and `data/masks_manual/`:

```bash
cd scripts
python make_splits.py
```

What it does:

- Finds all images in `data/images` that **have** a corresponding mask in `data/masks_manual`.
- Randomly shuffles them.
- Splits into:
  - 80% train
  - 10% val
  - 10% test
- Writes:
  - `data/splits/train.txt`
  - `data/splits/val.txt`
  - `data/splits/test.txt`

You can edit these text files manually if you want specific images to go into a given split.

---

## 6. Training the models

### 6.1 Training the main ResNet‑34 U‑Net: `scripts/train_unet.py`

This is the heavier model using `segmentation_models_pytorch.Unet` with a ResNet‑34 encoder.

From the repo root:

```bash
cd scripts
python train_unet.py
```

Configuration (edit at top of the file if needed):

- **Data paths**:
  - `DATA_ROOT = Path("data")`
  - `IMG_DIR = DATA_ROOT / "images"`
  - `MASK_DIR = DATA_ROOT / "masks_manual"`
  - `SPLIT_DIR = DATA_ROOT / "splits"`
- **Training hyperparameters**:
  - `BATCH_SIZE = 4`
  - `IMG_SIZE = 256`
  - `EPOCHS = 15`
  - `LR = 1e-4`
  - `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

During training:

- Uses `UltrasoundBoneDataset` for train and val sets.
- Optimizer: `AdamW`.
- Loss: `BCEWithLogitsLoss`.
- Metric: Dice coefficient computed from thresholded predictions.
- After each epoch:
  - Prints train/val loss and Dice.
  - Saves the best model (by val Dice) to `models/unet_best.pth`.

### 6.2 Training the small U‑Net: `scripts/train_unet_small.py`

This trains a lighter `SimpleUNet` defined in `scripts/simple_unet.py` using a hybrid **BCE + Dice** loss.

Run:

```bash
cd scripts
python train_unet_small.py
```

Key configurations:

- Same data layout and splits as `train_unet.py`.
- `EPOCHS = 80` (longer training).
- Model: `SimpleUNet(in_channels=1, out_channels=1, base_ch=32)`.
- Loss: `BCEDiceLoss` (defined in `scripts/losses.py`):
  - \(\text{loss} = \alpha \cdot \text{BCEWithLogits} + (1 - \alpha) \cdot (1 - \text{Dice})\)
  - Default `alpha = 0.5`.
- Best model checkpoint:
  - Saved to `models/unet_small_pseudo.pth`.

If you modify the architecture or loss, update `simple_unet.py` or `losses.py` accordingly.

---

## 7. Working with unlabelled data

### 7.1 Moving annotated vs raw frames: `scripts/move_annotated.py`

If you start with everything in `data/images/` and a subset has masks in `data/masks_manual/`:

```bash
cd scripts
python move_annotated.py
```

Behavior:

- Reads masks from `data/masks_manual/` where filenames contain a numeric ID followed by `_mask` (e.g. `123_mask.png`).
- Collects those IDs.
- For each image in `data/images/`:
  - If its stem **matches** a mask ID, it is considered **annotated** and left in place.
  - Otherwise, it is moved into `data/raw_unlabelled/`.

Adjust the mask filename pattern if your naming is different.

### 7.2 Selecting frames for manual annotation: `scripts/select_frames_for_manual.py`

This script picks the **top‑N most promising frames** from `data/raw_unlabelled/` based on brightness in a central depth band (heuristic for visibility of bone).

Run:

```bash
cd scripts
python select_frames_for_manual.py
```

What it does:

- For each frame in `data/raw_unlabelled/`:
  - Converts to grayscale.
  - Looks at the region between 30–60% of the image height.
  - Computes the mean of the top 5% brightest pixels in that region.
- Ranks all frames by this score (descending).
- Takes the top `N` (default `N = 120`) and copies them into `data/to_label/`.

You can then manually draw masks for `data/to_label/` and save them into `data/masks_manual/` to expand your labeled dataset.

---

## 8. Predicting masks and generating pseudo‑labels

### 8.1 Using the main U‑Net on raw data: `scripts/predict_all_raw.py`

This script runs the **trained ResNet‑34 U‑Net** (`models/unet_best.pth`) on every image in `data/raw_unlabelled/` and saves:

- Binary masks in `data/masks_pred/` as `.png` (0/255).
- RGB overlays in `data/overlays_pred/` with the mask blended in red on top of the grayscale image.

Run:

```bash
cd scripts
python predict_all_raw.py
```

Key details:

- Images are resized to 256×256 for inference, then masks are upscaled back to original resolution.
- A post‑processing step:
  - Keeps only a vertical band between 10–90% of the height.
  - Removes tiny connected components below `min_area_frac` of the image area.
  - Optionally falls back to the largest component if post‑processing wipes everything out.

You can adjust thresholds in `postprocess_mask` / `keep_largest_component` if your anatomy distribution differs.

### 8.2 Using the small U‑Net for pseudo‑labels: `scripts/predict_unet_small.py`

This script uses `models/unet_small_pseudo.pth` to generate **pseudo‑labels** for all frames in `data/raw_unlabelled/`.

Run:

```bash
cd scripts
python predict_unet_small.py
```

Output:

- `data/masks_pseudo_small/<stem>.png`: binary mask (0/255).
- `data/overlays_pseudo_small/<stem>_overlay.png`: overlay of mask on original grayscale frame.
- `data/pseudo_stats_small.csv`: one row per image with:
  - `filename`
  - `area_frac` – mask area fraction relative to image area.
  - `cy_rel`, `cx_rel` – relative center of mass of the mask in \([0,1]\).
  - `plausible` – heuristic flag (`1` or `0`) based on area fraction.

You can filter pseudo‑labels by `plausible == 1` to construct a high‑confidence pseudo‑labeled set.

---

## 9. Visualization and sanity checks

### 9.1 Checking image–mask pairs: `scripts/check_pairs.py`

Run:

```bash
cd scripts
python check_pairs.py
```

It will:

- Count how many images and masks it finds.
- Report:
  - Images that have **no** matching mask.
  - Masks that have **no** matching image.
- Check all matched masks for:
  - Readability.
  - Non‑emptiness (flags all‑zero masks).

### 9.2 Quick sample visualization: `scripts/show_samples.py`

Nice for quickly seeing if masks are aligned and look reasonable.

```bash
cd scripts
python show_samples.py
```

It:

- Randomly selects 5 pairs from `data/images` and `data/masks_manual`.
- Plots:
  - Image.
  - Mask.
  - Simple grayscale overlay (`cv2.addWeighted`).

### 9.3 Colored overlays: `scripts/show_pairs.py`

```bash
cd scripts
python show_pairs.py
```

It:

- Finds images and masks by matching stems.
- Samples up to 6 random matched pairs.
- Creates colored overlays where the mask region is tinted red on the grayscale image.

### 9.4 Visualizing model predictions on labeled data: `scripts/predict_visualize.py`

```bash
cd scripts
python predict_visualize.py
```

It:

- Loads `models/unet_best.pth`.
- Reads filenames from:
  - `data/splits/val.txt`
  - `data/splits/test.txt`
- For a few examples, displays:
  - Original image.
  - Ground‑truth mask.
  - Predicted mask.
  - Red overlay of prediction on the image.

Use this to qualitatively inspect how well the model is doing, especially on the test set.

---

## 10. Inspecting mask distribution: `scripts/analyze_masks.py`

Run:

```bash
cd scripts
python analyze_masks.py
```

It analyzes all masks in `data/masks_manual` and prints:

- Relative minimum and maximum vertical positions of mask pixels.
- Mean/min/max area fraction of the mask relative to the image.

These stats are used to choose reasonable thresholds in:

- Post‑processing of predicted masks.
- Pseudo‑label filtering (e.g. in `predict_unet_small.py`).

You can re‑run this after updating or expanding your manual masks to see how your distribution changes.

---

## 11. Quick smoke test of the dataloader: `scripts/test_dataloader.py`

After setting up data and splits, you can quickly ensure that the dataset and dataloader work:

```bash
cd scripts
python test_dataloader.py
```

Expected output:

- Dataset size (number of training samples).
- Shapes of a single batch, e.g.:
  - `Batch images shape: torch.Size([4, 1, 256, 256])`
  - `Batch masks shape:  torch.Size([4, 1, 256, 256])`

If this fails, fix the data layout / filenames before starting long trainings.

---

## 12. Tips and common gotchas

- **Virtualenv not tracked**:
  - The repo ignores `.venv`, `.venv310`, etc., on purpose. Always create your own environment.
- **Large data**:
  - You typically do **not** want to push raw ultrasound data to GitHub (size, privacy).
  - Keep large data folders (`data/raw_unlabelled`, `data/masks_pred`, etc.) locally or on a separate storage.
- **File naming**:
  - Most scripts assume that image and mask stems match (`1.jpg` ↔ `1.png`).
  - If your manual masks use a suffix like `_mask`, adjust the relevant script or rename files.
- **Training on GPU**:
  - If `torch.cuda.is_available()` is `False`, training will fall back to CPU and be slower.
  - To use GPU, install a CUDA‑enabled PyTorch build and ensure a compatible NVIDIA driver.

---

## 13. Minimal “getting started” checklist

If you have a new machine and clone this repo, a typical run might look like:

```bash
git clone https://github.com/<your-user>/lumbar_model.git
cd lumbar_model

python -m venv .venv
source .venv/bin/activate
pip install -r <(printf "torch\nsegmentation-models-pytorch\nnumpy\npillow\nmatplotlib\nopencv-python\ntqdm\nscipy\n")

mkdir -p data/images data/masks_manual
# copy your labeled images into data/images and masks into data/masks_manual

cd scripts
python make_splits.py
python test_dataloader.py
python train_unet.py          # or: python train_unet_small.py
python predict_visualize.py   # inspect results
```

From there, you can explore:

- `predict_all_raw.py` and `predict_unet_small.py` for pseudo‑labeling and bulk prediction.
- `select_frames_for_manual.py` to grow your labeled dataset intelligently.

If you run into any mismatch between your data layout and these assumptions, the quickest fix is usually to rename/move files and then re‑run `make_splits.py`.