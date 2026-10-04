# ToxCNN: Drug Toxicity Prediction from Cell Morphology

**AI for Life Science Challenge — Pazhou Algorithm Competition**
**Submission Category:** Model & Algorithm

## Project Summary

ToxCNN predicts drug toxicity from high-content fluorescence microscopy images
of cells, using transfer learning on a pretrained ResNet18. Given a 3-channel
fluorescence composite image (DNA/DAPI, microtubules/Tubulin, actin
cytoskeleton/Actin) of MCF7 cells treated with a compound, the model classifies
the treatment as **toxic** or **non-toxic** relative to the DMSO vehicle control.

The model is trained and evaluated on the [BBBC021](https://bbbc.broadinstitute.org/BBBC021)
benchmark dataset, using a leave-one-compound-out cross-validation protocol
to ensure the model is evaluated on compounds it has never seen during
training — a critical requirement for honest generalization claims on this
type of correlated, small-sample biological imaging data.

**Headline result:** 90.6% ± 12.2% accuracy, 88.3% ± 14.7% macro-F1 under
leave-one-compound-out cross-validation across 7 compounds spanning 6
distinct toxicity mechanisms.

## Problem & Motivation

High-content imaging generates large volumes of cell images from drug
screening experiments, but manually assessing toxicity from these images is
slow and subjective. An automated, interpretable classifier can help
prioritize compounds for further testing and reduce the cost of early-stage
toxicity screening in drug development, including for organ-on-a-chip
platforms.

## Data

- **Source:** BBBC021 (Broad Bioimage Benchmark Collection), a public,
  freely available dataset of MCF7 breast cancer cells imaged after
  treatment with 38 compounds across 12 known mechanisms of action (MOA).
- **Subset used:** 3 of ~10 available plate batches (Week1_22123,
  Week2_24121, Week2_24141), due to compute/storage constraints.
  228 labeled images across 7 compounds + DMSO control.
- **Labels:** Binary toxic/non-toxic, derived from MOA annotation — DMSO
  (vehicle control, no active compound) = non-toxic; all 6 active MOA
  classes present in this subset = toxic.
- **License/compliance:** BBBC021 is a public research dataset, freely
  available for academic and research use. No personal, clinical, or
  restricted data was used.

## Method

1. **Image preprocessing:** Combine the 3 single-channel fluorescence TIFFs
   (DAPI, Tubulin, Actin) per field of view into a single RGB composite
   image, with per-channel percentile normalization.
2. **Model:** ResNet18 pretrained on ImageNet. Early layers frozen; `layer4`
   and the final classification head fine-tuned on the toxicity task.
   Class-weighted cross-entropy loss to handle class imbalance.
3. **Splitting:** Train/val/test splits are constructed **by compound**, not
   by image, to prevent information leakage between highly-correlated
   images from the same treatment condition.
4. **Evaluation:** Leave-one-compound-out (LOCO) cross-validation —
   the model is trained on 6 compounds and tested on the 1 held-out
   compound, repeated for all 7. This gives an honest estimate of
   generalization to entirely unseen compounds, which a single train/test
   split cannot provide with this few compounds.
5. **Interpretability:** Grad-CAM is used to visualize which image regions
   drive the model's toxic/non-toxic predictions, to sanity-check that the
   model attends to actual cell structures rather than background or
   imaging artifacts.

## Results

| Held-out compound | # test images | Accuracy | Macro-F1 |
|---|---|---|---|
| taxol              | 106 | 1.000 | 1.000 |
| cytochalasin B     | 30  | 1.000 | 1.000 |
| cytochalasin D     | 30  | 1.000 | 1.000 |
| nocodazole         | 30  | 1.000 | 1.000 |
| methotrexate       | 30  | 0.833 | 0.722 |
| ALLN               | 38  | 0.789 | 0.756 |
| AZ-A               | 46  | 0.717 | 0.700 |
| **Mean ± std**     |     | **0.906 ± 0.122** | **0.883 ± 0.147** |

Full per-fold results: `loco_cv_results.csv`

### Grad-CAM findings

Grad-CAM visualizations (`gradcam_*.png`) show the model's attention
consistently concentrated on dense regions of actual cells rather than
background, supporting that it has learned real morphological features.
Heatmaps from the final convolutional block are relatively coarse (7×7
feature map upsampled), so they indicate general attended regions rather
than precise sub-cellular localization — finer-grained attribution methods
would be a natural next step.

## Limitations

- **Small compound diversity (7 compounds, 1 subset of BBBC021):** more
  plates/compounds would substantially strengthen generalization claims.
- **Class imbalance:** toxic images outnumber non-toxic (DMSO) images
  roughly 2:1, addressed with class-weighted loss but still a source of
  potential bias.
- **AZ-A (Aurora kinase inhibitor) is the hardest compound** to classify —
  likely because its morphological toxicity signature is subtler than the
  more dramatic cytoskeletal disruption seen in taxol/cytochalasin/
  nocodazole-treated cells.
- **Binary toxic/non-toxic framing** collapses 6 distinct mechanisms of
  action into one class; a multi-class MOA model (not pursued here due to
  data scarcity per class) would carry more biological information.
- Grad-CAM resolution limits precise localization (see above).

## Reproduction Instructions

### Requirements
See `requirements.txt`. GPU recommended (Colab free tier T4 is sufficient).

### Steps
1. Download BBBC021 metadata and image batches (see `full_pipeline.py`,
   Cells 1-3) — URLs point to the official Broad Institute host.
2. Run `full_pipeline.py` top to bottom (cell markers `# %%`) in a Jupyter/
   Colab environment to reproduce data preprocessing, training, and LOCO-CV
   evaluation.
3. Use `inference.py` to run the trained model on a new image set (see
   below).

### Quick inference on new images
```bash
python inference.py --model best_model.pt --image_dir path/to/images --metadata path/to/metadata.csv
```

## Team

Solo submission — AI/ML background, no prior bio/organ-on-a-chip domain
expertise. (No cross-disciplinary bonus applicable.)

## Acknowledgments

- BBBC021 dataset: Caie et al., Broad Bioimage Benchmark Collection
- ResNet architecture: He et al., 2016
