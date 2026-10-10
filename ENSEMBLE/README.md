# Image Classification Ensemble

## Notebooks

- **`dino-v3.ipynb`** — DINOv3 model using **336 × 336** input resolution.
- **`dino-v3-448x448.ipynb`** — DINOv3 model using **448 × 448** input resolution.
- **`dino-520.ipynb`** — DINOv3 model using **520 × 520** input resolution.
- **`evaaa.ipynb`** — EVA model using **448 × 448** input resolution.
- **`FTR maker.ipynb`** — Extracts frozen DINOv3 features at multiple resolutions (**336, 448, 560, 672, 784**) and generates out-of-fold and test predictions using logistic regression.
- **`submission-5-blend (1).ipynb`** — Combines predictions from the individual models, including EVA and frozen-feature models, into the final ensemble submission.

