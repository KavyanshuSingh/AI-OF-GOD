# AI OF GOD - 5: Shape-Based Animal Classification from Segmentation Maps

**Team:** Kavyanshu Singh, Karndevsinh Vala, Mitra Hari
**Institute:** IIT (ISM) Dhanbad
**Competition:** AI of GOD 5 (Kaggle)
**Repository:** [github.com/kavyanshusingh/ai-of-god](https://github.com/kavyanshusingh/ai-of-god)

A diverse ensemble of fine-tuned DINOv3 runs at several resolutions plus frozen multi-resolution heads, selected with out-of-fold (OOF) evidence and not by leaderboard probing. Public leaderboard score: about **0.941** (macro F1).

---

## Overview

This project builds a high-accuracy classifier for a colour-segmented animal dataset of 72 classes, using a deep ensemble of Vision Transformer based models.

Colours in the segmentation maps are arbitrary (even within a class), so the models are trained to rely on **shape and layout** rather than colour or texture.

**Stack:** PyTorch, torchvision, PIL, scikit-learn. Self-supervised DINO backbones are either fine-tuned or frozen, and models are evaluated on macro F1 with out-of-fold validation.

## Problem and Dataset

**Task:** classify the animal in a semantic-segmentation map (animal + ground, sky, plants, objects).

| Item | Value |
|---|---|
| Train images | 3,087 |
| Test images | 1,179 |
| Classes | 72 |
| Mean / class | 42.9 |
| Max / class | 51 (hedgehog) |
| Min / class | 23 (jellyfish) |
| Median size | ≈350×500 px |

Challenges:

- Colours are arbitrary, so a colour-invariant model is needed
- Only ~43 images per class
- Strong pose and texture variation
- Metric is macro F1; public/private splits may cover different classes

## Pipeline

```
PNG map (RGB) -> Letterbox to S×S (black pad) -> GPU augment (train only)
             -> DINOv3 ViT-L/16 (last 6 blocks tuned) -> Softmax (72 classes)
```

- 5-fold CV, EMA weights, class-weighted cross entropy
- 8-pass TTA: 4 channel permutations × horizontal flip
- The same split is used everywhere: `StratifiedKFold(5, shuffle=True, random_state=0)`, so every run's OOF predictions are comparable and blendable.

### Preprocessing

1. **Resize:** letterbox every map to the model's input size (S × S) with black padding, preserving the animal's aspect ratio.
2. **Augmentation:** applied on the GPU, per batch, during training only. Colour is destroyed by augmentation, not by preprocessing.
3. **Normalize:** ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Pre-trained models expect ImageNet-style inputs, and the dataset was too small to use its own statistics.

### Data Augmentation (GPU, per batch)

| Group | Transform | Probability / range |
|---|---|---|
| Colour destruction | random 3×3 channel mixing + random monotone tone curve | 60% |
| | random channel permutation | 20% |
| | grayscale | 20% |
| | inversion (1 − x) | 15% |
| Degradation | 5×5 box blur | 30% |
| | Gaussian noise (std ≤ 0.05) | 50% |
| Geometry | horizontal flip | 50% |
| | rotation | ±20° |
| | zoom / aspect jitter / translation | 0.75–1.25 / ±10% / ±5% |
| | cutout (solid random colour) | 30%, up to 20% side |
| Padding | letterbox kept black after colour changes | always |

Not used: mixup, cutmix, random-resized crops, elastic warp, vertical flip.

## Training Recipe

- Backbone frozen except last 6 blocks + final norm
- AdamW, weight decay 0.05
- Layer-wise LR decay 0.85 (top 3e-5), head 1e-3
- Class-weighted cross entropy, label smoothing 0.1
- Dropout 0.3, grad clip 1.0, fp16
- EMA of weights (evaluated copy)
- Batch 8; one fold ≈ 38 min @336

**Why training loss looks high:** with label smoothing 0.1 over 72 classes the loss floor is ≈0.74. Fold-4 training loss reached 0.80, so the model had fit the training set.

## Inference

- 4 channel permutations × horizontal flip = 8 passes, softmax averaged
- Test prediction = mean over the 5 fold models
- No multi-scale TTA

## Roadmap of Experiments

| Phase | Idea | Best result | Public LB |
|---|---|---|---|
| 1 | Silhouette / region maps + DINOv2 ViT-B | F1 0.604 (val) | – |
| 2 | Raw map + colour-destroying aug, ViT-B/14 @224 | CV acc 0.949 | 0.810 |
| 3 | DINOv2 ViT-L/14 @336, stronger aug + TTA | CV acc 0.979 | 0.839 |
| 4 | CNN baselines: EfficientNet-B3, Stylized-IN ResNet-50 | acc 0.94 / 0.84 | – |
| **5** | **DINOv3 ViT-L/16 @336** | **CV F1 0.983** | **0.940** |
| 7 | Frozen DINOv3 multi-resolution heads + ensembles | 41–46 OOF errors | +0.001 |
| 8 | DINOv2 ViT-B/14 @224, colour jitter + flip aug | CV F1 0.946 | 0.856 |
| 9 | DINOv3 ViT-B/16 @320, colour + geometric aug | – | 0.889 |

### Phase 3: DINOv2 ViT-L/14 @336

| Run | OOF | Public LB |
|---|---|---|
| ViT-B/14 @224 (Phase 2) | acc 0.949 / F1 0.948 | 0.810 |
| ViT-L/14 @336, strong aug + 8-pass TTA | acc 0.980 / F1 0.979 | 0.839 |

CV improved by +3.0 points and the leaderboard by +2.9, but a gap of ≈14 points between CV and leaderboard remained.

### Phase 4: CNN and Stylized-ImageNet Baselines

| Model | Setup | Acc | Macro F1 |
|---|---|---|---|
| EfficientNet-B3 (ns, JFT) | fold 0, @336, 25 ep | 0.940 | 0.937 |
| EfficientNet-B3 | fold 1 | 0.943 | 0.942 |
| Stylized-ImageNet ResNet-50 (Geirhos et al.) | fold 1, @336, 25 ep | 0.835 | 0.833 |
| DINOv2-L (reference) | fold 1 | 0.982 | 0.982 |
| DINOv2-L + SIN-ResNet50 blend | weight 0.1 / 0.3 / 0.5 | 0.984 / 0.982 / 0.976 | 0.984 / 0.982 / 0.976 |

- SIN-ResNet50 (strong shape bias) was 15 points weaker than DINOv2-L and did not help the blend (+1 image at weight 0.1, noise)
- No stylized ViT checkpoint was available; none was trained
- A plain (non-stylized) ResNet was not run separately

### Phase 5: DINOv3 ViT-L/16, the Breakthrough

| Backbone | OOF F1 | OOF errors | Public LB |
|---|---|---|---|
| DINOv2 ViT-L/14 @336 | 0.979 | – | 0.839 |
| DINOv3 ViT-L/16 @336 | 0.983 | 52 / 3,087 | 0.940 |

About **+10 leaderboard points from the backbone alone.** The augmentation, TTA and training recipe stayed the same; only the backbone changed (plus patch size 16 and a rewritten GPU pipeline with a black-letterbox mask). Likely cause: DINOv3's pretraining transfers better to unseen test maps. Not proven.

### Resolution and Architecture Variants (folds 0–1, 1,236 rows)

| Model | Errors | Hard-subset acc | Note |
|---|---|---|---|
| DINOv3-L @336 | 16 | 0.939 | reference |
| DINOv3-L @448 | 14 | 0.947 | 2 folds; fold ≈ 1.2–1.5 h |
| DINOv3-L @336 + @448 (50/50) | 12 | 0.956 | public score slightly lower |
| EVA02-L @448 | 44 | 0.826 | different family, much weaker |
| DINOv3-L @560 | – | – | 2 folds; no gain in any blend |

Higher resolution helped OOF by 2 images but did not show on the leaderboard. EVA02 added diversity but was too weak to help blends.

### Frozen DINOv3 Features at Many Resolutions

Features: CLS + mean patch token, flip TTA; logistic regression (C=0.5) on standardised features. 5-fold OOF, 3,087 images.

| Res. | LR errors | kNN errors |
|---|---|---|
| 336 | 71 | 98 |
| 448 | 63 | 99 |
| 560 | 59 | 94 |
| 672 | 56 | 90 |
| 784 | 52 | 94 |
| all-res mean | 54 | 90 |

Resolution keeps helping the frozen head; at 784 it equals the fine-tuned D3L336 model (52 errors). kNN was dropped.

## Ensembles Tested (OOF errors, lower is better)

**All 3,087 rows**

| Ensemble | Errors | Fixed | Broken |
|---|---|---|---|
| D3L336 alone | 52 | – | – |
| D3L336 + frozen LR @448 (w=0.5) | 48 | – | – |
| D3L336 + frozen LR (336/448/672), w=0.5 | 46 | – | – |
| D3L336 + D3L448 (reference) | 45 | 0 | 0 |
| + frozen LR (560/672/784) | 45 | 6 | 6 |
| + frozen LR (all res) | 42 | 7 | 4 |
| + frozen LR @784 | 41 | 10 | 6 |

**Folds 0–1 only (1,236 rows)**

| Ensemble | Errors |
|---|---|
| Reference | 12 |
| + D3L560 + frozen | 14 |
| + D3L560 + EVA448 + frozen | 12 |
| Five-way (submitted) | 12 |

Differences of a few images are within noise: the ensemble has converged.

## Final Submissions

1. **Five-way ensemble** (equal weights): D3L336 (5 folds) + D3L448 (5 folds) + D3L560 (2 folds) + EVA448 (2 folds) + frozen DINOv3 LR (336/448/672)
2. **D3L336 + frozen DINOv3 heads**

The two submissions differ by 5 labels.

## Lessons

- Keep the raw map; destroy colour with augmentation, not preprocessing
- Backbone choice mattered most (DINOv2 → DINOv3: +10 LB points)
- OOF saturates near 0.98 and cannot rank near-equal models
- Macro F1 on a class-restricted split amplifies every error
- Ensembles of near-identical models converge; they add little

## Model Naming

| Short name | Model |
|---|---|
| D3L336 / D3L448 / D3L560 | DINOv3 ViT-L/16 fine-tuned at 336 / 448 / 560 px |
| EVA448 | EVA02-L fine-tuned at 448 px |
| Frozen LR | Frozen DINOv3-L features + logistic regression at 3+ resolutions |

## Acknowledgements

Thank you to the AI of GOD organisers and to everyone who reviewed our work. We look forward to exploring future improvements and refined methodologies.
