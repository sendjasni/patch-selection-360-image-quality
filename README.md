# Similarity-Preserving Instance Selection for Data-Efficient 360-Degree Image Quality Assessment

This repository contains the code associated with the article **Similarity-preserving instance selection with residual-based outlier detection for data-efficient omnidirectional image quality assessment**, currently under review.

Patch-based deep networks dominate omnidirectional (360-degree) image quality assessment, but the construction of their training sets stops at deciding *where to look*. The resulting patch collections are large and strongly correlated, so a substantial fraction of the training signal is redundant. This work formulates the overlooked stage that follows sampling — deciding *which* of the sampled patches to keep — as an instance-selection problem posed in embedding space.

The selection module sits between sampling and training and is agnostic to both the sampling operator and the downstream architecture: it never inspects either, only the embeddings the sampler produces.

![Study Diagram](fram.png)

**Figure**: Diagram illustrating the components of the proposed study. This repository provides the code for the *embedding similarity-selection* component.

**Note:** The code associated with patch sampling, encoding and quality estimation can be delivered on demand.

## Method

Patch embeddings `E` (2048-d, from a frozen ImageNet-pretrained ResNet-50) are mapped to a low-dimensional space by a transformation `W` constrained to preserve their pairwise similarity structure, while a residual matrix `R` absorbs the samples that structure cannot explain:

```
min_{W, R}  ‖E W − Z − R‖²_F  +  α ‖W‖_{2,1}  +  β ‖R‖_{2,1}
```

where `Z` is obtained from the eigendecomposition of the similarity matrix `S` such that `S = Z Zᵀ`. The `ℓ2,1` penalty on `W` keeps the mapping supported on few feature directions; the `ℓ2,1` penalty on `R` makes it row-sparse (one row per patch), so that the `ℓ2` norm of each row of `R` is a direct per-sample **irrelevance score**. Patches are ranked by that score in ascending order and the top-`k` are retained.

The problem is jointly convex and solved by alternating majorisation–minimisation with closed-form updates for `W` and `R`. The objective is proved to be non-increasing and converges within 6–8 iterations in practice. With `n ≪ d` (typical: `n ≈ 180`, `d = 2048`), the Woodbury form of the `W`-update costs `O(n² d + n³)` per iteration, and the whole selection stage costs `O(n·C_enc + n² d + n³ + t(n³ + n² d))` for `t` iterations. The cost is quadratic-to-cubic in the number of patches and independent of the downstream model.

## Key results

- A baseline regressor matches or exceeds its full-data accuracy while retaining only **40–60%** of the sampled patches on CVIQ, OIQA and MVAQD.
- This holds across three structurally different sampling operators (uniform ERP, latitude-aware LAT, scanpath-driven SP) and across Euclidean, Manhattan and covariance-weighted similarities.
- Used as a drop-in preprocessing module for four state-of-the-art models (SAP-Net, MC360IQA, SAL-360IQA, Assessor360), the same selection matches the full-data PLCC and SRCC in 11 of 12 model–dataset pairs while feeding the model **20–40% fewer input patches**, at equal or better accuracy.
- A multi-factor ANOVA shows that the similarity measure and the projection dimension `h` have no statistically detectable effect on performance, leaving the **selection rate as the only quantity that requires tuning**.

## Features

- **Distance metrics**: similarity matrix construction using Euclidean, Manhattan or a Mahalanobis-type quadratic form. Note that, following the manuscript (Section 3.4.2), the MAH variant uses the empirical covariance `Σ̂` directly rather than its inverse `Σ̂⁻¹`, which avoids an ill-posed inversion when `n < d`. This differs from the classical Mahalanobis distance.
- **Dimensionality reduction**: eigendecomposition of the similarity matrix to obtain the optimal low-dimensional target `Z`.
- **Joint optimisation**: closed-form alternating updates of the transformation matrix `W` and the residual matrix `R`, with monotone convergence.
- **Irrelevance scoring**: per-patch scores read directly off the row norms of the converged residual, with no separate heuristic criterion.

## Installation

Requires Python 3.8+ and the following packages:

```bash
pip install numpy scipy
```

## Usage

```bash
python emb_selection.py -sim [SIMILARITY_METRIC] -mat [PATH_TO_FEATURE_FILE]
```

A typical run:

```bash
python emb_selection.py -sim EUC -mat features.mat --h 10 --rate 0.5 --step 180 --out R_EUC.pkl
```

### Arguments

- `-sim`: similarity distance metric. Choose from `MAN` (Manhattan), `MAH` (Mahalanobis-type), or `EUC` (Euclidean). (Required)
- `-mat`: path to the `.mat` file containing the feature data. (Required)
- `--h`: projection dimension (default: `10`).
- `--alpha`: `ℓ2,1` weight on `W` (default: `0.1`).
- `--beta`: `ℓ2,1` weight on `R`; larger = sparser residual (default: `0.1`).
- `--rate`: selection rate, fraction of patches kept (default: `0.5`).
- `--step`: patches per image when this is constant (default: `180`).
- `--counts`: file of per-image patch counts, for variable-size pools (LAT, SP).
- `--max-it`: maximum alternating iterations (default: `50`).
- `--tol`: relative convergence tolerance (default: `1e-6`).
- `--kernel`: `rbf` or `none` (default: `rbf`).
- `--seed`: RNG seed (default: `0`).
- `--out`: output `.pkl` path (default: `R_matrix_[SIM].pkl`).

### Recommended settings

Because neither the distance metric nor the projection dimension has a measurable effect on selection quality, both can be fixed at their cheapest values:

| Hyper-parameter | Recommended | Note |
|---|---|---|
| Similarity metric | `EUC` | Cheapest; best or near-best in most configurations |
| Projection dimension `h` | `10` | Flat accuracy from `h = 1` to `h = 2048` |
| Selection rate | `0.4`–`0.5` | The only parameter worth tuning; 0.6–0.7 for pre-tuned architectures |

Expect the alternating optimisation to stabilise within 6–8 iterations.

## Output

The script generates a `.pkl` file named `R_matrix_[SIM].pkl`, where `[SIM]` is the similarity metric used. It contains the matrices `W` (shape `(d, h)`) and `R` (shape `(n, h)`) together with the optimisation details (`loss`, `n_iter`, `time`, `alpha`, `beta`, `h`). The per-patch irrelevance scores are the `ℓ2` norms of the **rows** of `R`; sorting them in ascending order and keeping the first `k` gives the selected subset.

## Insertion into downstream models

The selection is applied to the set of inputs that each downstream model consumes; the model itself is trained with its published architecture and settings, on the retained inputs only.

- For patch-based CNNs such as **SAP-Net** and **SAL-360IQA**, the selection is applied to the pool of patches the model samples from each image, and the model receives the `k` retained patches.
- For **MC360IQA**, whose branches each process one cubemap face, the selection is applied to the six faces together, since several projections are made per input image.
- For **Assessor360**, the selection is applied per exploration sequence: each sequence of `M` viewports is shortened to a common length `k' = ⌊rate × M⌋`, preserving the temporal order. This avoids padding and keeps the GRU input length uniform across sequences.

In all cases, no learned layer is modified.

## Datasets

The three benchmarks used in the paper are publicly available from their respective authors:

- **CVIQ** — 528 distorted images from 16 references (JPEG, AVC, HEVC), homogeneous compression distortions.
- **OIQA** — 320 images from 16 references (JPEG, JPEG 2000, Gaussian noise, Gaussian blur), moderate heterogeneity.
- **MVAQD** — 300 images from 15 references (JPEG, JPEG 2000, HEVC, blur, white noise), the most heterogeneous of the three.

## Citation

```bibtex
@article{sendjasni2025similarity,
  title   = {Similarity-preserving instance selection with residual-based outlier
             detection for data-efficient omnidirectional image quality assessment},
  author  = {Sendjasni, Abderrezzaq and Benkabou, Seif-Eddine and Larabi, Mohamed-Chaker},
  year    = {2025},
  note    = {Under review}
}
```

## Authors

- **Dr. Abderrezzaq Sendjasni**, XLIM, Univ. de Poitiers
- **Dr. Seif-Eddine Benkabou**, Univ. de Poitiers
- **Prof. Mohamed-Chaker Larabi**, XLIM, Univ. de Poitiers

## Acknowledgements

This work is partially funded by the Nouvelle-Aquitaine Research Council under project REALISME AAPR2022-2021-17027310.

## License

This project is licensed under the MIT License, see the [LICENSE](LICENSE) file for details.
