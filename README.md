This repository contains the code associated with the article: *A Two-Fold Patch Selection Approach for Improved 360-Degree Image Quality Assessment*.

The project involves computing similarity metrics between embeddings of patches generated from 360-degree images. The code performs dimensionality reduction and optimization on input data, represented as an `hxw` shape, and provides results based on different distance metrics.

**Note:** The patch sampling and quality regression processes can be delivered on demand.

## Features

- **Distance Metrics**: Computes distances using Manhattan, Mahalanobis, or Euclidean metrics.
- **Dimensionality Reduction**: Uses RBF kernel and eigen decomposition for dimensionality reduction.
- **Optimization**: Finds optimal matrices \( W \) and \( B \) through iterative optimization.

## Installation

Ensure you have the following Python packages installed:
- `numpy`
- `scipy`
- `argparse`
- `pickle`

## Usage

To run the script, use the following command:

```bash
python emb_selection.py -sim [SIMILARITY_METRIC] -mat [PATH_TO_FEATURE_FILE]
```

### Arguments

- `-sim`: Similarity distance metric. Choose from `MAN` (Manhattan), `MAH` (Mahalanobis), or `EU` (Euclidean). (Required)
- `-mat`: Path to the `.mat` file containing the feature data. (Required)

## Results

The script will generate a `.pkl` file named `B_matrix_[SIM].pkl`, where `[SIM]` is the similarity metric used. This file contains the results of the optimization, including matrices \( W \), \( B \), and other details.

## Authors

- **Dr. Abderrezzaq Sendjasni**, XLIM, Univ. de Poitiers
- **Dr. Seif-eddine Benkebbou**, LIAS, ENSMA de Poitiers
- **Prof. Mohamed-Chaker Larabi**, XLIM, Univ. de Poitiers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
