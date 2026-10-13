"""
Similarity-preserving instance selection with residual-based outlier detection.

Implements Algorithm 1: given patch embeddings E, learn a similarity-preserving
transformation W and an l2,1-penalised residual R by alternating minimisation,
then rank patches by the l2 norm of the rows of R (ascending = most relevant).

Objective (Eq. 9 of the manuscript):
    min_{W,R}  ||E W - Z - R||_F^2 + alpha ||W||_{2,1} + beta ||R||_{2,1}

Convention (aligned with the manuscript):
    E : (n, d)   -- n patches, d-dimensional embeddings
    Z : (n, h)   -- rank-h spectral factor of the affinity matrix S
    W : (d, h)   -- similarity-preserving projection
    R : (n, h)   -- residual, one row per patch; row j is the residual of patch j
"""

import argparse
import pickle
import time

import numpy as np
import scipy.io as sio
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

EPS = 1e-8       # floor for the reweighting denominators (epsilon in Eq. 12)
RIDGE = 1e-6     # ridge term (rho) for numerical stability of the linear solves


def _reweight(norms):
    """Diagonal entries of the l2,1 reweighting matrix: d_jj = 1 / (2 ||m_j|| + eps)."""
    return 1.0 / (2.0 * norms + EPS)


def run(E, Z, alpha, beta, max_it=50, tol=1e-6):
    """
    Alternating minimisation of Eq. 9 (manuscript).

    Args:
        E (np.ndarray): (n, d) embedding matrix for one image.
        Z (np.ndarray): (n, h) optimal low-dimensional target, S ~ Z Z^T.
        alpha (float): regularisation weight on W (row sparsity over features).
        beta (float): regularisation weight on R (row sparsity over samples).
        max_it (int): maximum number of alternating iterations.
        tol (float): relative decrease below which the loop stops.

    Returns:
        dict: W, R, per-iteration loss, wall-clock time and the settings used.
    """
    n, d = E.shape
    h = Z.shape[1]

    # Algorithm 1, line 5: R starts at zero; the reweighting matrices start at I
    # so that the first updates are plain ridge solves.
    R = np.zeros((n, h))                     # (n, h): one row per patch
    d_w = np.ones(d)
    d_r = np.ones(n)

    I_d = np.eye(d)
    I_n = np.eye(n)
    EtE = E.T @ E                            # fixed per image
    loss = []
    start_time = time.time()

    for t in range(max_it):
        # --- Step 1: update W with R fixed (Eq. 13 of the manuscript) ------
        # Direct form: (E^T E + A) W = E^T (Z + R), with A = alpha D_w + rho I.
        A = EtE + alpha * np.diag(d_w) + RIDGE * I_d
        W = np.linalg.solve(A, E.T @ (Z + R))              # (d, h)

        # --- Step 2: update R with W fixed (Eq. 14 of the manuscript) ------
        # R = B^{-1} (E W - Z) with B = (1 + rho) I + beta D_r.
        B = (1.0 + RIDGE) * I_n + beta * np.diag(d_r)
        R = np.linalg.solve(B, (E @ W - Z))                # (n, h)

        # --- Reweighting for the next iteration ----------------------------
        w_row_norms = np.linalg.norm(W, axis=1)   # rows of W -> feature directions
        r_row_norms = np.linalg.norm(R, axis=1)   # rows of R -> samples (patches)
        d_w = _reweight(w_row_norms)
        d_r = _reweight(r_row_norms)

        # --- Objective (Eq. 9), consistent with the penalties above --------
        l21_W = w_row_norms.sum()
        l21_R = r_row_norms.sum()
        fit = np.linalg.norm(E @ W - Z - R, 'fro') ** 2
        loss.append(fit + alpha * l21_W + beta * l21_R)

        if t >= 1:
            decrease = loss[-2] - loss[-1]
            if abs(decrease) <= tol * max(1.0, abs(loss[-2])):
                break

    return {
        'W': W,
        'R': R,
        'loss': np.asarray(loss),
        'n_iter': len(loss),
        'time': time.time() - start_time,
        'alpha': alpha,
        'beta': beta,
        'h': h,
    }


def similarity_matrix(E, sim, kernel='rbf'):
    """
    Build the similarity matrix S from the embeddings.

    Note on MAH: the manuscript (Section 3.4.2) defines a Mahalanobis-type
    quadratic form that uses the empirical covariance directly, not its inverse.
    scipy's cdist uses the VI argument as-is in the quadratic form
    (x-y)^T VI (x-y), so passing the covariance reproduces the manuscript's
    convention. Passing np.linalg.inv(cov) would instead give the classical
    Mahalanobis distance.
    """
    if sim == 'MAN':
        distances = cdist(E, E, metric='cityblock')
    elif sim == 'MAH':
        cov = np.cov(E, rowvar=False)
        cov += 1e-10 * np.eye(cov.shape[0])
        distances = cdist(E, E, metric='mahalanobis', VI=cov)
    else:  # EUC
        distances = cdist(E, E, metric='euclidean')

    if kernel == 'rbf':
        gamma = 1.0 / (2.0 * np.mean(distances) + EPS)
        S = np.exp(-gamma * distances ** 2)
    else:
        S = -distances
        S = S - S.min()

    return 0.5 * (S + S.T)   # enforce exact symmetry before eigh


def low_rank_target(S, h):
    """
    Z = U_h Lambda_h^{1/2} from the top-h eigenpairs of S, so that S ~ Z Z^T
    (Section 3.4.2). eigh returns eigenvalues in ascending order, so the top-h
    are the LAST h columns; we reverse the order to get a descending spectrum.
    """
    vals, vecs = eigh(S)
    h_eff = min(h, len(vals))                      # guard when h > n
    idx = np.argsort(vals)[::-1][:h_eff]
    top_vals = np.clip(vals[idx], 0.0, None)
    return vecs[:, idx] * np.sqrt(top_vals)


def select_patches(R, rate):
    """
    Algorithm 1, line 9: rank patches by the l2 norm of the ROWS of R in
    ascending order and keep the most relevant fraction.

    Aligned with the manuscript convention: R is (n, h), one row per patch.

    Returns:
        (indices of the kept patches, irrelevance score of every patch)
    """
    scores = np.linalg.norm(R, axis=1)             # row norms -> (n,)
    order = np.argsort(scores)
    k = max(1, int(round(rate * len(scores))))
    return order[:k], scores


def process_image(E, sim, h, alpha, beta, max_it, tol, kernel):
    """Full pipeline for the embeddings of a single image."""
    S = similarity_matrix(E, sim, kernel=kernel)
    Z = low_rank_target(S, h)
    return run(E, Z, alpha=alpha, beta=beta, max_it=max_it, tol=tol)


def split_by_image(embeddings, step, counts_file=None):
    """
    Split the stacked embedding matrix into one block per image.

    A fixed `step` assumes every image contributed the same number of patches,
    which does not hold for the LAT and SP samplers. Pass a .npy/.txt file of
    per-image patch counts via --counts when they vary.
    """
    if counts_file is not None:
        counts = np.loadtxt(counts_file, dtype=int).ravel()
        if counts.sum() != len(embeddings):
            raise ValueError(
                f'Patch counts sum to {counts.sum()} but {len(embeddings)} '
                'embeddings were loaded.'
            )
        bounds = np.cumsum(counts)[:-1]
        return np.split(embeddings, bounds)

    if len(embeddings) % step != 0:
        raise ValueError(
            f'{len(embeddings)} embeddings is not a multiple of step={step}; '
            'pass --counts with the per-image patch counts instead.'
        )
    return [embeddings[i:i + step] for i in range(0, len(embeddings), step)]


def main():
    parser = argparse.ArgumentParser(
        description='Similarity-preserving instance selection for 360-degree IQA.')
    parser.add_argument('-sim', '--sim', choices=['MAN', 'MAH', 'EUC'], required=True,
                        help='Similarity distance metric.')
    parser.add_argument('-mat', '--mat', required=True,
                        help='Path to the .mat file containing the embeddings.')
    parser.add_argument('--h', type=int, default=10,
                        help='Projection dimension (default: 10).')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='l2,1 weight on W (default: 0.1).')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='l2,1 weight on R; larger = sparser residual (default: 0.1).')
    parser.add_argument('--rate', type=float, default=0.5,
                        help='Selection rate, fraction of patches kept (default: 0.5).')
    parser.add_argument('--step', type=int, default=180,
                        help='Patches per image when this is constant (default: 180).')
    parser.add_argument('--counts', default=None,
                        help='File of per-image patch counts, for variable-size pools.')
    parser.add_argument('--max-it', type=int, default=50,
                        help='Maximum alternating iterations (default: 50).')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='Relative convergence tolerance (default: 1e-6).')
    parser.add_argument('--kernel', choices=['rbf', 'none'], default='rbf',
                        help='Kernelise the distance matrix (default: rbf).')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed (default: 0).')
    parser.add_argument('--out', default=None,
                        help='Output .pkl path (default: R_matrix_[SIM].pkl).')
    args = parser.parse_args()

    np.random.seed(args.seed)

    embeddings = np.asarray(sio.loadmat(args.mat)['data'])
    print(f'[INFO] Embeddings shape: {embeddings.shape} | dim: {embeddings.shape[1]}')

    blocks = split_by_image(embeddings, args.step, args.counts)
    print(f'[INFO] {len(blocks)} image(s) | metric: {args.sim} | h: {args.h}')

    results = {}
    for i, E in enumerate(blocks, start=1):
        res = process_image(E, args.sim, args.h, args.alpha, args.beta,
                            args.max_it, args.tol, args.kernel)
        kept, scores = select_patches(res['R'], args.rate)
        res['selected_idx'] = kept
        res['irrelevance'] = scores
        results[f'img_{i}'] = res
        print(f'[INFO] img_{i}: {res["n_iter"]} iters, '
              f'{res["time"]:.2f}s, kept {len(kept)}/{len(scores)} patches')

    out = args.out or f'R_matrix_{args.sim}.pkl'
    with open(out, 'wb') as fp:
        pickle.dump(results, fp)
    print(f'[INFO] Saved to {out}')


if __name__ == '__main__':
    main()
