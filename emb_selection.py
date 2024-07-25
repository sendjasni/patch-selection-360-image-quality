import numpy as np
import time
import scipy.io as sio
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import pickle
import argparse


def run(X, Z, lamda, beta, maxIt):
    """
    Perform optimization to find W and B matrices.

    Args:
        X (np.ndarray): Input data matrix.
        Z (np.ndarray): Reduced space matrix.
        lamda (float): Regularization parameter for W.
        beta (float): Regularization parameter for B.
        maxIt (int): Maximum number of iterations.

    Returns:
        dict: Results including W, B, loss, and time taken.
    """
    n, m = X.shape
    h = Z.shape[1]

    # Initialization
    W = np.random.rand(m, h)
    D_w = np.eye(m)
    B = np.random.rand(h, n)
    D_b = np.eye(n)
    Iw = np.eye(m)
    Ib = np.eye(n)
    epsIt = 1e-3
    loss = np.zeros(maxIt)
    start_time = time.time()

    for t in range(maxIt):
        # Update W and B
        W = np.linalg.inv(X.T @ X + lamda * D_w + epsIt * np.eye(m)) @ (X.T @ (B.T + Z))
        B = (X @ W - Z).T @ np.linalg.inv(Ib + beta * D_b + epsIt * np.eye(n)).T

        # Update diagonal matrices D_w and D_b
        D_w = np.diag(0.5 * np.linalg.norm(W, axis=1)) + epsIt
        D_b = np.diag(0.5 * np.linalg.norm(B.T, axis=1)) + epsIt

        # Compute the L21 norms
        l_21_W = np.sum(np.linalg.norm(W, axis=0))
        l_21_B = np.sum(np.linalg.norm(B, axis=0))

    
        # Compute the Loss
        loss[t] = np.linalg.norm(X@W-B.T-Z, 'fro')**2 + lamda * l_21_W + beta*l_21_B
        if t >= 10 and (loss[t] - loss[t-1]) <= 1e-4:
            break
        
    elapsed_time = time.time() - start_time
    return {'W': W, 'B': B, 'loss': loss, 'time': elapsed_time, 'maxIt': maxIt, 'lambda': lamda, 'beta': beta, 'epsIt': epsIt}


def process_image(X, sim, h):
    """
    Process an image by computing distances, RBF kernel, and running optimization.

    Args:
        X (np.ndarray): Input data matrix.
        sim (str): Similarity metric to use ('MAN', 'MAH', or 'EU').
        h (int): Number of components for dimensionality reduction.

    Returns:
        dict: Results from the optimization.
    """
    print(f'[INFO] Computing distances for similarity: {sim} ...')

    # Compute distance matrix
    if sim == 'MAN':
        distances = cdist(X, X, metric='cityblock')
    elif sim == 'MAH':
        cov_matrix = np.cov(X, rowvar=False)
        cov_matrix += 1e-10 * np.eye(cov_matrix.shape[0])
        distances = cdist(X, X, metric='mahalanobis', VI=cov_matrix)
    else:  # Default to Euclidean distance
        distances = cdist(X, X, metric='euclidean')

    # Compute RBF kernel
    gamma = 1.0 / (2.0 * np.mean(distances))
    K = np.exp(-gamma * distances ** 2)

    # Eigen decomposition
    _, V = eigh(K)
    V = np.flip(V, axis=0)

    # Reduce dimensions
    Z = V[:, :h]

    # Run optimization
    results = run(X, Z, lamda=0.1, beta=0.1, maxIt=50)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process patches features and compute B matrix.')
    parser.add_argument("-sim", "--sim", choices=['MAN', 'MAH', 'EU'], required=True, help="Similarity distance.")
    parser.add_argument("-mat", "--mat", required=True, help="Path to the feature file.")

    args = parser.parse_args()

    sim = args.sim
    mat = args.mat

    features = sio.loadmat(mat)['data']
    features = np.array(features)
    dim = features[0].shape[0]

    print(f'[INFO] Features shape: {features.shape} | Dim is: {dim}')

    step = 180  # Number of instances per image
    image_args = [(features[i:i + step], sim, 10)  # Reduced dimension h = 10 as example
                  for i in range(0, len(features), step)]

    results = [process_image(X, sim, h) for X, _, h in image_args]

    keys = ['img_' + str(i) for i in range(1, len(results) + 1)]
    B_dict = dict(zip(keys, results))

    with open(f'B_matrix_{sim}.pkl', 'wb') as fp:
        pickle.dump(B_dict, fp)
        print(f'Dictionary for {sim} distance saved successfully to file')
