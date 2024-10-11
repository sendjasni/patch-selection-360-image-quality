import numpy as np
import time
import scipy.io as sio
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import pickle
import argparse


def run(E, Z, lamda, beta, maxIt):
    """
    Perform optimization to find W and R matrices.

    Args:
        E (np.ndarray): Input data matrix (Embeddings).
        Z (np.ndarray): Reduced space matrix.
        lamda (float): Regularization parameter for W.
        beta (float): Regularization parameter for R.
        maxIt (int): Maximum number of iterations.

    Returns:
        dict: Results including W, R, loss, and time taken.
    """
    n, m = E.shape
    h = Z.shape[1]

    # Initialization
    W = np.random.rand(m, h)
    D_w = np.eye(m)
    R = np.random.rand(h, n)
    D_r = np.eye(n)
    Iw = np.eye(m)
    Ib = np.eye(n)
    epsIt = 1e-3
    loss = np.zeros(maxIt)
    start_time = time.time()

    for t in range(maxIt):
        # Update W and B
        W = np.linalg.inv(E.T @ E + lamda * D_w + epsIt * np.eye(m)) @ (E.T @ (R.T + Z)) # Eq. 18
        R = (E @ W - Z).T @ np.linalg.inv(Ib + beta * D_r + epsIt * np.eye(n)).T  # Eq. 21

        # Update diagonal matrices D_w and D_r
        D_w = np.diag(0.5 * np.linalg.norm(W, axis=1)) + epsIt
        D_r = np.diag(0.5 * np.linalg.norm(R.T, axis=1)) + epsIt

        # Compute the L21 norms
        l_21_W = np.sum(np.linalg.norm(W, axis=0))
        l_21_R = np.sum(np.linalg.norm(R, axis=0))

    
        # Compute the Loss
        loss[t] = np.linalg.norm(E@W-R.T-Z, 'fro')**2 + lamda * l_21_W + beta*l_21_R # Eq. 15
        if t >= 10 and (loss[t] - loss[t-1]) <= 1e-4:
            break
        
    elapsed_time = time.time() - start_time
    return {'W': W, 'R': R, 'loss': loss, 'time': elapsed_time, 'maxIt': maxIt, 'lambda': lamda, 'beta': beta, 'epsIt': epsIt}


def process_image(E, sim, h):
    """
    Process an image by computing distances, RBF kernel, and running optimization.

    Args:
        E (np.ndarray): Input data matrix (Embeddings).
        sim (str): Similarity metric to use ('MAN', 'MAH', or 'EU').
        h (int): Number of components for dimensionality reduction.

    Returns:
        dict: Results from the optimization.
    """
    print(f'[INFO] Computing distances for similarity: {sim} ...')

    # Compute distance matrix
    if sim == 'MAN':
        distances = cdist(E, E, metric='cityblock')

    elif sim == 'MAH':
        cov_matrix = np.cov(E, rowvar=False)
        cov_matrix += 1e-10 * np.eye(cov_matrix.shape[0])
        distances = cdist(E, E, metric='mahalanobis', VI=cov_matrix)
    
    else:  # Default to Euclidean distance
        distances = cdist(E, E, metric='euclidean')

    # Compute RBF kernel
    gamma = 1.0 / (2.0 * np.mean(distances))
    K = np.exp(-gamma * distances ** 2)

    # Eigen decomposition
    _, V = eigh(K)
    V = np.flip(V, axis=0)

    # Reduce dimensions
    Z = V[:, :h]

    # Run optimization
    results = run(E, Z, lamda=0.1, beta=0.1, maxIt=50)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process patches features (Embeddings) and compute R matrix.')
    parser.add_argument("-sim", "--sim", choices=['MAN', 'MAH', 'EUC'], required=True, help="Similarity distance.")
    parser.add_argument("-mat", "--mat", required=True, help="Path to the feature file.")

    args = parser.parse_args()

    sim = args.sim
    mat = args.mat

    embeddings = sio.loadmat(mat)['data']
    embeddings = np.array(embeddings)
    dim = embeddings[0].shape[0]

    print(f'[INFO] Embeddings shape: {embeddings.shape} | Dim is: {dim}')

    step = 180  # Number of instances per image
    image_args = [(embeddings[i:i + step], sim, 10)  # Reduced dimension h = 10 as example
                  for i in range(0, len(embeddings), step)]

    results = [process_image(E, sim, h) for E, _, h in image_args]

    keys = ['img_' + str(i) for i in range(1, len(results) + 1)]
    R_dict = dict(zip(keys, results))

    with open(f'R_matrix_{sim}.pkl', 'wb') as fp:
        pickle.dump(R_dict, fp)
        print(f'Dictionary for {sim} distance saved successfully to file')
