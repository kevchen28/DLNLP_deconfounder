import numpy as np
import scipy.io as sio
import scipy.sparse as sp

import torch
import torch.nn.functional as F


def normalize(mx):
    """
    Row-normalize sparse matrix.

    Args:
        mx (sp.csr_matrix): The sparse matrix to normalize.

    Returns:
        sp.csr_matrix: The row-normalized sparse matrix.
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(path, name="BlogCatalog", exp_id="0", original_X=False):
    """
    Load the BlogCatalog dataset from the given path.

    Args:
        path (str): The path to the dataset.
        name (str, optional): The folder containing the .mat files. Defaults to "BlogCatalog".
        exp_id (str, optional): The experiment ID. Defaults to "0".
        original_X (bool, optional): Whether to use the original X matrix. Defaults to False.

    Returns:
        tuple: A tuple containing the X matrix, the adjacency matrix, the treatment assignment matrix, the Y1 matrix, and the Y0 matrix.
    """
    data = sio.loadmat(path + name + "/BlogCatalog" + exp_id + ".mat")
    A = data["Network"]  # adjacency matrix

    # If original_X is False, use the X_100 matrix. Otherwise, use the Attributes matrix.
    if not original_X:
        X = data["X_100"]
    else:
        X = data["Attributes"]

    Y1 = data["Y1"]
    Y0 = data["Y0"]
    T = data["T"]

    return X, A, T, Y1, Y0


def wasserstein(x, y, p=0.5, lam=10, its=10):
    """
    Compute the Wasserstein distance between two distributions.

    Args:
        x (pytorch.Tensor): The first distribution (a batch of distributions)
        y (pytorch.Tensor): The second distribution (a batch of distributions)
        p (float, optional): Power of the Wasserstein distance. Defaults to 0.5.
        lam (int, optional): Strength of the entropy regularizer. Defaults to 10.
        its (int, optional): Number of Sinkhorn iterations. Defaults to 10.

    Returns:
        torch.Tensor: The Wasserstein distance between the two distributions.
        The Wasserstein distance between two distributions is defined as
        the minimum cost of transporting mass in order to transform one
        distribution into the other.
    """
    device = x.device  # Ensure device consistency.

    nx = x.shape[0]
    ny = y.shape[0]

    x = x.squeeze()
    y = y.squeeze()

    M = pdist(x, y)  # Compute the pairwise distance matrix

    # Estimate lambda and delta
    M_mean = torch.mean(M)  # Mean of the pairwise distance matrix
    M_drop = F.dropout(
        M, 10.0 / (nx * ny), training=False
    )  # Drop out some elements of the pairwise distance matrix
    delta = torch.max(M_drop).detach()  # Maximum of the pairwise distance matrix
    eff_lam = (lam / M_mean).detach()  # Effective lambda

    # Compute new distance matrix with augmented row and column for slack variable
    row = delta * torch.ones(M[0:1, :].shape, device=device)  # Augmented row
    col = torch.cat(
        [
            delta * torch.ones(M[:, 0:1].shape, device=device),
            torch.zeros((1, 1), device=device),
        ],
        0,
    )  # Augmented column
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    # Compute marginals
    a = torch.cat(
        [
            p * torch.ones((nx, 1), device=device) / nx,
            (1 - p) * torch.ones((1, 1), device=device),
        ],
        0,
    )  # Marginal a
    b = torch.cat(
        [
            (1 - p) * torch.ones((ny, 1), device=device) / ny,
            p * torch.ones((1, 1), device=device),
        ],
        0,
    )  # Marginal b

    # Compute kernel
    Mlam = eff_lam * Mt  # Modified distance matrix
    temp_term = (
        torch.ones(1, device=device) * 1e-6
    )  # Add a small term to the diagonal to ensure invertibility
    K = torch.exp(-Mlam) + temp_term  # Kernel

    u = torch.ones_like(a)  # Initialize u

    for _ in range(its):
        v = b / (K.t().matmul(u))  # Update v
        u = a / (K.matmul(v))  # Update u

    # Calculate the transport plan and distance
    upper_t = (
        u * (v.t() * K).detach()
    )  # Correct matrix multiplication for computing the transport plan
    E = upper_t * Mt
    D = 2 * torch.sum(E)  # Calculate the Wasserstein distance

    return D


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.0:  # Then we can use the norms to speed up the computation.
        norms_1 = torch.sum(
            sample_1**2, dim=1, keepdim=True
        )  # Squared norms of each row of sample 1
        norms_2 = torch.sum(
            sample_2**2, dim=1, keepdim=True
        )  # Squared norms of each row of sample 2
        norms = norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(
            n_1, n_2
        )  # Sum of the squared norms
        distances_squared = norms - 2 * sample_1.mm(
            sample_2.t()
        )  # Compute the pairwise squared distances
        return torch.sqrt(
            eps + torch.abs(distances_squared)
        )  # Return the pairwise Euclidean distances
    else:
        dim = sample_1.size(1)  # Dimension of the samples
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)  # Expand the samples
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)  # Expand the samples
        differences = (
            torch.abs(expanded_1 - expanded_2) ** norm
        )  # Compute the pairwise differences
        inner = torch.sum(differences, dim=2, keepdim=False)  # Sum the differences
        return (eps + inner) ** (1.0 / norm)  # Return the pairwise distances


# Function to compute the RBF kernel between two datasets
def rbf_kernel(x, y, gamma=None):
    """
    Function to compute the RBF kernel between two datasets.

    Args:
        x (torch.Tensor): The first distribution (a batch of distributions)
        y (torch.Tensor): The second distribution (a batch of distributions)
        gamma (float, optional): The gamma parameter of the RBF kernel. Defaults to None.

    Returns:
        torch.Tensor: The RBF kernel matrix between the two datasets.
    """
    # Compute pairwise squared Euclidean distances between x and y
    x_sq = torch.sum(x**2, dim=1, keepdim=True)
    y_sq = torch.sum(y**2, dim=1, keepdim=True)
    xy_sq = torch.mm(x, y.t())
    dist_sq = x_sq - 2 * xy_sq + y_sq.t()

    if gamma is None:
        gamma = 1 / (2 * torch.mean(dist_sq))

    return torch.exp(-gamma * dist_sq)


# Function to compute the Maximum Mean Discrepancy (MMD)
def mmd(x, y, kernel=rbf_kernel):
    """
    Function to compute the Maximum Mean Discrepancy (MMD).

    Args:
        x (torch.Tensor): The first distribution (a batch of distributions)
        y (torch.Tensor): The second distribution (a batch of distributions)
        kernel (function, optional): The kernel function to use. Defaults to rbf_kernel.

    Returns:
        torch.Tensor: The Maximum Mean Discrepancy (MMD) between the two distributions.
    """
    # Compute kernel matrices for x and y
    K_xx = kernel(x, x)
    K_yy = kernel(y, y)
    K_xy = kernel(x, y)

    # Get the sizes of x and y
    nx = x.size(0)
    ny = y.size(0)

    # Compute MMD
    mmd_val = (1 / (nx * (nx - 1))) * (torch.sum(K_xx) - torch.trace(K_xx))
    mmd_val += (1 / (ny * (ny - 1))) * (torch.sum(K_yy) - torch.trace(K_yy))
    mmd_val -= (2 / (nx * ny)) * torch.sum(K_xy)

    return mmd_val


def convert_sparse_matrix_to_edge_list(sparse_mx):
    """
    Convert a scipy.sparse matrix to torch_geometric edge list format.
    Args:
        sparse_mx (sp.csr_matrix): The sparse matrix to convert.
    Returns:
        edge_index (torch.LongTensor): The edge index tensor of shape [2, num_edges].
        edge_weight (torch.FloatTensor): The edge weights tensor of shape [num_edges].
    """
    sparse_mx = sparse_mx.tocoo()  # Convert to COOrdinate format
    row = torch.from_numpy(
        sparse_mx.row.astype(np.int64)
    )  # Convert row indices to tensor
    col = torch.from_numpy(
        sparse_mx.col.astype(np.int64)
    )  # Convert column indices to tensor
    edge_index = torch.stack(
        [row, col], dim=0
    )  # Stack row and column indices to create edge index tensor
    edge_weight = torch.from_numpy(
        sparse_mx.data.astype(np.float32)
    )  # Convert data to tensor

    return edge_index, edge_weight
