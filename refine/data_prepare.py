import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def _co_occurrence(cascades_matrix):
    n, _ = cascades_matrix.shape
    co_occurrence_matrix = np.zeros((n, n))
    for u in range(n):
        for v in range(u + 1, n):
            co_occurrence_matrix[u, v] = co_occurrence_matrix[v, u] = np.sum(
                np.logical_and(cascades_matrix[u, :], cascades_matrix[v, :])
            )
    for u in range(n):
        normalizer_term = np.sum(cascades_matrix[u, :])
        co_occurrence_matrix[u, :] /= normalizer_term if normalizer_term != 0 else 1
    return co_occurrence_matrix


def _reaction_time_matrix(cascades_times):
    n, m = cascades_times.shape
    r = np.zeros((m, n, n))
    for c in range(m):
        cascade_times = cascades_times[:, c]
        matrix_of_differences = np.abs(np.subtract.outer(cascade_times, cascade_times))
        mask = cascade_times == 0
        matrix_of_differences[mask, :] = np.inf
        matrix_of_differences[:, mask] = np.inf
        r[c, :, :] = matrix_of_differences
    r_inv = np.exp(-r)
    aggregated = np.sum(r_inv, axis=0)
    normalizer_term = np.sum(aggregated, axis=1)[:, np.newaxis]
    normalizer_term[normalizer_term == 0] = 1
    aggregated /= normalizer_term
    return aggregated


def _dimension_reduction(X, r):
    X = csr_matrix(X)
    svd = TruncatedSVD(n_components=r, random_state=42)
    X = svd.fit_transform(X)
    return X


def interaction_pattern(cascades_times, r):
    reaction_time = _reaction_time_matrix(cascades_times)
    co_occurrence = _co_occurrence(cascades_times)
    I = np.multiply(co_occurrence, reaction_time)
    I_r = _dimension_reduction(I, r)
    return I_r
