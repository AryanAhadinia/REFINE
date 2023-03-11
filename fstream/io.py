from pathlib import Path

import pandas as pd


def _read_matrix(path):
    return pd.read_csv(path, header=None, index_col=False).to_numpy()


def _write_matrix(matrix, path):
    pd.DataFrame(matrix).to_csv(path, header=None, index=False)


def read_structure(dataset_path: Path, observed=True):
    matrix_path = (
        dataset_path / "MatrixG.txt" if observed else dataset_path / "MatrixGASL.txt"
    )
    return _read_matrix(matrix_path)


def read_cascades(dataset_path: Path, observed=True):
    matrix_path = (
        dataset_path / "MatrixC.txt" if observed else dataset_path / "MatrixCASL.txt"
    )
    return _read_matrix(matrix_path)


def write_scores(structure, dataset_path: Path):
    _write_matrix(structure, dataset_path / "scores.csv")
