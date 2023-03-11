import argparse
import json
from pathlib import Path

import torch
from ray import tune

from fstream import io
from refine import refine


def main():
    args = parse_arguments()

    data = args.data
    output = args.output

    if not output:
        output = data / "REFINE_results"
        output.mkdir(parents=True, exist_ok=True)

    no_cuda = args.no_cuda

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    )

    regularization = args.regularization
    learning_rate = args.learning_rate
    epochs = args.epochs
    shape_ratio = args.shape_ratio
    r_ratio = args.r_ratio
    batch_size = args.batch_size

    search_space = {
        "regularization_constant": tune.grid_search(regularization),
        "learning_rate": tune.grid_search(learning_rate),
        "epochs": tune.grid_search(epochs),
        "layer_shape_ratio": tune.grid_search(shape_ratio),
        "r_ratio": tune.grid_search(r_ratio),
        "batch_size": tune.grid_search(batch_size),
    }

    objective = objective_generator(data, device)

    tuner = tune.Tuner(objective, param_space=search_space)
    results = tuner.fit().get_best_result(metric="f1", mode="max").metrics

    predicted_scores = results["predicted"]
    del results["predicted"]
    io.write_scores(predicted_scores, output)
    with open(output / "results.json", "w") as f:
        json.dump(str(results), f, indent=4)


def objective_generator(dataset, device):
    cascades_matrix = io.read_cascades(dataset)
    observed_structure = io.read_structure(dataset, observed=True)
    ground_truth_structure = io.read_structure(dataset, observed=False)
    n, m = cascades_matrix.shape

    def objective(config):
        regularization_constant = config["regularization_constant"]
        learning_rate = config["learning_rate"]
        epochs = config["epochs"]
        layer_shape_ratio = config["layer_shape_ratio"]
        r_ratio = config["r_ratio"]
        batch_size = config["batch_size"]
        layers_size = [
            int(r_ratio * n),
            int(r_ratio * n * layer_shape_ratio),
            int(r_ratio * n * layer_shape_ratio * layer_shape_ratio),
        ]
        r = int(r_ratio * n)
        return refine.refine(
            cascades_matrix,
            observed_structure,
            ground_truth_structure,
            r,
            layers_size,
            regularization_constant,
            learning_rate,
            epochs,
            batch_size,
            device,
        )

    return objective


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        help="Path to the dataset with matrices",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the output file, images, results",
    )
    parser.add_argument(
        "-n",
        "--no_cuda",
        action="store_true",
        help="Disable CUDA for training auto encoder",
    )
    parser.add_argument(
        "-g",
        "--regularization",
        type=float,
        nargs="+",
        help="Regularization constant for auto encoder",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        nargs="+",
        help="Learning rate of training auto encoder",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        nargs="+",
        help="Epochs num for training",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--shape_ratio",
        type=float,
        nargs="+",
        help="Layer size reduction",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--r_ratio",
        type=float,
        nargs="+",
        help="Dimension reduction ratio",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="+",
        help="Batch size of SGD",
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
