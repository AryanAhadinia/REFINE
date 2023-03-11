from evaluate import evaluate
from fstream import io

from .data_prepare import interaction_pattern
from .auto_encoder import AutoEncoderTrainer


def refine(
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
    results_path=None,
):
    interaction_matrix = interaction_pattern(cascades_matrix, r)
    trainer = AutoEncoderTrainer(
        interaction_matrix,
        layers_size,
        learning_rate,
        regularization_constant,
        batch_size,
        device,
    )
    trainer.train(epochs)
    if results_path:
        trainer.plot_losses(results_path)
    scores = trainer.scores()
    if results_path:
        io.write_scores(scores, results_path)
    evaluation = evaluate.evaluate(ground_truth_structure, observed_structure, scores)
    return evaluation
