from torch import Tensor


def accuracy(predictions: Tensor, targets: Tensor) -> float:
    """Return the fraction of correctly classified samples."""
    if targets.numel() == 0:
        raise ValueError("targets must not be empty")
    return (predictions.argmax(dim=1) == targets).float().mean().item()


def macro_f1(predictions: Tensor, targets: Tensor, num_classes: int) -> float:
    """Return macro-averaged F1 over the supplied class range."""
    if targets.numel() == 0:
        raise ValueError("targets must not be empty")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    predicted_classes = predictions.argmax(dim=1)
    scores: list[float] = []
    for class_index in range(num_classes):
        predicted_positive = predicted_classes == class_index
        target_positive = targets == class_index
        true_positive = (predicted_positive & target_positive).sum().item()
        false_positive = (predicted_positive & ~target_positive).sum().item()
        false_negative = (~predicted_positive & target_positive).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / num_classes
