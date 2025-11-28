def compute_metrics(p):
    """
    Takes as input the predictions and gold labels and calculates evaluation metrics 
    (precision, recall, F1 score, and accuracy) for classification. Special tokens 
    (with label -100) are ignored.

    Parameters:
    p (tuple): A tuple containing:
    - 'predictions': model predictions for each token
    - 'labels': gold labels for each token, with special tokens labeled as -100

    Returns:
    A dictionary with: precision, recall, f-score, and accuracy
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)    # normalizes output in 0 and 1

    # removing ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }