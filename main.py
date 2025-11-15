import numpy as np
import time
from Hybrid_algorithm import ovr_variant  # import your model
import pandas as pd


def f1_score_macro(y_true, y_pred):
    classes = np.unique(y_true)
    f1_scores = []

    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        f1_scores.append(f1)

    return np.mean(f1_scores)


# -------- MAIN PIPELINE -------- #

if __name__ == "__main__":
    start_time = time.time()

    print("Loading MNIST dataset...")
    train_data = pd.read_csv("MNIST_train.csv")
    val_data   = pd.read_csv("MNIST_validation.csv")
    y_train = train_data['label']
    x_train = train_data.drop(columns=['label','even'])
    y_val   = val_data['label']
    x_val   = val_data.drop(columns=['label','even'])


    print("Training Hybrid OvR Model...")
    model = ovr_variant(
        n_estimators=40,
        lamda=3,
        learning_rate=0.3,
        max_depth=6,
        subsample_features=1/28,
        knn_k=5
    )

    model.fit(x_train, y_train)

    print("Predicting on validation set...")
    predictions = model.predict(x_val, epsilon=0.4)

    print("Evaluating F1 score...")
    f1 = f1_score_macro(y_val, predictions)

    print(f"\nFinal Macro F1 Score: {f1:.4f}")
    print(f"Total Runtime: {time.time() - start_time:.2f} seconds")

