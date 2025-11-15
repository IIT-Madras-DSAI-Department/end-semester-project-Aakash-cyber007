import numpy as np
import time
from Hybrid_algorithm import ovr_variant  # import your model
import pandas as pd

def load_mnist_csv(file_path):
    data = pd.read_csv(file_path)
    y_train = data['label']
    x_train = data.drop(columns=['label','even'])
    x_train = x_train.values / 255.0  # Normalize pixel values
    return x_train, y_train
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


if __name__ == "__main__":
    start_time = time.time()

    print("Loading MNIST dataset...")
    x_train, y_train= load_mnist_csv("MNIST_train.csv")
    x_val, y_val = load_mnist_csv("MNIST_validation.csv")


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


