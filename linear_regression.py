
"""Linear Regression.py
This file implements a simple linear regression model using numpy.
It includes methods for training the model, making predictions, and calculating loss.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegression:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # Gradient for a single weight
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    def get_loss(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        return np.mean(np.square(ground_truth - model_prediction))

    learning_rate = 0.01

    def train_model(
        self,
        X_train: NDArray[np.float64],
        Y_train: NDArray[np.float64],
        X_val: NDArray[np.float64],
        Y_val: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        N = len(X_train)
        weights = initial_weights.copy()
        for i in range(num_iterations):
            pred_train = self.get_model_prediction(X_train, weights)
            loss_train = self.get_loss(pred_train, Y_train)

            # validation loss every 100 iterations
            if i % 100 == 0:
                pred_val = self.get_model_prediction(X_val, weights)
                loss_val = self.get_loss(pred_val, Y_val)
                print(f"Iteration {i}, Train Loss: {loss_train:.6f}, Val Loss: {loss_val:.6f}")

            for j in range(len(weights)):
                grad = self.get_derivative(pred_train, Y_train, N, X_train, j)
                weights[j] -= self.learning_rate * grad

        return np.round(weights, 5)


if __name__ == "__main__":
    # ----- Step 1: Create dummy data -----
    np.random.seed(42)
    N = 1000
    X_raw = np.random.rand(N, 2)  # 2 features
    true_weights = np.array([4.0, -3.0, 2.0])  # bias + w1 + w2

    # Add bias column of ones to X
    X = np.hstack([np.ones((N, 1)), X_raw])
    Y = X @ true_weights + np.random.randn(N) * 0.1  # noisy target

    # ----- Step 2: Split data -----
    indices = np.arange(N)
    np.random.shuffle(indices)

    train_end = int(0.7 * N)
    val_end = int(0.85 * N)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # ----- Step 3: Train model -----
    initial_weights = np.zeros(3)
    model = LinearRegression()
    learned_weights = model.train_model(
        X_train, Y_train,
        X_val, Y_val,
        num_iterations=2000,
        initial_weights=initial_weights
    )

    # ----- Step 4: Evaluate on test data -----
    pred_test = model.get_model_prediction(X_test, learned_weights)
    test_loss = model.get_loss(pred_test, Y_test)

    print("True weights:    ", np.round(true_weights, 5))
    print("Learned weights: ", learned_weights)
    print(f"Test Loss: {test_loss:.6f}")

    print("---Metrics---")
    print("MSE:", mean_squared_error(Y_test, pred_test))
    print("R2:", r2_score(Y_test, pred_test))