from collections import Counter

import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )

    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1]
    # You can try other normalization methods, e.g., z-score, etc.
    # TODO: 1%
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def normalize_zscore(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1]
    # You can try other normalization methods, e.g., z-score, etc.
    # TODO: 1%
    return (X - X.mean(axis=0)) / X.std(axis=0)


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    unique, y_encoded = np.unique(y, return_inverse=True)
    return y_encoded


# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.insert(X, 0, 1, axis=1)
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        if self.model_type == "logistic":
            self.w = np.zeros((n_features, n_classes))
            for _ in range(self.iterations):
                self.w -= self.learning_rate * self._compute_gradients(X, y)
        else:
            self.w = np.zeros(n_features)
            for _ in range(self.iterations):
                self.w -= self.learning_rate * self._compute_gradients(X, y)
        # TODO: 2%

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        if self.model_type == "linear":
            # TODO: 2%
            return X @ self.w
        elif self.model_type == "logistic":
            # TODO: 2%
            return np.argmax(self._softmax(X @ self.w), axis=1)

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        if self.model_type == "linear":
            # TODO: 2%
            return 1 / n_samples * (X.T @ (X @ self.w - y))
        elif self.model_type == "logistic":
            # TODO: 2%
            Y_encoded = np.zeros((n_samples, n_classes), dtype=int)
            for i in range(n_samples):
                Y_encoded[i, y[i]] = 1

            return -1 / n_samples * (X.T @ (Y_encoded - self._softmax(X @ self.w)))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree:
    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:

        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        left_child = self._build_tree(X[X[:, feature] <= threshold], y[X[:, feature] <= threshold], depth + 1)
        right_child = self._build_tree(X[X[:, feature] > threshold], y[X[:, feature] > threshold], depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):

        if self.model_type == "classifier":
            return np.argmax(np.bincount(y))
        else:
            # TODO: 1%
            return y.mean()

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = y[mask], y[~mask]

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 2%
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right
        gini_left = 1 - np.sum((np.bincount(left_y) / n_left) ** 2)
        gini_right = 1 - np.sum((np.bincount(right_y) / n_right) ** 2)

        return n_left / n_total * gini_left + n_right / n_total * gini_right


    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 2%
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right
        mse_left = np.mean((left_y - left_y.mean()) ** 2)
        mse_right = np.mean((right_y - right_y.mean()) ** 2)
        return n_left / n_total * mse_left + n_right / n_total * mse_right


    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        model_type: str = "classifier",
    ):
        # TODO: 1%
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_type = model_type
        assert model_type in [
            "classifier",
            "regressor"
        ], "model_type must be either 'classifier' or 'regressor'"


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.trees = []
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            # TODO: 2%
            tree = DecisionTree(max_depth=self.max_depth, model_type=self.model_type)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        n_samples = X.shape[0]
        if self.model_type == "classifier":
            predictions = []
            for tree in self.trees:
                predictions.append(tree.predict(X))
            
            matrix = np.array(predictions).T
            res = np.zeros(n_samples)
            for i in range(n_samples):
                values, counts = np.unique(matrix[i], return_counts=True)
                res[i] = values[np.argmax(counts)]
            return res
        else:
            y_pred = np.array([tree.predict(X) for tree in self.trees])
            return np.mean(y_pred, axis=0)


# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    return np.mean((y_true - y_pred) ** 2)


# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    
    logistic_regression = LinearModel(model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print('---Logistic Regression---')
    print("In-sample Accuracy:", accuracy(y_train, logistic_regression.predict(X_train)))
    print("Out-sample Accuracy:", accuracy(y_test, y_pred))
    
    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print('---Decision Tree Classifier---')
    print("In-sample Accuracy:", accuracy(y_train, decision_tree_classifier.predict(X_train)))
    print("Out-sample Accuracy:", accuracy(y_test, y_pred))
    
    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print('---Random Forest Classifier---')
    print("In-sample Accuracy:", accuracy(y_train, random_forest_classifier.predict(X_train)))
    print("Out-sample Accuracy:", accuracy(y_test, y_pred))
    

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)

    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print('---Linear Regression---')
    print("In-sample MSE:", mean_squared_error(y_train, linear_regression.predict(X_train)))
    print("Out-sample MSE:", mean_squared_error(y_test, y_pred))
    
    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print('---Decision Tree Regressor---')
    print("In-sample MSE:", mean_squared_error(y_train, decision_tree_regressor.predict(X_train)))
    print("Out-sample MSE:", mean_squared_error(y_test, y_pred))
    
    random_forest_regressor = RandomForest(model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print('---Random Forest Regressor---')
    print("In-sample MSE:", mean_squared_error(y_train, random_forest_regressor.predict(X_train)))
    print("Out-sample MSE:", mean_squared_error(y_test, y_pred))

if __name__ == "__main__":
    main()
