# template for the classes

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split


class Model:
    def __init__(self, name, estimator, params):
        """
        Initialize a model with a name, estimator, and parameters.

        Parameters:
        - name: str, name of the model
        - estimator: sklearn estimator object
        - params: dict, parameters for grid search
        """
        self.name = name
        self.estimator = estimator
        self.params = params
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, X_train, X_test, y_train, y_val):
        """Load training and testing data."""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_val

    def find_best_params(self, cv=5):
        """
        Use GridSearchCV to find the best parameters.

        Parameters:
        - cv: int, number of cross-validation folds
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not loaded yet. Call load_data first.")

        grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.params,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
        }

    def evaluate(self):
        """Evaluate the best model on the test set."""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call find_best_params first.")

        return self.best_model.score(self.X_test, self.y_test)


# Generate some example data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create two model objects
dummy_model = Model(
    name="Dummy Classifier",
    estimator=DummyClassifier(),
    params={"strategy": ["most_frequent", "stratified", "uniform"]},
)

logistic_model = Model(
    name="Logistic Regression",
    estimator=LogisticRegression(max_iter=1000),
    params={"C": [0.1, 1.0, 10.0], "penalty": ["l2"], "solver": ["liblinear", "saga"]},
)

# Load data for both models
dummy_model.load_data(X_train, X_test, y_train, y_test)
logistic_model.load_data(X_train, X_test, y_train, y_test)

# Find best parameters for each model
dummy_results = dummy_model.find_best_params()
print(f"\n{dummy_model.name} best parameters:")
print(dummy_results)
print(f"Test accuracy: {dummy_model.evaluate():.4f}")

logistic_results = logistic_model.find_best_params()
print(f"\n{logistic_model.name} best parameters:")
print(logistic_results)
print(f"Test accuracy: {logistic_model.evaluate():.4f}")
