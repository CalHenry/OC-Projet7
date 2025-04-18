import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Model class
class Model:
    def __init__(self, name, estimator, params):
        """
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
        self.y_pred = None
        self.y_pred_binary = None

        self.results = None
        self.best_threshold = None
        self.min_cost = None

    def load_data(self, X_train, X_test, y_train, y_test):
        """Load training and testing data."""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def find_best_params(self, cv=5, beta=10, search_type="grid", use_smote=False):
        """
        2 methods: GridSearchCV or RandomizedSearchCV
        Possibility to use oversampling with SMOTE in an integrated pipeline to avoid data leakage
        """
        self.beta = beta

        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not loaded yet. Call load_data first.")

        # The best parameters will maximize the fbeta_scorer
        fbeta_scorer = make_scorer(fbeta_score, beta=beta)

        # simple pipeline
        steps = []
        if use_smote:
            steps.append(("smote", SMOTE()))  # Add SMOTE if use_smote is True
        steps.append(("estimator", self.estimator))

        # Create the pipeline
        pipeline = Pipeline(steps)

        if search_type == "grid_search":
            param_grid = {
                f"estimator__{param}": config["values"]
                for param, config in self.params.items()
            }

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=fbeta_scorer,
                n_jobs=6,
            )

        elif search_type == "random_search":
            param_distributions = {}
            for param, config in self.params.items():
                if config["type"] == "int":
                    param_distributions[f"estimator__{param}"] = randint(
                        config["min"], config["max"] + 1
                    )
                elif config["type"] == "float":
                    param_distributions[f"estimator__{param}"] = uniform(
                        config["min"], config["max"] - config["min"]
                    )
                elif config["type"] == "str":
                    param_distributions[f"estimator__{param}"] = config["values"]

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions,
                cv=cv,
                scoring=fbeta_scorer,
                n_iter=100,  # Randomized search is 'always' better than grid search if n_iter is > 90
                n_jobs=-4,
            )

        else:
            raise ValueError(
                "search_type must be 'grid_search' or 'random_search' for GridSearchCV or RandomizedSearchCV"
            )

        search.fit(self.X_train, self.y_train)
        self.best_model = search.best_estimator_.steps[0][1]

        return {
            "best_params": search.best_params_,
            "best_score (fβ)": search.best_score_,
        }

    def predict_proba(self):
        # Get predictions as probabilities
        y_pred = self.best_model.predict_proba(self.X_test)
        self.y_pred = y_pred

    def find_optimal_threshold_for_min_cost(self, cost_fn=10, cost_fp=1):
        """
        Minize the total cost given the simple formula.
        Find the best threshold given the minimized total cost
        :param desired_fnr: targeted False Negative Rate (FNR).
        """
        # Get predictions as probabilities
        self.predict_proba()

        thresholds = np.linspace(0, 1, 1000)
        min_cost = float("inf")
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred_binary = (self.y_pred[:, 1] >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred_binary).ravel()

            # Simple cost function that directly depends on the costs ratios to determine the importance of FP or FN.
            # In our case, a False Negative cost much ore that a False positive.
            total_cost = (fp * cost_fp) + (fn * cost_fn)

            if total_cost < min_cost:
                min_cost = total_cost
                best_threshold = threshold
                best_y_pred_binary = y_pred_binary

        y_pred_binary = (self.y_pred[:, 1] >= best_threshold).astype(int)

        self.y_pred_binary = best_y_pred_binary
        self.best_threshold = best_threshold
        self.min_cost = min_cost

        print(f"best_threshold: {best_threshold:.4f}, minimized cost: {min_cost:.4f}")

    def evaluate(self, metrics=None):
        """
        Evaluate the best model on the test set using multiple metrics.
        Parameters: metrics : List of metric names to calculate'
        Returns: Dictionary containing the scores for all requested metrics
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call find_best_params first.")

        results = {}

        if metrics is None:
            metrics = ["fbeta", "precision", "recall", "roc_auc"]

        # Calculate requested metrics
        for metric in metrics:
            if metric == "fbeta":
                results["fbeta"] = fbeta_score(
                    self.y_test, self.y_pred_binary, beta=self.beta
                )
            elif metric == "precision":
                results["precision"] = precision_score(self.y_test, self.y_pred_binary)
            elif metric == "recall":
                results["recall"] = recall_score(self.y_test, self.y_pred_binary)
            elif metric == "roc_auc":
                results["roc_auc"] = roc_auc_score(self.y_test, self.y_pred_binary)

        print(f"(β={self.beta}) score: {results}")

        self.results = results

    def get_model_info(self):
        """
        Returns a dictionary containing the model's information.
        """
        return {
            "name": self.name,
            "estimator": self.estimator,
            "params": self.params,
            "best_model": self.best_model,
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_pred": self.y_pred,
            "y_pred_binary": self.y_pred_binary,
            "results": self.results,
            "min_cost": self.min_cost,
            "best_threshold": self.best_threshold,
        }


from sklearn.pipeline import Pipeline


def cleaning(features, preprocessor_pipeline, print_shape=True):
    """Fonction cleaning finale using sklearn Pipeline"""

    categorical_columns = features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_columns = features.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Fit the preprocessor to the training data and transform both datasets
    features_transformed = preprocessor_pipeline.transform(features)

    # Get feature names after one-hot encoding
    # First, get numerical feature names directly
    feature_names = numerical_columns.copy()

    # Then add categorical feature names after one-hot encoding
    if categorical_columns:
        ohe = preprocessor_pipeline.named_transformers_["categorical"].named_steps[
            "encoder"
        ]
        categorical_feature_names = ohe.get_feature_names_out(categorical_columns)
        feature_names.extend(categorical_feature_names)

    return features_transformed
