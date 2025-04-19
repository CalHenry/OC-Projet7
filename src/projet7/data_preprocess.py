def cleaning(features, preprocessor_pipeline):
    """Fonction cleaning finale using sklearn Pipeline"""

    if "SK_ID_CURR" in features.columns:
        features = features.drop(columns=["SK_ID_CURR"])

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


def cleaning2(features, preprocessor_pipeline):
    """Fonction cleaning finale using sklearn Pipeline"""

    # Fit the preprocessor to the training data and transform both datasets
    features_transformed = preprocessor_pipeline.transform(features)

    return features_transformed
