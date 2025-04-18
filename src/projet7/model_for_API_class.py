class CreditRiskModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)  # Load your trained model

    def preprocess(self, data):
        # Your preprocessing pipeline
        return preprocessing_function(data)

    def predict(self, data):
        processed_data = self.preprocess(data)
        probabilities = self.model.predict_proba(processed_data)[:, 1]
        binary_predictions = self.model.predict(processed_data)
        return probabilities, binary_predictions
