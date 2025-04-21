from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

model_info = joblib.load("models/lgbm_model_15_info.joblib")
preprocessor = joblib.load("models/pipelines/preprocessor_top15.joblib")

# Load the test set
test_set = joblib.load("data/processed/app_test_domain_top15.joblib")

print("Model:", model_info["name"])


# Create a Pydantic model for request validation
# we have selected the 15 most important variables
class ClientData(BaseModel):
    ORGANIZATION_TYPE: str
    DAYS_EMPLOYED: int
    OCCUPATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_EDUCATION_TYPE: str
    CODE_GENDER: str
    WEEKDAY_APPR_PROCESS_START: str
    FLAG_OWN_CAR: str
    NAME_CONTRACT_TYPE: str
    NAME_INCOME_TYPE: str
    NAME_HOUSING_TYPE: str
    REGION_RATING_CLIENT: int
    WALLSMATERIAL_MODE: str
    NAME_TYPE_SUITE: str
    FLAG_OWN_REALTY: str


# Create a Pydantic model for the response
class PredictionRequest(BaseModel):
    inputs: List[ClientData]


class PredictionResponse(BaseModel):
    predict_proba: List[float]
    binary_prediction: List[int]


class ClientIDRequest(BaseModel):
    client_id: int


app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting if a client will repay a credit",
    version="0.1.0",
)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert Pydantic models to a pandas DataFrame
        input_data = pd.DataFrame([client.dict() for client in request.inputs])

        # Apply your preprocessing function
        # X_test_processed = cleaning2(input_data, preprocessor_pipeline=preprocessor)
        X_test_processed = preprocessor.transform(input_data)

        # Get prediction probabilities
        y_pred_proba = model_info["best_model"].predict_proba(X_test_processed)[:, 1]

        # Apply the custom threshold to get binary predictions
        y_pred_binary = (y_pred_proba >= model_info["best_threshold"]).astype(int)

        return PredictionResponse(
            predict_proba=y_pred_proba.tolist(),
            binary_prediction=y_pred_binary.tolist(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_by_id", response_model=PredictionResponse)
def predict_by_id(request: ClientIDRequest):
    try:
        # Retrieve client data from the test set using the client ID
        client_data = test_set[test_set["SK_ID_CURR"] == request.client_id]

        if client_data.empty:
            raise HTTPException(status_code=404, detail="Client ID not found")

        # Apply your preprocessing function
        X_test_processed = cleaning2(client_data, preprocessor_pipeline=preprocessor)

        # Get prediction probabilities
        y_pred_proba = model_info["best_model"].predict_proba(X_test_processed)[:, 1]

        # Apply the custom threshold to get binary predictions
        y_pred_binary = (y_pred_proba >= model_info["best_threshold"]).astype(int)

        return PredictionResponse(
            predict_proba=y_pred_proba.tolist(),
            binary_prediction=y_pred_binary.tolist(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
