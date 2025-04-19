from typing import List

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from projet7.data_preprocess import cleaning

model_info = joblib.load("models/lgbm_model_15_info.joblib")
preprocessor = joblib.load("models/pipelines/processed/preprocessor_top15.joblib")

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
    probabilities: List[float]
    binary_predictions: List[int]


app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting if a client will repay a credit",
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert Pydantic models to a pandas DataFrame
        input_data = pd.DataFrame([client.dict() for client in request.inputs])

        # Apply your preprocessing function
        X_test_processed = cleaning(input_data, preprocessor_pipeline=preprocessor)

        # Get prediction probabilities
        y_pred_proba = model_info["best_model"].predict_proba(X_test_processed)[:, 1]

        # Apply the custom threshold to get binary predictions
        y_pred_binary = (y_pred_proba >= model_info["best_threshold"]).astype(int)

        return PredictionResponse(
            probabilities=y_pred_proba.tolist(),
            binary_predictions=y_pred_binary.tolist(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model_info")
def get_model_info():
    try:
        # Return the model information as a JSON
        model_info_fromAPI = model_info.get_model_info()
        return model_info_fromAPI
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# For running the application locally
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
