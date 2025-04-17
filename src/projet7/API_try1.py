import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

model_info = joblib.load("data/processed/lgbm_model_15_info.joblib")

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
    X_test: list


class PredictionResponse(BaseModel):
    probabilities: list
    binary_predictions: list


# Initialize the FastAPI application
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
        # Convert the input data to a numpy array
        X_test = np.array(request.X_test)

        # Get prediction probabilities
        y_pred_proba = model_info["best_model"].predict_proba(X_test)[:, 1]

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
        # Return the model information as a JSON response
        model_info_fromAPI = model_info.get_model_info()
        return model_info_fromAPI
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# For running the application locally
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
