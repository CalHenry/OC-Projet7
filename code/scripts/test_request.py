import sys

import requests

# API endpoints
predict_url = "http://127.0.0.1:8000/predict"
predict_by_id_url = "http://127.0.0.1:8000/predict_by_id"

data = {
    "inputs": [
        {
            "ORGANIZATION_TYPE": "Kindergarten",
            "DAYS_EMPLOYED": -2329,
            "OCCUPATION_TYPE": "Drivers",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_EDUCATION_TYPE": "Higher education",
            "CODE_GENDER": "F",
            "WEEKDAY_APPR_PROCESS_START": "TUESDAY",
            "FLAG_OWN_CAR": "N",
            "NAME_CONTRACT_TYPE": "Cash loans",
            "NAME_INCOME_TYPE": "Working",
            "NAME_HOUSING_TYPE": "House / apartment",
            "REGION_RATING_CLIENT": 2,
            "WALLSMATERIAL_MODE": " Stone, brick",
            "NAME_TYPE_SUITE": "Unaccompanied",
            "FLAG_OWN_REALTY": "Y",
        }
    ]
}

# Sample data for predict_by_id endpoint
id_client = {"client_id": 100005}


def test_predict():
    """Test the prediction endpoint with sample data"""
    print(f"Testing {predict_url}...")
    response = requests.post(predict_url, json=data)
    print(response.json())


def test_predict_by_id():
    """Test the prediction by ID endpoint"""
    print(f"Testing {predict_by_id_url}...")
    response = requests.post(predict_by_id_url, json=id_client)
    print(response.json())


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        method = sys.argv[1].lower()

        if method == "predict":
            test_predict()
        elif method == "id":
            test_predict_by_id()
        else:
            print("Unknown method. Use 'predict' or 'id'")
    else:
        # If no argument provided, run both tests
        print("Running both test methods:")
        test_predict()
        print("\n" + "-" * 50 + "\n")
        test_predict_by_id()
