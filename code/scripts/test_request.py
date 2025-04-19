import requests

url = "http://127.0.0.1:8000/predict/"


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

response = requests.post(url, json=data)
print(response.json())
