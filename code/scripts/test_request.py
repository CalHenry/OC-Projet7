import requests

url = "http://127.0.0.1:8000/predict/"

data = {
    "ORGANIZATION_TYPE": "Kindergarten",
    "DAYS_EMPLOYED": -2329,
    "OCCUPATION_TYPE": "Drivers",
    "_FAMILY_STATUS": "Married",
    "_EDUCATION_TYPE": "Higher education",
    "CODE_GENDER": "F",
    "WEEKDAY_APPR_PROCESS_START": "TUESDAY",
    "FLAG_OWN_CAR": "N",
    "_CONTRACT_TYPE": "Cash loans",
    "_INCOME_TYPE": "Working",
    "_HOUSING_TYPE": "House / apartment",
    "REGION_RATING_CLIENT": 2,
    "WALLSMATERIAL_MODE": " Stone, brick",
    "_TYPE_SUITE": "Unaccompanied",
    "FLAG_OWN_REALTY": "Y",
}

response = requests.post(url, json=data)
print(response.json())
