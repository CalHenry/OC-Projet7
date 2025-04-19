import requests
import streamlit as st

st.title("Credit Risk Prediction App")

st.write("""
This application allows you to predict credit risk using our model in two ways:
1. By entering a client ID (for existing clients)
2. By entering client information manually (for new clients)
""")

# Create tabs for the two different methods
tab1, tab2 = st.tabs(["Predict by Client ID", "Predict by Client Information"])

# Define API endpoints
API_ENDPOINT = "http://localhost:8000"  # Update this to your API host

# Tab 1: Predict by Client ID
with tab1:
    st.header("Predict by Client ID")
    client_id = st.number_input("Enter Client ID", min_value=1, step=1)

    if st.button("Get Prediction by ID"):
        try:
            response = requests.post(
                f"{API_ENDPOINT}/predict_by_id", json={"client_id": client_id}
            )

            if response.status_code == 200:
                result = response.json()
                st.success("Prediction successful!")

                # Display results in a more readable format
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Probability", f"{result['predict_proba'][0]:.4f}")
                with col2:
                    prediction = (
                        "High Risk"
                        if result["binary_prediction"][0] == 1
                        else "Low Risk"
                    )
                    st.metric("Prediction", prediction)

            elif response.status_code == 404:
                st.error("Client ID not found. Please try a different ID.")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")

# Tab 2: Predict by Client Information
with tab2:
    st.header("Predict by Client Information")

    # Define the expected values for categorical variables
    organization_types = [
        "Business Entity Type 1",
        "Business Entity Type 2",
        "Business Entity Type 3",
        "Self-employed",
        "Government",
        "Other",
    ]  # Replace with actual values
    occupation_types = [
        "Laborers",
        "Core staff",
        "Managers",
        "Drivers",
        "High skill tech staff",
        "Accountants",
        "Medicine staff",
        "Other",
    ]  # Replace with actual values
    family_status = [
        "Single / not married",
        "Married",
        "Civil marriage",
        "Widow",
        "Separated",
        "Unknown",
    ]
    education_types = [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ]
    gender_options = ["M", "F"]
    weekdays = [
        "MONDAY",
        "TUESDAY",
        "WEDNESDAY",
        "THURSDAY",
        "FRIDAY",
        "SATURDAY",
        "SUNDAY",
    ]
    yes_no_options = ["Y", "N"]
    contract_types = ["Cash loans", "Revolving loans"]
    income_types = [
        "Working",
        "State servant",
        "Commercial associate",
        "Pensioner",
        "Unemployed",
        "Student",
        "Businessman",
        "Maternity leave",
    ]
    housing_types = [
        "House / apartment",
        "With parents",
        "Municipal apartment",
        "Rented apartment",
        "Office apartment",
        "Co-op apartment",
    ]
    walls_material = [
        "Panel",
        "Stone, brick",
        "Block",
        "Wooden",
        "Monolithic",
        "Others",
    ]
    type_suite = [
        "Unaccompanied",
        "Family",
        "Spouse, partner",
        "Children",
        "Other_A",
        "Other_B",
        "Group of people",
    ]

    # Create form for input fields
    with st.form("client_data_form"):
        st.subheader("Client Information")

        col1, col2 = st.columns(2)

        with col1:
            organization_type = st.selectbox("Organization Type", organization_types)
            days_employed = st.number_input(
                "Days Employed (negative values for years)", value=-1000
            )
            occupation_type = st.selectbox("Occupation Type", occupation_types)
            family_status = st.selectbox("Family Status", family_status)
            education_type = st.selectbox("Education Type", education_types)
            gender = st.selectbox("Gender", gender_options)
            weekday = st.selectbox("Application Weekday", weekdays)

        with col2:
            own_car = st.selectbox("Owns Car", yes_no_options)
            contract_type = st.selectbox("Contract Type", contract_types)
            income_type = st.selectbox("Income Type", income_types)
            housing_type = st.selectbox("Housing Type", housing_types)
            region_rating = st.selectbox("Region Rating", [1, 2, 3])
            walls_material_mode = st.selectbox("Walls Material", walls_material)
            type_suite = st.selectbox("Type Suite", type_suite)
            own_realty = st.selectbox("Owns Realty", yes_no_options)

        submit_button = st.form_submit_button("Get Prediction")

    if submit_button:
        # Create client data dictionary
        client_data = {
            "ORGANIZATION_TYPE": organization_type,
            "DAYS_EMPLOYED": days_employed,
            "OCCUPATION_TYPE": occupation_type,
            "NAME_FAMILY_STATUS": family_status,
            "NAME_EDUCATION_TYPE": education_type,
            "CODE_GENDER": gender,
            "WEEKDAY_APPR_PROCESS_START": weekday,
            "FLAG_OWN_CAR": own_car,
            "NAME_CONTRACT_TYPE": contract_type,
            "NAME_INCOME_TYPE": income_type,
            "NAME_HOUSING_TYPE": housing_type,
            "REGION_RATING_CLIENT": region_rating,
            "WALLSMATERIAL_MODE": walls_material_mode,
            "NAME_TYPE_SUITE": type_suite,
            "FLAG_OWN_REALTY": own_realty,
        }

        try:
            # Format the request payload
            payload = {"inputs": [client_data]}

            # Make the API request
            response = requests.post(f"{API_ENDPOINT}/predict", json=payload)

            if response.status_code == 200:
                result = response.json()
                st.success("Prediction successful!")

                # Display results in a more readable format
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Probability", f"{result['predict_proba'][0]:.4f}")
                with col2:
                    prediction = (
                        "High Risk"
                        if result["binary_prediction"][0] == 1
                        else "Low Risk"
                    )
                    st.metric("Prediction", prediction)

            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")

st.markdown("---")
st.caption("Credit Risk Prediction App v0.1.0")
