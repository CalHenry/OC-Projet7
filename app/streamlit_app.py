import requests
import streamlit as st

st.title("Credit Risk Prediction App")

st.write("""
This application allows you to predict credit risk using our model in two ways:
1. By entering a client ID (for existing clients)
2. By entering client information manually (for new clients)
""")

# 2 tabs for the two different methods
tab1, tab2 = st.tabs(["Predict by Client ID", "Predict by Client Information"])

# Define API endpoints
# API_ENDPOINT = "http://localhost:8000"
API_ENDPOINT = "https://calhenry-api-mlmodel-2.hf.space"

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
                        "Credit risky"
                        if result["binary_prediction"][0] == 1
                        else "Credit possible"
                    )

                    # Create a container for the metric
                    metric_container = col2.container()

                    # Apply color styling based on prediction
                    if result["binary_prediction"][0] == 1:
                        metric_container.markdown(
                            f"""
                            <div style="color: red;">
                                <small>Prediction</small><br>
                                <span style="font-size: 1.5rem; font-weight: bold;">{prediction}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        metric_container.markdown(
                            f"""
                            <div style="color: green;">
                                <small>Prediction</small><br>
                                <span style="font-size: 1.5rem; font-weight: bold;">{prediction}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
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
        "Business Entity Type 3",
        "School",
        "Government",
        "Religion",
        "Other",
        "XNA",
        "Electricity",
        "Medicine",
        "Business Entity Type 2",
        "Self-employed",
        "Transport: type 2",
        "Construction",
        "Housing",
        "Kindergarten",
        "Trade: type 7",
        "Industry: type 11",
        "Military",
        "Services",
        "Security Ministries",
        "Transport: type 4",
        "Industry: type 1",
        "Emergency",
        "Security",
        "Trade: type 2",
        "University",
        "Transport: type 3",
        "Police",
        "Business Entity Type 1",
        "Postal",
        "Industry: type 4",
        "Agriculture",
        "Restaurant",
        "Culture",
        "Hotel",
        "Industry: type 7",
        "Trade: type 3",
        "Industry: type 3",
        "Bank",
        "Industry: type 9",
        "Insurance",
        "Trade: type 6",
        "Industry: type 2",
        "Transport: type 1",
        "Industry: type 12",
        "Mobile",
        "Trade: type 1",
        "Industry: type 5",
        "Industry: type 10",
        "Legal Services",
        "Advertising",
        "Trade: type 5",
        "Cleaning",
        "Industry: type 13",
        "Trade: type 4",
        "Telecom",
        "Industry: type 8",
        "Realtor",
        "Industry: type 6",
    ]
    occupation_types = [
        "Laborers",
        "Core staff",
        "Accountants",
        "Managers",
        "Drivers",
        "Sales staff",
        "Cleaning staff",
        "Cooking staff",
        "Private service staff",
        "Medicine staff",
        "Security staff",
        "High skill tech staff",
        "Waiters/barmen staff",
        "Low-skill Laborers",
        "Realty agents",
        "Secretaries",
        "IT staff",
        "HR staff",
    ]
    family_status = [
        "Single / not married",
        "MarriedCivil marriage",
        "Widow",
        "Seprated",
        "Unknown",
    ]
    education_types = [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ]
    gender_options = ["M", "F", "XNA"]
    weekdays = [
        "WEDNESDAY",
        "MONDAY",
        "THURSDAY",
        "SUNDAY",
        "SATURDAY",
        "FRIDAY",
        "TUESDA",
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
        "Rented apartment",
        "With parents",
        "Municipal apartment",
        "Office apartment",
        "Co-op apartment",
    ]
    walls_material = [
        "Stone, brick",
        "Block",
        "",
        "Panel",
        "Mixed",
        "Wooden",
        "Others",
        "Monolithic",
    ]
    type_suite = [
        "Unaccompanied",
        "Family",
        "Spouse, partner",
        "Children",
        "Other_A",
        "Other_B",
        "Group of people",
        "",
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
                        "Credit risky"
                        if result["binary_prediction"][0] == 1
                        else "Credit possible"
                    )
                    # Create a container for the metric
                    metric_container = col2.container()

                    # Apply color styling based on prediction
                    if result["binary_prediction"][0] == 1:
                        metric_container.markdown(
                            f"""
                            <div style="color: red;">
                                <small>Prediction</small><br>
                                <span style="font-size: 1.5rem; font-weight: bold;">{prediction}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        metric_container.markdown(
                            f"""
                            <div style="color: green;">
                                <small>Prediction</small><br>
                                <span style="font-size: 1.5rem; font-weight: bold;">{prediction}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")

st.markdown("---")
st.caption("Credit Risk Prediction App v0.1.0")
