import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load trained model and label encoders
model = joblib.load("final_gb_classifier.pkl")
encoders = joblib.load("label_encoders.joblib")

# Preprocessing function
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    for col in ['Contract', 'InternetService', 'PaymentMethod']:
        df[col] = encoders[col].transform(df[col])
    return df

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction")
st.markdown("### Get insights on whether a customer is likely to churn or stay loyal.")

# Define layout columns
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", ['Male', 'Female'])
    senior_citizen = st.radio("Senior Citizen", ['Yes', 'No'])
    partner = st.radio("Partner", ['Yes', 'No'])
    dependents = st.radio("Dependents", ['Yes', 'No'])
    phone_service = st.radio("Phone Service", ['Yes', 'No'])
    multiple_lines = st.radio("Multiple Lines", ['Yes', 'No'])
    internet_service = st.radio("Internet Service", ['DSL', 'Fiber optic', 'No'])
    contract = st.radio("Contract", ['Month-to-month', 'One year', 'Two year'])

with col2:
    online_security = st.radio("Online Security", ['Yes', 'No', 'No internet service'])
    online_backup = st.radio("Online Backup", ['Yes', 'No', 'No internet service'])
    device_protection = st.radio("Device Protection", ['Yes', 'No', 'No internet service'])
    tech_support = st.radio("Tech Support", ['Yes', 'No', 'No internet service'])
    streaming_tv = st.radio("Streaming TV", ['Yes', 'No'])
    streaming_movies = st.radio("Streaming Movies", ['Yes', 'No'])
    paperless_billing = st.radio("Paperless Billing", ['Yes', 'No'])
    payment_method = st.radio("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input("Monthly Charges", value=0.0)
    total_charges = st.number_input("Total Charges", value=0.0)
    tenure_group = st.number_input("Tenure Group", value=0)

# On prediction button
if st.button("🔍 Predict"):
    # Map string inputs to model-ready format
    map_yes_no = {'Yes': 1, 'No': 0}
    map_gender = {'Male': 1, 'Female': 0}
    map_internet_na = {'Yes': 1, 'No': 0, 'No internet service': 2}

    user_data = {
        'gender': map_gender[gender],
        'SeniorCitizen': map_yes_no[senior_citizen],
        'Partner': map_yes_no[partner],
        'Dependents': map_yes_no[dependents],
        'PhoneService': map_yes_no[phone_service],
        'MultipleLines': map_yes_no[multiple_lines],
        'InternetService': internet_service,
        'OnlineSecurity': map_internet_na[online_security],
        'OnlineBackup': map_internet_na[online_backup],
        'DeviceProtection': map_internet_na[device_protection],
        'TechSupport': map_internet_na[tech_support],
        'StreamingTV': map_yes_no[streaming_tv],
        'StreamingMovies': map_yes_no[streaming_movies],
        'Contract': contract,
        'PaperlessBilling': map_yes_no[paperless_billing],
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group
    }

    processed = preprocess_input(user_data)
    prediction = model.predict(processed)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ The customer is likely to **churn**. Consider engagement strategies.")
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjE4cXI5dTZhcmpuYWRldjRvYzBzbmNrNmlkcXdnYW1yMGxzMnlzeCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/vPA5ZfXBapiAnxQIWs/giphy.gif", width=300)
    else:
        st.success("✅ Great news! The customer is likely to **stay** loyal.")
        st.image("https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3NDI0YTRrNDB6MG4xanc3cWluZWttZGViN2lobG53MXd1czI1Yml4aSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3oh25nEUPF4zqZUBPp/giphy.gif", width=300)