import streamlit as st
import pandas as pd
import joblib

# Load saved model files
model = joblib.load("car_price_model.pkl")
encoder = joblib.load("encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")
st.title("🚗 Car Price Prediction App")

st.write("Enter car details below to estimate the selling price.")

# User Inputs
name = st.text_input("Car Name", "Maruti Swift")
year = st.number_input("Year", min_value=1990, max_value=2024, value=2017)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=30000)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])

# Predict Button
if st.button("Predict Price"):
    # Create input dataframe
    input_data = pd.DataFrame([{
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner
    }])

    # Encode input
    encoded_input = encoder.transform(input_data)

    # Predict
    predicted_price = model.predict(encoded_input)[0]

    st.success(f"💰 Estimated Selling Price: ₹ {int(predicted_price):,}")
