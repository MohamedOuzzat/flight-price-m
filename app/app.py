import streamlit as st
import joblib
import json
import pandas as pd

# ── Load model (cached) ────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipeline = joblib.load("model/flight_model.joblib")
    with open("model/metadata.json") as f:
        metadata = json.load(f)
    return pipeline, metadata

pipeline, metadata = load_model()

# ── UI ─────────────────────────────────────────────────────────────
st.title("✈️ Flight Price Predictor")
st.markdown("Enter the flight details to get an estimated ticket price.")

# Inputs — adapt these to your actual features
airline = st.selectbox("Airline", ["IndiGo", "Air India", "SpiceJet", "Vistara", "GoAir"])
source = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"])
destination = st.selectbox("Destination City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"])
stops = st.selectbox("Number of Stops", [0, 1, 2, 3])
departure_time = st.selectbox("Departure Time", ["Morning", "Afternoon", "Evening", "Night", "Late Night"])
arrival_time = st.selectbox("Arrival Time", ["Morning", "Afternoon", "Evening", "Night", "Late Night"])
flight_class = st.selectbox("Class", ["Economy", "Business"])
duration = st.number_input("Duration (hours)", min_value=0.5, max_value=50.0, value=2.5, step=0.5)
days_left = st.number_input("Days Before Departure", min_value=1, max_value=365, value=30)

# ── Predict ────────────────────────────────────────────────────────
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "airline": airline,
        "source_city": source,
        "destination_city": destination,
        "stops": stops,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "class": flight_class,
        "duration": duration,
        "days_left": days_left,
    }])

    prediction = pipeline.predict(input_data)[0]

    st.success(f"💰 Estimated Price: **{prediction:,.0f} INR**")

    with st.expander("Prediction details"):
        st.write("**Input used:**")
        st.dataframe(input_data)