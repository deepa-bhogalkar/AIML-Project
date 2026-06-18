import streamlit as st
import numpy as np
from new_app import data, results

st.title("⚠️ Locust Swarm Risk Prediction")

# Select model
model_choice = st.selectbox(
    "Select Model",
    list(results.keys())
)

# Use first 6 features as inputs
sample_cols = data["features"][:6]
col_means = data["X"].mean()

st.write("Enter feature values:")

cols = st.columns(3)
user_vals = {}


display_names = {
    "X": "📍 Longitude",
    "Y": "📍 Latitude",
    "OBJECTID": "🆔 Observation ID",
    "TmSTARTDAT": "🕒 Start Time",
    "TmFINISHDA": "🕒 End Time",
    "AREAHA": "🌾 Affected Area (Hectares)"
}
for i, col in enumerate(sample_cols):
    mn = float(data["X"][col].min())
    mx = float(data["X"][col].max())
    med = float(data["X"][col].median())

    user_vals[col] = cols[i % 3].number_input(
        display_names.get(col, col),
        min_value=mn,
        max_value=mx,
        value=med,
        format="%.3f"
    )

if st.button("🔍 Predict Risk"):

    # Create complete input row
    full_input = col_means.copy()

    for col, val in user_vals.items():
        full_input[col] = val

    input_data = np.array([full_input.values])

    # Standard scaling
    input_scaled = data["scaler_std"].transform(input_data)

    # Selected model
    model = results[model_choice]["model"]

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")

    if prediction == 1:
        st.error(
            f"🔴 HIGH LOCUST SWARM RISK\n\n"
            f"Confidence: {probability[1]*100:.2f}%"
        )
    else:
        st.success(
            f"🟢 LOW LOCUST SWARM RISK\n\n"
            f"Confidence: {probability[0]*100:.2f}%"
        )