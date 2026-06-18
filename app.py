import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="LocustGuard AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Main Page
# -----------------------------
st.title("🌾 LocustGuard AI")
st.subheader("Locust Swarm Risk Prediction Platform")

st.markdown("---")

st.write("""
Welcome to **LocustGuard AI**, an intelligent system designed to predict
the risk of locust swarm occurrences using Machine Learning techniques.

This platform helps in:
- 🌾 Early warning of locust outbreaks
- 📊 Data analysis and visualization
- 🤖 Machine Learning-based risk prediction
- 🧠 AI-generated insights for agriculture
- 🌍 Supporting farmers and agricultural authorities
""")

st.markdown("---")

# -----------------------------
# Project Workflow
# -----------------------------
st.header("🔄 Project Workflow")

st.markdown("""
**Dataset**
⬇️

**Data Preprocessing**
- Handling Missing Values
- Encoding Categorical Variables
- Feature Scaling

⬇️

**Machine Learning Model**
- Logistic Regression

⬇️

**Risk Prediction**
- High Risk
- Low Risk

⬇️

**Interactive Dashboard**
- Visualizations
- Insights
- Predictions
""")

st.markdown("---")

# -----------------------------
# Technologies Used
# -----------------------------
st.header("🛠️ Technologies Used")

col1, col2 = st.columns(2)

with col1:
    st.write("✅ Python")
    st.write("✅ Pandas")
    st.write("✅ NumPy")
    st.write("✅ Scikit-Learn")

with col2:
    st.write("✅ Streamlit")
    st.write("✅ Git & GitHub")
    st.write("✅ Render")
    st.write("✅ Logistic Regression")

st.markdown("---")

# -----------------------------
# Sidebar Instructions
# -----------------------------
st.info(
    "👈 Use the sidebar to navigate through Dashboard, "
    "Data Explorer, Model Comparison, Risk Prediction, "
    "AI Insights, and About Project."
)

st.success("✅ LocustGuard AI is Ready for Prediction!")