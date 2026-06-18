import streamlit as st
import pandas as pd

st.title("📁 Data Explorer")

df = pd.read_csv("cleaned_locust_dataset.csv")
st.dataframe(df.head())