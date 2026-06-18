import streamlit as st
import pandas as pd

st.title("📁 Data Explorer")

df = pd.read_csv("cleaned_locust_dataset.csv")

st.write("Dataset Shape:", df.shape)

st.subheader("First 5 Records")
st.dataframe(df.head())

st.subheader("Statistical Summary")
st.write(df.describe())