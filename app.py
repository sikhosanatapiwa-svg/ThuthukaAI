import streamlit as st
from google import genai

MODEL_NAME = "gemini-3.8-flash"

st.title(" THUTHUKA-AI")

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"])

if st.button("Test the AI"):
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="In two sentences, explain what a learnership is in South Africa.",
    )
    st.write(response.text)
