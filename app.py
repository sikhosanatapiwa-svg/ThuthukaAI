import json
import streamlit as st
from google import genai

MODEL_NAME = "gemini-3.8-flash"

with open("programmes.json", encoding="utf-8") as f:
    programmes = json.load(f)

st.title("THUTHUKA-AI")
st.write(f"Loaded {len(programmes)} programmes")

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"])

if st.button("Test the AI"):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents="In two sentences, explain what a learnership is in South Africa.",
    )
    st.write(response.text)
