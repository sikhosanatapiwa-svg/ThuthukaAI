import streamlit as st

st.title("My Career Guide")
name = st.text_input("What's your name?")
if st.button("Start"):
    st.write(f"Welcome, {name}!")
