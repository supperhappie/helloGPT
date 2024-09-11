import streamlit as st
from header import set_header

# Call the global configuration
set_header()

tabs = st.tabs(["tap1", "tap2"])

# Select which tab is active
with tabs[0]:
    st.write("This is tap1")

with tabs[1]:
    st.write("This is tap2")
    
    
    
model = st.selectbox(
    "Choose your model",
    (
        "GPT-3",
        "GPT-4",
    ),
)

if model == "GPT-3":
    st.write("cheap")
else:
    st.write("not cheap")
    name = st.text_input("What is your name?")
    st.write(name)

    value = st.slider(
        "temperature",
        min_value=0.1,
        max_value=1.0,
    )

    st.write(value)