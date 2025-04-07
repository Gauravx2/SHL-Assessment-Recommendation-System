import streamlit as st
import requests

API_URL = "http://localhost:8000/recommend"  # change when deployed

st.title("SHL Assessment Recommender")
st.markdown("Enter a job description or requirements to get relevant assessments")

query = st.text_area("Input text:", height=150)
num_results = st.slider("Number of results:", 1, 10, 5)

if st.button("Get Recommendations"):
    if not query:
        st.warning("Please enter some text first")
    else:
        with st.spinner("Fetching recommendations…"):
            resp = requests.post(
                API_URL,
                json={"text": query, "max_results": num_results},
                timeout=10
            )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                st.table(results)
            else:
                st.info("No assessments found for that query.")
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")
