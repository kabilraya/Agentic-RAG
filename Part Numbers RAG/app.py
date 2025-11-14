
import streamlit as st
from agent import chat  
st.set_page_config(page_title="MISUMI Chat Assistant", layout="wide")
st.title("MISUMI Part Number Chat Assistant")
st.write("Ask about part numbers and get detailed specs.")


query = st.text_input("Enter your query:", placeholder="e.g., CF3UU or 10mm ID cam follower")

if st.button("Send"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing..."):
            response = chat(query)  
        st.success("Response:")
        st.write(response)
