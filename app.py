import streamlit as st
from augmentation_and_generation import chat  

st.set_page_config(page_title="Laptop Assistant", layout="centered")

st.markdown(
    """
    <h2 style='text-align:center;'>Deep Thinking Laptop Assistant</h2>
    <p style='text-align:center; color:gray;'>Ask anything about laptops (e.g., "Best Acer laptop for gaming")</p>
    <hr>
    """,
    unsafe_allow_html=True
)


user_input = st.chat_input("Type your question and press Enter...")

if user_input:
    
    st.chat_message("user").markdown(user_input)

    try:
        
        answer = chat(user_input)
        if answer:
            st.chat_message("assistant").markdown(answer)
        else:
            st.chat_message("assistant").markdown("No detailed answer returned.")
    except Exception as e:
        st.chat_message("assistant").markdown(f"Error: {e}")
