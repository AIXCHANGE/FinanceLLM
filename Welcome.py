import streamlit as st

st.set_page_config(page_title="Welcome to HarmonyAI", page_icon="chart_with_upwards_trend")
st.header("Welcome to Finance GPT")
st.subheader("Powered by Harmony AI")

st.sidebar.success("Select the next page to proceed")

st.markdown(
    """
    We process the LLM on two types of data ! The first data comes from historical bitcoin prices taken from the present moment to a 
    time in the past as chosen by you. The second data comes from the bitcoin address you give. It tracks your transactions and makes a dataframe.
    ### Steps to use >>>
    - Fill up the openai api key,personal bitcoin address.
    - We scrap bitcoin data from the present date to a previous date. Choose the historical date and time uptill which you wish to track
    - Press update data to keep the data updated to the latest second.
    - To view the data you can visit know your data on the sidebar.
    - To use the finance gpt application click on Chatbot.
    - Choose which dataframe you wish to ask questions to . You can use both too ! 
    """
)

st.warning("The app is deployed on streamlit and hence a bit slow! This is absolutely for testing purposes and none of the api keys or addresses that you enter are being saved !")
