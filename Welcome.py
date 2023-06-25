import streamlit as st

st.set_page_config(page_title="Welcome to HarmonyAI", page_icon="chart_with_upwards_trend")
st.header("Welcome to Finance GPT")
st.subheader("Powered by Harmony AI")

st.sidebar.success("Select any of the pages mentioned")

st.markdown(
    """
    ## How to use ?
    - We process the LLM on two types of data ! The first data comes from historical bitcoin prices taken from the present moment to a 
    time in the past as chosen by you. The second data comes from the bitcoin address you give. It tracks your transactions and makes a dataframe.
    - To see both these dataframes , you can head over to Know your data and then proceed to the application!
    
    ## The app is currently deployed on Streamlit servers, hence maybe a bit slow! Request for the users to have patience while testing.
    """
)
