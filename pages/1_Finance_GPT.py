import requests
import pandas
import os
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
from langchain.agents import create_csv_agent
from langchain.prompts import PromptTemplate
import datetime
import time
from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

import openai
import streamlit as st
from streamlit_chat import message

def make_data(from_unix,to_unix,bitcoin_address):
    x = requests.get(f'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={from_unix}&to={to_unix}&precision=1')
    output = x.json()
    output = output['prices']
    columns = ['Time', 'Bitcoin_price_usd']
    df_bitcoin = pd.DataFrame(output, columns=columns)
    df_bitcoin['Time'] = df_bitcoin['Time'].apply(lambda x: datetime.datetime.utcfromtimestamp(int(x) / 1000).strftime('%Y-%m-%d %H:%M:%S'))

    transactions_url = 'https://blockchain.info/rawaddr/' + bitcoin_address
    df = pandas.read_json(transactions_url)
    transactions = df['txs']
    test = []
    for t in transactions:
        test.append(t)
    df_user = pd.DataFrame(test)
    df_user['Time_of_transaction'] = df_user['time'].apply(
        lambda x: datetime.datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
    df_user.drop(columns=['inputs', 'out'], inplace=True)

    return df_bitcoin,df_user

def generate_response(prompt):
    if(st.session_state['data']=="Historical Bitcoin Data"):
        data=st.session_state['df_bitcoin']
    if (st.session_state['data'] == "User Bitcoin Transaction"):
        data = st.session_state['df_user']
    else:
        data=[st.session_state['df_bitcoin'],st.session_state['df_user']]
    pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0.4), data, verbose=False)
    st.session_state['messages'].append({"role": "user", "content": prompt})
    response = pd_agent.run(prompt)
    # completion = openai.ChatCompletion.create(
    #     model='GPT4',
    #     messages=st.session_state['messages']
    # )
    # response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})


    return response   #, total_tokens, prompt_tokens, completion_tokens



# Initialise session state variables

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

if 'data' not in st.session_state:
    st.session_state['data'] = []

if 'df_bitcoin' not in st.session_state:
    st.session_state['df_bitcoin'] = []

if 'df_user' not in st.session_state:
    st.session_state['df_user'] = []






# Setting page title and header
st.set_page_config(page_title="HarmonyAI", page_icon="chart_with_upwards_trend")
st.markdown("<h1 style='text-align: center;'> Finance GPT</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'> Powered by HarmonyAI</h3>", unsafe_allow_html=True)


api_key=st.text_input(label="Enter your openai api key",placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
openai.api_key=str(api_key)
bitcoin_address=st.text_input(label="Enter your Bitcoin Address",placeholder="bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh")
start_date = st.date_input('Enter start date', value=datetime.datetime(2019,7,6))
start_time = st.time_input('Enter start time', datetime.time(8, 45,30))
from_time=datetime.datetime.combine(start_date, start_time)
to_time=datetime.datetime.now()

from_unix=time.mktime(from_time.timetuple())
to_unix=time.mktime(to_time.timetuple())
os.environ["OPENAI_API_KEY"]=openai.api_key

st.session_state['data']=st.radio(
        "Which Dataframe would you like to ask questions to?",
        options=["Historical Bitcoin Data", "User Bitcoin Transaction", "Both"],
    )

if st.button('Update data'):
    df_bitcoin,df_user= make_data(from_unix,to_unix,bitcoin_address)
    st.session_state['df_bitcoin'] = df_bitcoin
    st.session_state['df_user'] = df_user



# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Welcome")
sub_page_name = st.sidebar.radio("Contents", ("Chatbot", "Know your data"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write("See how the dataframe looks to ask questions better")
clear_button = st.sidebar.button("Clear Conversation", key="clear")


if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]





if(sub_page_name=="Chatbot"):

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=40)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = generate_response(user_input) #, total_tokens, prompt_tokens, completion_tokens
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)


    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',avatar_style="avataaars",seed=122)
                message(st.session_state["generated"][i], key=str(i),avatar_style="avataaars",seed=1)

else:
    try:
        st.dataframe(data=st.session_state['df_bitcoin'])
        st.dataframe(data=st.session_state['df_user'])
    except:
        st.error("Please update data")
