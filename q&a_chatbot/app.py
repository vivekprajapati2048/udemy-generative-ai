import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"  # os.getenv("LANGCHAIN_PROJECT")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assitant. Please respond to the user query."),
        ("user", "Question: {question}")
    ]
)


def generate_response(question, api_key, engine, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer
    
st.title("Enhanced Q&A Chatbot with OpenAI")
st.sidebar.title("Settings")

api_key = st.sidebar.text_input("Enter you OpenAI API Key:", type="password")
engine = st.sidebar.selectbox("Select an OpenAI model:", ["gpt-4o-mini", "gpt-4o"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0)
max_tokens = st.sidebar.slider("Max Tokens:", min_value=50, max_value=100)

st.write("How can I help you?")

user_input = st.text_input("You:")

if user_input and api_key:
    response = generate_response(user_input, api_key, engine, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter the OpenAI API Key in the sidebar.")
else:
    st.write("Please input something in the text box.")