import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai
import time

from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

groq_api_key = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("course-resources/research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("RAG Document Q&A with Groqand Llama 3")

user_prompt = st.text_input("Enter your query from the research paper.")
        
if st.button("Document Embeddings"):
    create_vector_embeddings()
    st.write("Vector Database is ready!")
    
if user_prompt: 
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' button to create the vector database.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start_time = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        print(f"Response time : {time.process_time()-start_time}")
        st.write(response['answer'])
        
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("-"*10)