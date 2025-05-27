# %%
import os
import bs4
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize components
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

# Load and preprocess documents
loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))},
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Prompts
system_prompt = """You are an assistant for question-answering task. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know. Use three sentences at maximum and keep the answer concise.
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference the context in the chat history, "
               "formulate a standalone question which can be understood without the chat history. Do not answer the question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create components
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Setup memory store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Final conversational chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Example usage
response1 = conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={"configurable": {"session_id": "abc123"}}
)
print("Answer 1:", response1["answer"])

response2 = conversational_rag_chain.invoke(
    {"input": "What are the common ways to doing it?"},
    config={"configurable": {"session_id": "abc123"}}
)
print("Answer 2:", response2["answer"])
