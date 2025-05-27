from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter

import os 
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
model = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

### 1. Session-based memory store
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

### 2. Message trimmer (to manage context length)
trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

### 3. Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all the questions to the best of your ability in {language}."),
    MessagesPlaceholder(variable_name="messages")
])

### 4. Define full chain with message trimming
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

### 5. Wrap the chain with session-based message history
chat_chain = RunnableWithMessageHistory(
    chain=chain,
    get_chat_history=get_session_history,
    input_messages_key="messages"
)

### 6. Chat invocation with session ID
config = {"configurable": {"session_id": "chat_session_1"}}

# Sample messages
messages = [HumanMessage(content="Hi, I'm Vivek. What is 2 + 2?")]

response = chat_chain.invoke(
    {
        "messages": messages,
        "language": "English"
    },
    config=config
)

print(response.content)
