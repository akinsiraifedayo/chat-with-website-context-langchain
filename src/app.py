import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv


load_dotenv()

def get_response(user_message):
    return "This is a placeholder response."

def get_vectorstore_from_url(url):
    # get webcontent in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(document)
    st.write(split_docs)
    st.session_state.vectorstore = split_docs
    st.success("Website content loaded successfully!")
    
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=CohereEmbeddings(model="embed-english-v3.0"),
    )
    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatCohere()
    
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            {"role": "user", "content": "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"}
        ]
    )
    
    return create_history_aware_retriever(
        retriever=retriever,
        llm=llm,
        prompt=prompt,
    )

def get_conversational_rag_chain(retriever_chain):
    llm = ChatCohere()
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            {"role": "system", "content": "Answer the user's question based on the retrieved context."},
            ("system", "Use the following context to answer the question: {context}"),
        ]
    )
    
    stuff_documents =  create_stuff_documents_chain(
        llm=llm,
        retriever=retriever_chain,
        return_source_documents=True,
    )
    
st.set_page_config(
    page_title="Website Chat",
    page_icon=":speech_balloon:"
)

st.title("Website Chat")

with st.sidebar:
    st.header("Chat Settings")
    website_url = st.text_input("Website URL")
    
    if not website_url:
        st.info("Please enter a website URL to load its content.")
    else:
        vector_store = get_vectorstore_from_url(website_url)
        retriever_chain = get_context_retriever_chain(vector_store)
        
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content="Hello! How can I assist you today?",
        ),
    ]
    
user_query = st.chat_input("Enter your message here.... ")

if user_query:
    bot_response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=bot_response))
    
    retrieved_documents = retriever_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    st.write(retrieved_documents)
    
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
    