import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

st.set_page_config(page_title="Website Chatbot", page_icon=":speech_balloon:")
st.title("Website-Based Conversational Assistant")


def load_vectorstore(url: str):
    loader = WebBaseLoader(url)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    st.session_state["doc_chunks"] = chunks

    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

    st.success("Website content loaded successfully!")
    return vectorstore


def build_retriever_chain(vectorstore):
    llm = ChatCohere()
    retriever = vectorstore.as_retriever()

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    return create_history_aware_retriever(llm=llm, retriever=retriever, prompt=retriever_prompt)


def build_rag_chain(retriever_chain):
    llm = ChatCohere()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following context to answer the question: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", "Answer the user's question based on the retrieved context."),
        ("user", "{input}")
    ])

    combine_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
    return create_retrieval_chain(retriever_chain, combine_chain)


def generate_response(query: str):
    retriever_chain = build_retriever_chain(st.session_state.vectorstore)
    rag_chain = build_rag_chain(retriever_chain)

    response = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": query
    })

    return response["answer"]


# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    site_url = st.text_input("Website URL")

    if site_url:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello! How can I help you today?")
            ]

        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = load_vectorstore(site_url)
    else:
        st.info("Enter a website URL to get started.")


# Chat UI
user_input = st.chat_input("Type your message here...")

if user_input:
    response_text = generate_response(user_input)

    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=response_text)
    ])

for msg in st.session_state.get("chat_history", []):
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)
