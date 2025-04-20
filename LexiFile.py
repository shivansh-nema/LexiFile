# LexiFile : an AI powered file based ChatBot

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# headers = {
#     "authorization": st.secrets["API_KEY"],
    
# }
headers = {
    "authorization": st.secrets[API_KEY],
    "authorization": st.secrets[model_name],
    "authorization": st.secrets[project_id],
    "content-type": "application/json"
}

st.title("LexiFile")
st.write("Turn your files into interactive knowledge. Ask it anything about your documents, and it'll provide AI-driven answers. ")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file here,", type="pdf")

if file is not None:
    pdf_Reader = PdfReader(file)
    text = ""
    for page in pdf_Reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)

    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Type your question here :-   ")

    if user_question:
        match = vector_store.similarity_search(user_question)

        llm = ChatGoogleGenerativeAI(
            api_key=API_KEY,
            temperature=1,
            max_tokens=1000,
            model= model_name,
            project= project_id
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)
