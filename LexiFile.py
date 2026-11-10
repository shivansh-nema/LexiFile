# LexiFile : an AI powered file based ChatBot
import io
import logging
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


@st.cache_resource(show_spinner="Processing your document...")
def create_vector_store(pdf_bytes, api_key):

    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))

    text = ""

    for page in pdf_reader.pages:
        extracted_text = page.extract_text() or ""
        text += extracted_text + "\n"

    if not text.strip():
        raise ValueError(
            "No readable text found in this PDF."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    if not chunks:
        raise ValueError("No text chunks were generated.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    return vector_store

st.set_page_config(
    page_title="LexiFile",
    page_icon="LexiFile_Favicon.png",
)

api_key = st.secrets["API_KEY"]
headers = {
    "authorization": api_key,
    "content-type": "application/json"
}
model_name = "gemini-3.8-flash"
project_id = "323423126135"

st.title("LexiFile")
st.write("Turn your files into interactive knowledge. Ask it anything about your documents, and it'll provide AI-driven answers. ")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file here,", type="pdf")


if file is not None:

    try:
        pdf_bytes = file.getvalue()

        vector_store = create_vector_store(
            pdf_bytes,
            api_key
        )

        user_question = st.text_input(
            "Type your question here:"
        )

        if user_question:

            match = vector_store.similarity_search(
                user_question,
                k=4
            )

            context = "\n\n".join(
                doc.page_content for doc in match
            )

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.2,
                max_output_tokens=1000
            )

            prompt = f"""
            Answer the question using only the context below.

            If the answer is not present in the context,
            say that you could not find the answer
            in the document.

            Context:
            {context}

            Question:
            {user_question}

            Answer:
            """

            response = llm.invoke(prompt)

            st.write(response.content)

    except Exception as e:

        logging.exception(
            "LexiFile processing failed"
        )

        st.error(
            "Something went wrong while processing "
            "your document. Check the application logs."
        )
