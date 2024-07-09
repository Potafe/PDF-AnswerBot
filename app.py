import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains import create_retrieval_chain
from langchain_core.prompts.prompt import PromptTemplate
import os
import time


folder_path = "db"
pdf_folder = "pdf"

# Ensure the PDF folder exists
os.makedirs(pdf_folder, exist_ok=True)

cached_llm = Ollama(model = "llama3:8b")
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 250, chunk_overlap = 100, length_function = len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] 
        {input}
        Context: {context}
        Answer:
    [/INST]
    """
)

st.title("Document Search and Chatbot")

def upload_pdf():
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        save_file = f"{pdf_folder}/{uploaded_file.name}"
        with open(save_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Successfully uploaded {uploaded_file.name}")

        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()
        st.write(f"Number of documents loaded: {len(docs)}")

        chunks = text_splitter.split_documents(docs)
        st.write(f"Number of chunks created: {len(chunks)}")

        vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=folder_path)
        vector_store.persist()

        st.success("Document processing complete and data persisted.")
    else:
        st.warning("Please upload a PDF file.")

def show_uploaded_pdfs():
    st.subheader("Uploaded PDFs")
    pdf_files = os.listdir(pdf_folder)
    if pdf_files:
        for pdf in pdf_files:
            if st.button(pdf):
                st.write(f"Displaying contents of {pdf}:")
                pdf_path = f"{pdf_folder}/{pdf}"
                loader = PDFPlumberLoader(pdf_path)
                docs = loader.load_and_split()
                for doc in docs:
                    st.write(doc.page_content)
    else:
        st.write("No PDFs uploaded yet.")

def ask_query():
    query = st.text_input("Enter your query")
    if st.button("Get Answer"):
        if query:
            with st.spinner("Assistant is thinking...."):
                vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
                retriever = vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 2, "score_threshold": 0.1}
                )

                document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
                chain = create_retrieval_chain(retriever, document_chain)

                result = chain.invoke({"input": query})

            answer_placeholder = st.empty()
            answer = result["answer"]
            typed_answer = ""
            for char in answer:
                typed_answer += char
                answer_placeholder.write(typed_answer)
                time.sleep(0.05)

            sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]
            st.json({"answer": result["answer"], "sources": sources})
        else:
            st.warning("Please enter a query.")

st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Upload PDF", "Ask Query", "Show Uploaded PDFs"])

if options == "Upload PDF":
    upload_pdf()
elif options == "Ask Query":
    ask_query()
elif options == "Show Uploaded PDFs":
    show_uploaded_pdfs()
