import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains import create_retrieval_chain
from langchain_core.prompts.prompt import PromptTemplate

folder_path = "db"

cached_llm = Ollama(model = "llama3:8b")
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 250, chunk_overlap = 100, length_function = len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
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
        save_file = f"pdf/{uploaded_file.name}"
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

def ask_query():
    query = st.text_input("Enter your query")
    if st.button("Get Answer"):
        if query:
            vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 10, "score_threshold": 0.1}
            )

            document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
            chain = create_retrieval_chain(retriever, document_chain)

            result = chain.invoke({"input": query})
            st.write(result)

            sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]
            st.json({"answer": result["answer"], "sources": sources})
        else:
            st.warning("Please enter a query.")

st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Upload PDF", "Ask Query"])

if options == "Upload PDF":
    upload_pdf()
elif options == "Ask Query":
    ask_query()
