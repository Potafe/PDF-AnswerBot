# PDF AnswerBot

PDF AnswerBot is an advanced document processing and chatbot application designed to intelligently search and extract information from PDF documents. Leveraging state-of-the-art language models and embedding techniques, this project provides a seamless experience for uploading PDFs, querying their contents, and obtaining insightful answers.
 
## Features

- **PDF Uploading:** Easily upload PDF documents for processing.
- **Intelligent Querying:** Ask questions and get accurate answers based on the content of the uploaded documents.
- **Document Chunking:** Efficiently splits documents into manageable chunks for better processing and retrieval.
- **Persistent Storage:** Stores document embeddings for quick and efficient retrieval.
- **Flask Interface:** User-friendly web interface for interacting with the application.
- **Local Processing:** Ensures privacy and security by processing documents locally without the need for internet access.

## Advantages

- **Privacy and Security:** By using a local language model, users can safely upload their personal and private PDFs without worrying about data breaches or third-party access. All document processing and querying happen on your local machine, ensuring complete control over your data.
- **Efficiency:** Fast and efficient processing of documents with local embeddings and retrieval.
- **User-Friendly:** Simple and intuitive interface for uploading documents and querying information.

## Technologies Used

- **Python**: Core programming language.
- **Flask**: For building the web application.
- **Ollama**: Local language model for processing queries.
- **FastEmbedEmbeddings**: For generating document embeddings.
- **PDFPlumberLoader**: For loading and splitting PDF documents.
- **Chroma**: Vector store for embedding storage and retrieval.

## Installation

### Prerequisites

- Python 3.7+
- Pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/Potafe/PDF-AnswerBot.git
cd PDF-AnswerBot
```
### Package Installations

```bash
pip install -r requirements.txt
```

### Usage
Running the Application
To start the Flask application, run the following command:

```bash
streamlit run app.py
```
### Uploading a PDF
- Open the application in your web browser.
- Navigate to the "Upload PDF" section.
- Choose a PDF file from your local machine and upload it.
- The application will process the PDF and store the document embeddings.

### Asking a Query
- Navigate to the "Ask Query" section.
- Enter your query in the text input field.
- Click the "Get Answer" button.
- The application will retrieve the most relevant information from the uploaded documents and display the answer along with the sources.
