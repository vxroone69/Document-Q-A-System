# Document Q&A System

This Streamlit application allows users to upload a PDF document and ask questions about its content. It leverages advanced NLP techniques to extract relevant information and provide accurate answers.

## Objective

The primary objective of this project is to create a user-friendly interface for querying information from PDF documents. Users can upload a PDF, and the system will process the document to enable question-answering capabilities. This aims to simplify information retrieval from large documents, saving time and effort.

## Pipeline

The application follows a structured pipeline:

1.  **Document Upload:**
    * Users upload a PDF document through a Streamlit file uploader.
2.  **Document Processing:**
    * The uploaded PDF is temporarily saved.
    * The PDF is loaded and split into smaller chunks using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
    * Text chunks are embedded using `HuggingFaceEmbeddings` (sentence-transformers/all-MiniLM-L6-v2).
    * A vectorstore (`Chroma`) is created to store the embeddings for efficient similarity search.
3.  **Question Answering:**
    * Users enter their questions via a text input.
    * The question is used to perform a similarity search in the vectorstore, retrieving relevant document chunks.
    * The retrieved context and the question are combined and fed into the `google/flan-t5-base` model.
    * The model generates an answer based on the context and question.
    * The answer and the relevant document chunks are displayed to the user.

## API and Model Used

* **Streamlit:**
    * Used to create the web interface and handle user interactions.
* **Hugging Face Transformers:**
    * `google/flan-t5-base`: A text-to-text transfer transformer model used for question answering.
    * `sentence-transformers/all-MiniLM-L6-v2`: Used for creating document embeddings.
* **LangChain:**
    * `PyPDFLoader`: Loads the PDF document.
    * `RecursiveCharacterTextSplitter`: Splits the document into smaller chunks.
    * `Chroma`: Creates and manages the vectorstore for similarity search.
    * `HuggingFaceEmbeddings`: Used to create the embeddings.
* **PyPDFLoader:** used to load the pdf document.
* **Pytorch:** Used to handle the machine learning models.

## Installation

1.  Clone the repository:

    ```bash
    git clone [repository_url]
    cd [repository_directory]
    ```

2.  Install the required packages:

    ```bash
    pip install streamlit transformers langchain chromadb sentence-transformers pypdf torch
    ```

3.  Run the Streamlit application:

    ```bash
    streamlit run doc_qa_app.py
    ```

## Usage

1.  Open the Streamlit application in your web browser.
2.  Upload a PDF document using the file uploader.
3.  Click the "Process Document" button.
4.  Once the document is processed, enter your question in the text input.
5.  The application will display the answer and the relevant document chunks.

## Notes

* Ensure you have a stable internet connection for downloading the models.
* The processing time may vary depending on the size of the document.
* The application will try to use MPS if available, otherwise it will default to CPU.
* Consider using a virtual environment.
