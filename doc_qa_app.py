import os
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure page title and layout
st.set_page_config(page_title="Document Q&A System", layout="wide")
st.title('📚 Document Q&A System')
st.write('Upload a document and ask questions about its content')

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# Initialize session state variables if they don't exist
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = None

# Determine the appropriate device (MPS or CPU)
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

st.session_state.device = get_device()
st.write(f"Using device: {st.session_state.device}")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model.to(st.session_state.device)  # Move model to the appropriate device
    return tokenizer, model

st.session_state.tokenizer, st.session_state.model = load_model()

# Process the uploaded document
if uploaded_file is not None:
    # Create a temporary file to store the uploaded PDF
    with open("temp_document.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Process button
    if not st.session_state.processed and st.button("Process Document"):
        with st.spinner("Processing document..."):
            try:
                # Load and split the PDF
                loader = PyPDFLoader("temp_document.pdf")
                pages = loader.load()
                
                # Split the document into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                docs = text_splitter.split_documents(pages)
                
                # Initialize the embedding model
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Create the vectorstore
                vectorstore = Chroma.from_documents(
                    docs, 
                    embeddings, 
                    collection_name="document_qa"
                )
                
                # Save vectorstore to session state
                st.session_state.vectorstore = vectorstore
                st.session_state.processed = True
                
                st.success(f"Document processed! ({len(docs)} chunks created)")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

# Query section - only show if document is processed
if st.session_state.processed and st.session_state.vectorstore is not None:
    # Create a text input for the user's question
    query = st.text_input("Ask a question about the document:")
    
    # Process the query when submitted
    if query:
        with st.spinner("Finding answer..."):
            try:
                # Retrieve relevant documents
                docs = st.session_state.vectorstore.similarity_search(query, k=3)
                
                # Combine the content of the retrieved documents
                context = " ".join([doc.page_content for doc in docs])
                
                # Prepare the input for the model
                input_text = f"Context: {context} Question: {query}"
                inputs = st.session_state.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                inputs = {key: value.to(st.session_state.device) for key, value in inputs.items()}  # Move inputs to the appropriate device
                
                # Generate the answer
                outputs = st.session_state.model.generate(
                    **inputs,
                    max_new_tokens=512,  # Adjust this value as needed
                    num_beams=3,
                    early_stopping=True
                )
                answer = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Display the answer
                st.write("### Answer")
                st.write(answer)
                
                # Show the relevant document chunks
                with st.expander("View relevant document sections"):
                    for i, doc in enumerate(docs):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(doc.page_content)
                        st.write("---")
            except Exception as e:
                st.error(f"Error answering question: {str(e)}")
