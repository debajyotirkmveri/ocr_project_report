import os
import shutil
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import fitz  # PyMuPDF for image handling in PDFs
import pytesseract
from PIL import Image, UnidentifiedImageError
from io import BytesIO
# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google Generative AI
genai.configure(api_key=api_key)

# Create the generative model instance
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Dictionary for storing users (in-memory for demo purposes)
# In production, you would want to use a secure database
users = {
    "user1": hash_password("password1"),
    "user2": hash_password("password2"),
}

# User login system
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        hashed_password = hash_password(password)
        if username in users and users[username] == hashed_password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.sidebar.success(f"Welcome {username}!")
        else:
            st.sidebar.error("Incorrect username or password")

# Logout system
def logout():
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        st.sidebar.write(f"Logged in as {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None

# Function to extract text from multiple PDFs with document names and page numbers
def get_pdf_text_with_pages(pdf_paths):
    text_chunks_with_pages = []
    
    for pdf_path in pdf_paths:
        try:
            # Read the PDF for text extraction and image handling
            pdf_reader = PdfReader(pdf_path)
            pdf_document = fitz.open(pdf_path)  # Open PDF with PyMuPDF for image extraction
            doc_name = os.path.basename(pdf_path)
            
            # Iterate through each page
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                # Try to extract text directly from the page
                text = page.extract_text() or ""
                
                # Load page using PyMuPDF
                fitz_page = pdf_document.load_page(page_number - 1)
                images = fitz_page.get_images(full=True)
                
                # Initialize a variable to hold text extracted via OCR
                ocr_text = ""
                
                # If there are images on the page, apply OCR
                for img_index, img in enumerate(images):
                    xref = img[0]  # XREF for the image
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Try to load the image and apply OCR, with error handling for unsupported formats
                    try:
                        img_pil = Image.open(BytesIO(image_bytes))
                        ocr_text += pytesseract.image_to_string(img_pil, lang='eng')
                    except UnidentifiedImageError:
                        print(f"Warning: Unidentified image format on page {page_number}, image {img_index + 1} - skipping this image.")
                
                # Combine extracted text and OCR text, if available
                combined_text = text + "\n" + ocr_text if ocr_text else text
                
                # Append the text along with page number and document name
                if combined_text.strip():  # Only append if there's text
                    text_chunks_with_pages.append((combined_text, page_number, doc_name))
            
            # Close the PDF document after processing
            pdf_document.close()
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
    
    return text_chunks_with_pages

# Split text into chunks, including document name and page numbers
def get_text_chunks_with_pages(text_chunks_with_pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks_with_pages = []
    for text, page_number, doc_name in text_chunks_with_pages:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_pages.append((chunk, page_number, doc_name))
    return chunks_with_pages

# Create and save vector store with document names and page numbers
def get_vector_store_with_pages(text_chunks_with_pages, index_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    texts, page_numbers, doc_names = zip(*text_chunks_with_pages)
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    vector_store.save_local(index_name)
    # Save page numbers and document names to a file for later retrieval
    with open("page_numbers_docs.pkl", "wb") as f:
        pickle.dump((page_numbers, doc_names), f)

# Function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
        Answer the question in a concise and structured way using bullet points to ensure clarity and easy understanding. 
        Do not provide the answer in paragraph form. Use numbered lists for sub-points if applicable.   
        If the answer is not available in the provided context, simply state: "The answer is not available in the context."
        
        Context:\n {context}\n
        Question: \n{question}\n
        
        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input and get response with document names and pages
def user_input_with_page(user_question, index_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    new_db = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
    
    # Perform the similarity search
    search_results = new_db.similarity_search(user_question, return_scores=False)
    
    chain = get_conversational_chain()

    response = chain.invoke(
        {"input_documents": search_results, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

# Function to compute TF-IDF vectors for a list of documents
def compute_tfidf_vectors(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer

# Find sentences with a similarity score above the threshold, including document names and pages
def find_matching_sentences(result, text_chunks_with_pages, threshold=0.3):
    # Flatten the list of text chunks into sentences, along with page numbers and document names
    sentences_with_pages = [(sentence, page_number, doc_name)
                            for text_chunk, page_number, doc_name in text_chunks_with_pages
                            for sentence in re.split(r'(?<=[.!?]) +', text_chunk)]
    
    # Separate sentences, pages, and document names
    sentences, pages, doc_names = zip(*sentences_with_pages)
    
    # Compute TF-IDF vectors for all sentences
    tfidf_matrix, vectorizer = compute_tfidf_vectors(sentences)
    
    # Split the result into sentences
    result_sentences = re.split(r'(?<=[.!?]) +', result)
    
    # Compute TF-IDF vectors for result sentences
    result_tfidf_matrix = vectorizer.transform(result_sentences)
    
    # Calculate cosine similarity and find matches
    cosine_similarities = cosine_similarity(result_tfidf_matrix, tfidf_matrix)
    
    # Store matching sentences with pages and document names
    matching_sentences_pages = []
    for idx, similarities in enumerate(cosine_similarities):
        matching_indices = [i for i, score in enumerate(similarities) if score > threshold]
        matching_pages_docs = [(pages[i], doc_names[i]) for i in matching_indices]
        if matching_pages_docs:
            matching_sentences_pages.append((result_sentences[idx], set(matching_pages_docs)))
        else:
            matching_sentences_pages.append((result_sentences[idx], {("Page not found", "Doc not found")}))
    
    return matching_sentences_pages

# Ensure 'temp' directory exists before saving uploaded files
def ensure_temp_directory():
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

# Streamlit UI
def main():
    st.set_page_config(page_title='Multi-PDF QA Chatbot', layout="wide", page_icon="📄")

    # Display the EY logo in the right upper corner
    logo_url = "https://assets.ey.com/content/dam/ey-sites/ey-com/en_gl/topics/innovation-realized/ey-ey-stacked-logo.jpg"
    st.markdown(
        f"""
        <style>
        .logo-container {{
            position: absolute;
            top: 10px;
            right: 10px;
        }}
        .login-container {{
            background-image: url('https://static.wixstatic.com/media/9d7b99_dfcb8e88751c4cecb7ac677976976ec8~mv2.gif');
            background-size: cover;
            background-position: center;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        </style>
        <div class="logo-container">
            <img src="{logo_url}" alt="EY Logo" width="100">
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Handle login or logout logic
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        login()
        st.markdown('</div>', unsafe_allow_html=True)
        return
    else:
        logout()

    st.header('Multi-PDF Content-Based Question Answering System')

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Define the submit function
    def submit():
        user_question = st.session_state.user_question
        if user_question:  # Only process if there is a question
            # Count input tokens
            input_tokens = model.count_tokens(user_question)

            # Process the question and get the answer
            result = user_input_with_page(user_question, index_name="faiss_index")
            
            # Count output tokens using the generated content
            output_tokens = model.count_tokens(result)

            if "The answer is not available in the context." in result:
                st.session_state['chat_history'].append({
                    'question': user_question,
                    'response': result,
                    'pages': None,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                })
            else:
                st.session_state['chat_history'].append({
                    'question': user_question,
                    'response': result,
                    'pages': find_matching_sentences(result, text_chunks_with_pages),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                })

            st.session_state.user_question = ""

    # Display chat history
    st.subheader("Chat History")
    for entry in st.session_state['chat_history']:
        st.write(f"🧑 User Question: {entry['question']} (Input tokens: {entry['input_tokens']})")
        st.write(f"🤖 Bot Answer: {entry['response']} (Output tokens: {entry['output_tokens']})")
        st.write("📖 Source:")

        if entry.get('pages') and "The answer is not available in the context." not in entry['response']:
            for sentence, pages_docs_set in entry['pages']:
                if sentence.strip():
                    st.write(f"\"{sentence}\" appears in: {', '.join([f'Document: {doc}, Page: {page}' for page, doc in pages_docs_set])}")

    # Sidebar file uploader
    uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        # Ensure temp directory exists before saving
        temp_dir = ensure_temp_directory()

        # Save uploaded files in the temporary directory
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)

        # Process the PDF files
        text_chunks_with_pages = get_pdf_text_with_pages(saved_files)

        if text_chunks_with_pages:
            chunks_with_pages = get_text_chunks_with_pages(text_chunks_with_pages)
            get_vector_store_with_pages(chunks_with_pages, index_name="faiss_index")

            st.sidebar.success("PDF files processed and indexed successfully.")

    # Question input area
    st.text_input("Ask a question:", key="user_question", on_change=submit)

if __name__ == '__main__':
    main()
