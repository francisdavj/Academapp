import streamlit as st
import pandas as pd
import pytesseract
from pytesseract import Output
from PIL import Image
import os
import fitz  # PyMuPDF for PDF processing

# Set Streamlit page configuration
st.set_page_config(page_title="Advanced Lesson Builder", layout="wide")

# Sidebar: File Upload
st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "pdf"], help="Upload an image or PDF file.")

# Title and Description
st.title("Welcome to the Advanced Lesson Builder!")
st.write("Upload your files and build interactive lessons easily.")

def extract_metadata(file_path):
    """Extract metadata from PDF using PyMuPDF."""
    doc = fitz.open(file_path)
    metadata = doc.metadata
    doc.close()
    return metadata

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image, lang="eng")

def display_pdf(file_path):
    """Display PDF in Streamlit."""
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap()
            st.image(pix.tobytes(), caption=f"Page {page_num}", use_column_width=True)

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    save_path = os.path.join("temp", uploaded_file.name)
    
    # Save the uploaded file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write(f"File '{uploaded_file.name}' uploaded successfully!")
    
    if file_extension in ["png", "jpg", "jpeg"]:
        # Process images
        image = Image.open(save_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Extract text
        with st.spinner("Extracting text..."):
            text = extract_text_from_image(image)
        st.text_area("Extracted Text", text, height=200)
    
    elif file_extension == "pdf":
        # Process PDFs
        st.write("Processing PDF...")
        metadata = extract_metadata(save_path)
        st.subheader("Metadata:")
        st.json(metadata)
        
        # Display PDF
        st.subheader("PDF Preview:")
        display_pdf(save_path)
    
    else:
        st.error("Unsupported file type!")
else:
    st.info("Upload a file to begin.")

# Clean up temporary files (Optional, for long-term stability)
if os.path.exists("temp"):
    for f in os.listdir("temp"):
        os.remove(os.path.join("temp", f))
