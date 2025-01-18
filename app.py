import streamlit as st
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer, util

############################################
# Global Setup and Models
############################################
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

############################################
# Utility Functions
############################################

def parse_pdf(uploaded_pdf):
    """Extract text from a PDF."""
    text = ""
    try:
        reader = PdfReader(uploaded_pdf)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
    return text

def parse_docx(uploaded_docx):
    """Extract text from a DOCX file."""
    text = ""
    try:
        doc = Document(uploaded_docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error parsing DOCX: {e}")
    return text

def parse_image(uploaded_image):
    """Extract text from an image using OCR."""
    try:
        image = Image.open(uploaded_image)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error processing image for OCR: {e}")
        text = ""
    return text

def analyze_chunks(paragraphs, objective):
    """Analyze text chunks and calculate alignment scores."""
    objective_embedding = embedding_model.encode(objective, convert_to_tensor=True)
    data = []
    for i, para in enumerate(paragraphs):
        embedding = embedding_model.encode(para, convert_to_tensor=True)
        alignment_score = float(util.pytorch_cos_sim(embedding, objective_embedding)[0][0])
        word_count = len(para.split())
        data.append({
            "Chunk ID": i + 1,
            "Paragraph": para,
            "Word Count": word_count,
            "Alignment Score": alignment_score
        })
    return pd.DataFrame(data)

############################################
# App Workflow Functions
############################################

def page_metadata():
    st.title("Lesson Builder: Step 1 - Metadata")
    course_title = st.text_input("Course Title")
    module_title = st.text_input("Module Title")
    lesson_title = st.text_input("Lesson Title")
    lesson_objective = st.text_area("Lesson Objective")
    if st.button("Next"):
        st.session_state["metadata"] = {
            "Course Title": course_title,
            "Module Title": module_title,
            "Lesson Title": lesson_title,
            "Lesson Objective": lesson_objective
        }
        st.session_state["page"] = 1

def page_content():
    st.title("Lesson Builder: Step 2 - Content")
    st.write("Upload or enter content for textbook and SME.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Textbook Content")
        textbook_files = st.file_uploader(
            "Upload textbook files (PDF/DOCX/Images)", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)
        textbook_text = st.text_area("Or paste textbook content below")

        # Process files
        combined_textbook = ""
        if textbook_files:
            for file in textbook_files:
                if file.name.endswith(".pdf"):
                    combined_textbook += parse_pdf(file)
                elif file.name.endswith(".docx"):
                    combined_textbook += parse_docx(file)
                elif file.name.endswith((".png", ".jpg", ".jpeg")):
                    combined_textbook += parse_image(file)

    with col2:
        st.subheader("SME Content")
        sme_files = st.file_uploader(
            "Upload SME files (PDF/DOCX/Images)", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)
        sme_text = st.text_area("Or paste SME content below")

        # Process files
        combined_sme = ""
        if sme_files:
            for file in sme_files:
                if file.name.endswith(".pdf"):
                    combined_sme += parse_pdf(file)
                elif file.name.endswith(".docx"):
                    combined_sme += parse_docx(file)
                elif file.name.endswith((".png", ".jpg", ".jpeg")):
                    combined_sme += parse_image(file)

    if st.button("Next"):
        st.session_state["textbook_content"] = combined_textbook.strip() or textbook_text.strip()
        st.session_state["sme_content"] = combined_sme.strip() or sme_text.strip()
        st.session_state["page"] = 2

    if st.button("Back"):
        st.session_state["page"] = 0

def page_analyze():
    st.title("Lesson Builder: Step 3 - Analyze")
    metadata = st.session_state.get("metadata", {})
    objective = metadata.get("Lesson Objective", "")
    textbook_content = st.session_state.get("textbook_content", "")
    sme_content = st.session_state.get("sme_content", "")

    combined_content = (textbook_content + "\n" + sme_content).strip()
    if not combined_content:
        st.warning("No content available. Go back to Step 2.")
        return

    paragraphs = [p.strip() for p in combined_content.split("\n") if p.strip()]
    if st.button("Analyze Now"):
        analysis_df = analyze_chunks(paragraphs, objective)
        st.session_state["analysis_df"] = analysis_df
        st.success("Analysis complete!")
        st.dataframe(analysis_df)

    if st.button("Next"):
        st.session_state["page"] = 3
    if st.button("Back"):
        st.session_state["page"] = 1

def main():
    st.set_page_config(page_title="Lesson Builder", layout="wide")
    if "page" not in st.session_state:
        st.session_state["page"] = 0

    pages = [page_metadata, page_content, page_analyze]
    pages[st.session_state["page"]]()

if __name__ == "__main__":
    main()
