import streamlit as st
import mysql.connector
import datetime
import pandas as pd
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import pytesseract

############################################
# 1) MySQL Config and Logging Functions
############################################

def get_db_connection():
    """Connect to your Hostinger MySQL DB."""
    try:
        conn = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"],
            port=st.secrets["mysql"]["port"]
        )
        return conn
    except Exception as e:
        st.error(f"Could not connect to MySQL: {e}")
        return None

def log_usage_to_db(log_entry):
    """Insert usage logs into usage_logs table."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        changes_str = str(log_entry.get("changes", {}))
        log_time = datetime.datetime.now()

        sql = """
        INSERT INTO usage_logs (log_time, screen_index, changes)
        VALUES (%s, %s, %s)
        """
        cursor.execute(sql, (log_time, log_entry.get("screen_index"), changes_str))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error inserting usage log: {e}")


############################################
# 2) Advanced AI Setup
############################################

def load_embedding_model():
    """Load a small sentence-transformer for chunk alignment checks."""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def call_llm_for_summary(paragraph):
    """Stub for LLM-based summarization (no new content)."""
    return "LLM summary placeholder (no new content)."


############################################
# 3) File Parsing
############################################

def parse_pdf(uploaded_pdf):
    """Extract text from PDF using PyPDF2."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def parse_docx(uploaded_docx):
    """Extract text from .docx using python-docx."""
    text = ""
    try:
        doc = Document(uploaded_docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return text

def parse_image_ocr(uploaded_image):
    """Extract text from an image file using OCR."""
    try:
        image = Image.open(uploaded_image)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return ""


############################################
# 4) Multi-Step Flow
############################################

def main():
    st.set_page_config(page_title="Advanced Lesson Builder", layout="wide")

    if "page" not in st.session_state:
        st.session_state["page"] = 0
    if "screens_df" not in st.session_state:
        st.session_state["screens_df"] = pd.DataFrame()

    show_sidebar_progress()

    page = st.session_state["page"]
    if page == 0:
        page_metadata()
    elif page == 1:
        page_content()
    elif page == 2:
        page_analyze()
    elif page == 3:
        page_generate()
    elif page == 4:
        page_refine()
    else:
        st.write("Invalid page index!")

def show_sidebar_progress():
    st.sidebar.title("Lesson Steps Progress")
    steps = ["Metadata", "Content", "Analyze", "Generate", "Refine"]
    current_page = st.session_state["page"]
    for i, step_name in enumerate(steps):
        if i < current_page:
            st.sidebar.write(f"✅ {step_name} - Done")
        elif i == current_page:
            st.sidebar.write(f"▶️ {step_name} - In Progress")
        else:
            st.sidebar.write(f"⬜ {step_name} - Pending")


############################################
# Page 0: Metadata
############################################

def page_metadata():
    st.title("Lesson Builder: Step 1 - Metadata")
    course_title = st.text_input("Course # and Title", "")
    module_title = st.text_input("Module # and Title", "")
    unit_title = st.text_input("Unit # and Title", "")
    lesson_title = st.text_input("Lesson # and Title", "")
    lesson_objective = st.text_input("Lesson Objective", "")
    lesson_type = st.selectbox("Lesson Type", ["Core Learning Lesson", "Practice Lesson", "Other"])

    if st.button("Next"):
        st.session_state["metadata"] = {
            "Course and Title": course_title,
            "Module and Title": module_title,
            "Unit and Title": unit_title,
            "Lesson and Title": lesson_title,
            "Lesson Objective": lesson_objective,
            "Lesson Type": lesson_type
        }
        st.session_state["page"] = 1


############################################
# Page 1: Content
############################################

def page_content():
    st.title("Lesson Builder: Step 2 - Content")
    st.write("Upload multiple files (PDFs, DOCX, or images) or paste text for each category.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Textbook Content")
        textbook_files = st.file_uploader(
            "Upload multiple files for textbook content",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="textbook_files"
        )
        tb_text_fallback = st.text_area("Or paste textbook text below", height=150)

    with col2:
        st.subheader("SME Content")
        sme_files = st.file_uploader(
            "Upload multiple files for SME content",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="sme_files"
        )
        sme_text_fallback = st.text_area("Or paste SME text below", height=150)

    # Process uploaded files
    textbook_parsed = process_files(textbook_files)
    sme_parsed = process_files(sme_files)

    final_textbook = (textbook_parsed + "\n" + tb_text_fallback).strip()
    final_sme = (sme_parsed + "\n" + sme_text_fallback).strip()

    include_video = st.checkbox("Include Concept Teaching Video?")

    if st.button("Next"):
        st.session_state["textbook_text"] = final_textbook
        st.session_state["sme_text"] = final_sme
        st.session_state["include_video"] = include_video
        st.session_state["page"] = 2

def process_files(files):
    """Process multiple uploaded files and extract text."""
    all_text = ""
    for uploaded_file in files:
        if uploaded_file is not None:
            filename = uploaded_file.name.lower()
            if filename.endswith(".pdf"):
                all_text += parse_pdf(uploaded_file) + "\n"
            elif filename.endswith(".docx"):
                all_text += parse_docx(uploaded_file) + "\n"
            elif filename.endswith((".png", ".jpg", ".jpeg")):
                all_text += parse_image_ocr(uploaded_file) + "\n"
            else:
                st.warning(f"Unsupported file type: {filename}")
    return all_text.strip()


############################################
# Other Steps (Analyze, Generate, Refine)
############################################

# ... Remaining steps (page_analyze, page_generate, page_refine) remain unchanged ...
# Paste the previous code for these steps here

############################################
# Main Execution
############################################

if __name__ == "__main__":
    main()
