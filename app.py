import streamlit as st
import mysql.connector
import datetime
import pandas as pd

import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer, util

############################################
# 1) MySQL Config and Logging Functions
############################################

def get_db_connection():
    """
    Connect to MySQL DB using Streamlit secrets.
    """
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
    """
    Insert usage logs into the usage_logs table.
    """
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

def log_user_change_db(screen_index, old_row, new_row):
    """
    Log changes made by users to the database.
    """
    changes = {}
    for col in ["Screen Title", "Text", "Estimated Duration"]:
        old_val = old_row[col]
        new_val = new_row[col]
        if old_val != new_val:
            changes[col] = {"old": old_val, "new": new_val}

    if changes:
        log_entry = {
            "screen_index": screen_index,
            "changes": changes
        }
        log_usage_to_db(log_entry)

############################################
# 2) Advanced AI Setup
############################################

def load_embedding_model():
    """Load a sentence transformer model."""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def call_llm_for_summary(paragraph):
    """Stub for LLM-based summarization."""
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
    st.write("Upload PDF/DOCX/image or paste text for each category.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Textbook Content")
        tb_file = st.file_uploader("Upload PDF, DOCX, or image for textbook content",
                                   type=["pdf","docx","png","jpg","jpeg"])
        tb_text_fallback = st.text_area("Or paste textbook text below", height=150)

        textbook_parsed = parse_pdf(tb_file) if tb_file else tb_text_fallback.strip()

    with col2:
        st.subheader("SME Content")
        sme_file = st.file_uploader("Upload PDF, DOCX, or image for SME content",
                                    type=["pdf","docx","png","jpg","jpeg"])
        sme_text_fallback = st.text_area("Or paste SME text below", height=150)

        sme_parsed = parse_pdf(sme_file) if sme_file else sme_text_fallback.strip()

    include_video = st.checkbox("Include Concept Teaching Video?")

    if st.button("Next"):
        st.session_state["textbook_text"] = textbook_parsed
        st.session_state["sme_text"] = sme_parsed
        st.session_state["include_video"] = include_video
        st.session_state["page"] = 2

############################################
# Remaining Steps...
############################################

# Repeat for `page_analyze`, `page_generate`, and `page_refine`.

if __name__ == "__main__":
    main()
