import streamlit as st
import mysql.connector
import datetime
import pandas as pd

import PyPDF2
from docx import Document

# For embedding-based chunk alignment
from sentence_transformers import SentenceTransformer, util


############################################
# 1) MySQL Config and Logging Functions
############################################

def get_db_connection():
    """
    Connect to your MySQL DB using Streamlit secrets.
    This securely fetches the database credentials.
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
    Insert usage logs into the usage_logs table (id, log_time, screen_index, changes TEXT).
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


############################################
# 2) Advanced AI Setup
############################################

def load_embedding_model():
    """
    Load a small sentence-transformer for chunk alignment checks.
    """
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


embedding_model = load_embedding_model()


def call_llm_for_summary(paragraph):
    """
    Stub for LLM-based summarization. Must ensure no new facts are introduced.
    """
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

    tb_file = st.file_uploader("Upload Textbook File (PDF/DOCX)", type=["pdf", "docx"])
    sme_file = st.file_uploader("Upload SME File (PDF/DOCX)", type=["pdf", "docx"])

    tb_text = parse_pdf(tb_file) if tb_file else st.text_area("Or paste textbook content", height=150)
    sme_text = parse_docx(sme_file) if sme_file else st.text_area("Or paste SME content", height=150)

    if st.button("Next"):
        st.session_state["textbook_text"] = tb_text
        st.session_state["sme_text"] = sme_text
        st.session_state["page"] = 2


if __name__ == "__main__":
    main()
