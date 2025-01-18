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
    Connect to MySQL database.
    """
    try:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            user="u628260032_francisdavid",
            password="Chennai@202475",
            database="u628260032_academapp",
            port=3306
        )
        return conn
    except Exception as e:
        st.error(f"Could not connect to MySQL: {e}")
        return None

def log_usage_to_db(log_entry):
    """
    Log changes to the database.
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
    Track changes in the user edits and log them.
    """
    changes = {}
    for col in ["Screen Title", "Text", "Estimated Duration"]:
        old_val = old_row[col]
        new_val = new_row[col]
        if old_val != new_val:
            changes[col] = {"old": old_val, "new": new_val}

    if changes:
        log_entry = {"screen_index": screen_index, "changes": changes}
        log_usage_to_db(log_entry)

############################################
# 2) Advanced AI Setup
############################################
def load_embedding_model():
    """
    Load embedding model for content alignment.
    """
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def call_llm_for_summary(paragraph):
    """
    Placeholder for LLM summarization.
    """
    return "LLM summary placeholder."

############################################
# 3) File Parsing
############################################
def parse_pdf(uploaded_pdf):
    """
    Extract text from PDF.
    """
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_pdf)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def parse_docx(uploaded_docx):
    """
    Extract text from DOCX files.
    """
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
    """
    Display the sidebar for progress tracking.
    """
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
# Page Implementations
############################################
def page_metadata():
    """
    Step 1: Collect metadata.
    """
    st.title("Lesson Builder: Step 1 - Metadata")
    metadata_inputs = {
        "Course and Title": st.text_input("Course # and Title", ""),
        "Module and Title": st.text_input("Module # and Title", ""),
        "Unit and Title": st.text_input("Unit # and Title", ""),
        "Lesson and Title": st.text_input("Lesson # and Title", ""),
        "Lesson Objective": st.text_input("Lesson Objective", ""),
        "Lesson Type": st.selectbox("Lesson Type", ["Core Learning Lesson", "Practice Lesson", "Other"])
    }

    if st.button("Next"):
        st.session_state["metadata"] = metadata_inputs
        st.session_state["page"] = 1

def page_content():
    """
    Step 2: Upload and parse content files.
    """
    st.title("Lesson Builder: Step 2 - Content")
    st.write("Upload textbook and SME files or paste text.")

    # Textbook content
    textbook_parsed = handle_file_upload("Textbook Content", "textbook_file")
    sme_parsed = handle_file_upload("SME Content", "sme_file")

    if st.button("Next"):
        st.session_state["textbook_text"] = textbook_parsed
        st.session_state["sme_text"] = sme_parsed
        st.session_state["page"] = 2

def handle_file_upload(section, key):
    """
    Helper function to handle file uploads and parsing.
    """
    st.subheader(section)
    uploaded_file = st.file_uploader(f"Upload a file for {section}", type=["pdf", "docx"], key=key)
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            return parse_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            return parse_docx(uploaded_file)
    return ""

def page_analyze():
    """
    Step 3: Analyze chunks for alignment and summarization.
    """
    st.title("Lesson Builder: Step 3 - Analyze")
    objective = st.session_state.get("metadata", {}).get("Lesson Objective", "")
    combined_text = st.session_state.get("textbook_text", "") + st.session_state.get("sme_text", "")

    if not combined_text:
        st.warning("No content found. Please go back to Step 2.")
        return

    if st.button("Analyze Now"):
        paragraphs = [p for p in combined_text.split("\n") if p.strip()]
        df_analysis = analyze_chunks_with_llm(paragraphs, objective)
        st.session_state["analysis_df"] = df_analysis

    if "analysis_df" in st.session_state:
        st.dataframe(st.session_state["analysis_df"])

def analyze_chunks_with_llm(paragraphs, objective):
    """
    Analyze alignment of content to objective.
    """
    data = []
    objective_emb = embedding_model.encode(objective, convert_to_tensor=True)
    for i, para in enumerate(paragraphs):
        para_emb = embedding_model.encode(para, convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(para_emb, objective_emb)[0][0])
        data.append({
            "Chunk ID": i + 1,
            "Paragraph": para,
            "Alignment Score": round(sim_score, 3)
        })
    return pd.DataFrame(data)

############################################
# Final Steps and Output Generation
############################################
# Follow the same modular approach for steps 4 and 5 (generation and refinement).
# Add UI components for user edits, change logging, and final outputs.

if __name__ == "__main__":
    main()
