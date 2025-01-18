import streamlit as st
import mysql.connector
import datetime
import pandas as pd
import pytesseract
from PIL import Image
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer, util
import subprocess

############################################
# 1) MySQL Config and Logging Functions
############################################

def get_db_connection():
    conn = None
    try:
        conn = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"],
            port=st.secrets["mysql"]["port"],
        )
    except Exception as e:
        st.error(f"Could not connect to MySQL: {e}")
    return conn

def log_usage_to_db(log_entry):
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
# 2) OCR and File Parsing
############################################

# Set Tesseract path
try:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    tesseract_version = subprocess.run(
        ["tesseract", "--version"], capture_output=True, text=True
    )
    st.write(f"Tesseract Version: {tesseract_version.stdout}")
except Exception as e:
    st.error("Tesseract is not properly installed or configured. Contact admin.")

def parse_pdf(uploaded_pdf):
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_pdf)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def parse_docx(uploaded_docx):
    text = ""
    try:
        doc = Document(uploaded_docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return text

def parse_image_ocr(uploaded_image):
    text = ""
    try:
        image = Image.open(uploaded_image)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
    return text

############################################
# 3) Advanced AI Setup
############################################

def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

def analyze_chunks_with_llm(paragraphs, objective):
    data = []
    objective_emb = embedding_model.encode(objective, convert_to_tensor=True)
    for i, para in enumerate(paragraphs):
        para_emb = embedding_model.encode(para, convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(para_emb, objective_emb)[0][0])
        word_count = len(para.split())
        data.append({
            "chunk_id": i+1,
            "original_paragraph": para,
            "alignment_score": round(sim_score, 3),
            "word_count": word_count,
        })
    return pd.DataFrame(data)

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

def show_sidebar_progress():
    st.sidebar.title("Lesson Steps Progress")
    steps = ["Metadata", "Content", "Analyze"]
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
    course_title = st.text_input("Course Title", "")
    lesson_objective = st.text_input("Lesson Objective", "")
    if st.button("Next"):
        st.session_state["metadata"] = {
            "Course Title": course_title,
            "Lesson Objective": lesson_objective,
        }
        st.session_state["page"] = 1

############################################
# Page 1: Content
############################################

def page_content():
    st.title("Lesson Builder: Step 2 - Content")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Textbook Content")
        tb_files = st.file_uploader(
            "Upload PDF, DOCX, or image files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True
        )
        tb_text_fallback = st.text_area("Or paste textbook text below", height=150)

    with col2:
        st.subheader("SME Content")
        sme_files = st.file_uploader(
            "Upload PDF, DOCX, or image files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True
        )
        sme_text_fallback = st.text_area("Or paste SME text below", height=150)

    if st.button("Process Content"):
        tb_text = ""
        sme_text = ""

        for file in tb_files:
            ext = file.name.split(".")[-1].lower()
            if ext == "pdf":
                tb_text += parse_pdf(file)
            elif ext == "docx":
                tb_text += parse_docx(file)
            elif ext in ["png", "jpg", "jpeg"]:
                tb_text += parse_image_ocr(file)

        for file in sme_files:
            ext = file.name.split(".")[-1].lower()
            if ext == "pdf":
                sme_text += parse_pdf(file)
            elif ext == "docx":
                sme_text += parse_docx(file)
            elif ext in ["png", "jpg", "jpeg"]:
                sme_text += parse_image_ocr(file)

        st.session_state["textbook_text"] = tb_text.strip() or tb_text_fallback.strip()
        st.session_state["sme_text"] = sme_text.strip() or sme_text_fallback.strip()
        st.success("Content uploaded and processed!")
        st.session_state["page"] = 2

############################################
# Page 2: Analyze
############################################

def page_analyze():
    st.title("Lesson Builder: Step 3 - Analyze")
    objective = st.session_state["metadata"].get("Lesson Objective", "")
    combined_text = (
        st.session_state.get("textbook_text", "") + "\n" + st.session_state.get("sme_text", "")
    ).strip()
    if not combined_text:
        st.warning("No content found. Please upload content in the previous step.")
        return

    if st.button("Analyze Content"):
        paragraphs = [p for p in combined_text.split("\n") if p]
        df_analysis = analyze_chunks_with_llm(paragraphs, objective)
        st.session_state["analysis_df"] = df_analysis
        st.write("Analysis complete!")
        st.dataframe(df_analysis)

############################################

if __name__ == "__main__":
    main()
