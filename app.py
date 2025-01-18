import streamlit as st
import mysql.connector
import datetime
import pandas as pd
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer, util


############################################
# MySQL Config and Logging Functions
############################################
def get_db_connection():
    host = "127.0.0.1"
    user = "u628260032_francisdavid"
    password = "Chennai@202475"
    database = "u628260032_academapp"
    port = 3306
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )
        return conn
    except Exception as e:
        st.error(f"Could not connect to MySQL: {e}")
        return None


############################################
# Advanced AI Setup
############################################
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()


############################################
# File Parsing
############################################
def parse_pdf(uploaded_pdf):
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
    text = ""
    try:
        doc = Document(uploaded_docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return text


############################################
# Multi-Step Flow
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
    st.write("Upload multiple files (PDF/DOCX/images) or paste text for textbook and SME content.")

    col1, col2 = st.columns(2)

    # Placeholder for extracted text
    final_textbook = ""
    final_sme = ""

    ####################################
    # Textbook Content Upload and Parsing
    ####################################
    with col1:
        st.subheader("Textbook Content")
        tb_files = st.file_uploader(
            "Upload textbook content (PDF, DOCX, or images)",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="textbook_files"
        )
        tb_text_fallback = st.text_area("Or paste textbook text below", height=150)

        if tb_files:
            textbook_parsed = []
            for tb_file in tb_files:
                fname = tb_file.name.lower()
                if fname.endswith(".pdf"):
                    textbook_parsed.append(parse_pdf(tb_file))
                elif fname.endswith(".docx"):
                    textbook_parsed.append(parse_docx(tb_file))
                elif fname.endswith((".png", ".jpg", ".jpeg")):
                    textbook_parsed.append(parse_image_ocr(tb_file))
                else:
                    st.warning(f"Unsupported file type: {fname}")
            # Combine parsed content
            final_textbook = "\n".join(textbook_parsed).strip()

        # Fallback to manual input if no files
        final_textbook = final_textbook or tb_text_fallback.strip()

    ####################################
    # SME Content Upload and Parsing
    ####################################
    with col2:
        st.subheader("SME Content")
        sme_files = st.file_uploader(
            "Upload SME content (PDF, DOCX, or images)",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="sme_files"
        )
        sme_text_fallback = st.text_area("Or paste SME text below", height=150)

        if sme_files:
            sme_parsed = []
            for sme_file in sme_files:
                fname = sme_file.name.lower()
                if fname.endswith(".pdf"):
                    sme_parsed.append(parse_pdf(sme_file))
                elif fname.endswith(".docx"):
                    sme_parsed.append(parse_docx(sme_file))
                elif fname.endswith((".png", ".jpg", ".jpeg")):
                    sme_parsed.append(parse_image_ocr(sme_file))
                else:
                    st.warning(f"Unsupported file type: {fname}")
            # Combine parsed content
            final_sme = "\n".join(sme_parsed).strip()

        # Fallback to manual input if no files
        final_sme = final_sme or sme_text_fallback.strip()

    ####################################
    # Save Data and Proceed to Next Step
    ####################################
    if st.button("Next"):
        st.session_state["textbook_text"] = final_textbook
        st.session_state["sme_text"] = final_sme
        st.session_state["page"] = 2  # Move to Analyze Phase
############################################
# Page 2: Analyze (Chunk + Summaries)
############################################
def page_analyze():
    st.title("Lesson Builder: Step 3 - Analyze (Chunk + Summaries)")

    metadata = st.session_state.get("metadata", {})
    objective = metadata.get("Lesson Objective", "")
    textbook = st.session_state.get("textbook_text", "")
    sme = st.session_state.get("sme_text", "")

    combined_text = (textbook + "\n" + sme).strip()
    if not combined_text:
        st.warning("No combined content found. Please go back to Step 2.")
        return

    if st.button("Analyze Now"):
        paragraphs = [p.strip() for p in combined_text.split("\n") if p.strip()]
        analysis_results = analyze_chunks_with_llm(paragraphs, objective)
        st.session_state["analysis_df"] = analysis_results
        st.success("Analysis complete! View results below.")

    if "analysis_df" in st.session_state:
        df_show = st.session_state["analysis_df"]
        st.dataframe(df_show)


def analyze_chunks_with_llm(paragraphs, objective):
    data = []
    objective_emb = embedding_model.encode(objective, convert_to_tensor=True)
    for i, para in enumerate(paragraphs):
        para_emb = embedding_model.encode(para, convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(para_emb, objective_emb)[0][0])
        word_count = len(para.split())
        data.append({
            "chunk_id": i + 1,
            "paragraph": para,
            "alignment_score": round(sim_score, 3),
            "word_count": word_count
        })
    return pd.DataFrame(data)


############################################
# Main Execution
############################################
if __name__ == "__main__":
    main()
