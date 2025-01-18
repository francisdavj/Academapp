import streamlit as st
import pandas as pd
import PyPDF2
from docx import Document
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer, util


############################################
# Utility Functions
############################################

# Parse PDF files
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

# Parse DOCX files
def parse_docx(uploaded_docx):
    text = ""
    try:
        doc = Document(uploaded_docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return text

# Parse images using OCR
def parse_image_ocr(uploaded_image):
    text = ""
    try:
        image = Image.open(uploaded_image)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error processing image: {e}")
    return text

# Load embedding model
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Analyze chunks for alignment and summaries
def analyze_chunks_with_llm(paragraphs, objective):
    data = []
    objective_emb = embedding_model.encode(objective, convert_to_tensor=True)
    for i, para in enumerate(paragraphs):
        para_emb = embedding_model.encode(para, convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(para_emb, objective_emb)[0][0])
        word_count = len(para.split())
        data.append({
            "Chunk ID": i + 1,
            "Paragraph": para,
            "Alignment Score": round(sim_score, 3),
            "Word Count": word_count
        })
    return pd.DataFrame(data)

############################################
# Streamlit App Functions
############################################

# Sidebar progress
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

# Step 1: Metadata
def page_metadata():
    st.title("Lesson Builder: Step 1 - Metadata")
    course_title = st.text_input("Course # and Title", st.session_state.get("course_title", ""))
    module_title = st.text_input("Module # and Title", st.session_state.get("module_title", ""))
    unit_title = st.text_input("Unit # and Title", st.session_state.get("unit_title", ""))
    lesson_title = st.text_input("Lesson # and Title", st.session_state.get("lesson_title", ""))
    lesson_objective = st.text_input("Lesson Objective", st.session_state.get("lesson_objective", ""))
    lesson_type = st.selectbox("Lesson Type", ["Core Learning Lesson", "Practice Lesson", "Other"], index=0)

    if st.button("Next"):
        st.session_state["course_title"] = course_title
        st.session_state["module_title"] = module_title
        st.session_state["unit_title"] = unit_title
        st.session_state["lesson_title"] = lesson_title
        st.session_state["lesson_objective"] = lesson_objective
        st.session_state["lesson_type"] = lesson_type
        st.session_state["page"] = 1

# Step 2: Content Upload
def page_content():
    st.title("Lesson Builder: Step 2 - Content")
    st.write("Upload multiple files (PDF/DOCX/image) or manually provide text for textbook and SME content.")
    
    col1, col2 = st.columns(2)

    # Textbook Content
    with col1:
        st.subheader("Textbook Content")
        textbook_files = st.file_uploader(
            "Upload textbook files (PDF, DOCX, images)",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="textbook_files"
        )
        tb_text_fallback = st.text_area("Or paste textbook text below (mandatory)", 
                                        st.session_state.get("textbook_text", ""), 
                                        height=150)

        textbook_parsed = ""
        for file in textbook_files:
            fname = file.name.lower()
            if fname.endswith(".pdf"):
                textbook_parsed += parse_pdf(file)
            elif fname.endswith(".docx"):
                textbook_parsed += parse_docx(file)
            elif fname.endswith((".png", ".jpg", ".jpeg")):
                textbook_parsed += parse_image_ocr(file)

        final_textbook = textbook_parsed.strip() or tb_text_fallback.strip()
        if not final_textbook:
            st.error("Textbook content is required. Please upload files or enter text.")
            st.stop()

    # SME Content
    with col2:
        st.subheader("SME Content")
        sme_files = st.file_uploader(
            "Upload SME files (PDF, DOCX, images)",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="sme_files"
        )
        sme_text_fallback = st.text_area("Or paste SME text below (mandatory)", 
                                         st.session_state.get("sme_text", ""), 
                                         height=150)

        sme_parsed = ""
        for file in sme_files:
            fname = file.name.lower()
            if fname.endswith(".pdf"):
                sme_parsed += parse_pdf(file)
            elif fname.endswith(".docx"):
                sme_parsed += parse_docx(file)
            elif fname.endswith((".png", ".jpg", ".jpeg")):
                sme_parsed += parse_image_ocr(file)

        final_sme = sme_parsed.strip() or sme_text_fallback.strip()
        if not final_sme:
            st.error("SME content is required. Please upload files or enter text.")
            st.stop()

    # Save and Proceed
    if st.button("Next"):
        st.session_state["textbook_text"] = final_textbook
        st.session_state["sme_text"] = final_sme
        st.session_state["page"] = 2

    if st.button("Back"):
        st.session_state["page"] = 0

# Step 3: Analyze
def page_analyze():
    st.title("Lesson Builder: Step 3 - Analyze (Chunk + Summaries)")

    # Retrieve combined content
    textbook = st.session_state.get("textbook_text", "")
    sme = st.session_state.get("sme_text", "")
    combined_text = (textbook + "\n" + sme).strip()

    if not combined_text:
        st.warning("No combined content found. Please go back to Step 2 and upload or enter content.")
        return

    st.write("Analyzing combined content for alignment with lesson objectives and overall structure.")

    paragraphs = [p.strip() for p in combined_text.split("\n") if p.strip()]
    if st.button("Analyze Now"):
        objective = st.session_state.get("lesson_objective", "")
        df_analysis = analyze_chunks_with_llm(paragraphs, objective)
        st.session_state["analysis_df"] = df_analysis
        st.success("Analysis complete! Scroll down to view results.")

    if "analysis_df" in st.session_state:
        df_show = st.session_state["analysis_df"]
        st.dataframe(df_show)

        total_words = df_show["Word Count"].sum()
        est_minutes = total_words / 140.0
        st.write(f"Total words: {total_words}, approx {est_minutes:.1f} minutes.")
        if est_minutes < 10:
            st.warning("Content may be insufficient for a 15-minute lesson. Consider adding more SME content.")
        elif est_minutes > 15:
            st.warning("Content exceeds 15 minutes. Consider trimming or splitting into smaller sections.")

        if st.button("Next: Generate Outline"):
            st.session_state["page"] = 3

    if st.button("Back"):
        st.session_state["page"] = 1

# Main Function
def main():
    st.set_page_config(page_title="Advanced Lesson Builder", layout="wide")
    if "page" not in st.session_state:
        st.session_state["page"] = 0

    show_sidebar_progress()
    page = st.session_state["page"]
    if page == 0:
        page_metadata()
    elif page == 1:
        page_content()
    elif page == 2:
        page_analyze()

if __name__ == "__main__":
    main()
