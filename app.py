import streamlit as st
import pandas as pd
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer, util
import pytesseract
from PIL import Image
import json
from transformers import pipeline
import io

############################################
# 1) Advanced AI Setup
############################################
@st.cache_resource
def load_summarizer():
    """
    Load the summarization pipeline.
    """
    summarizer_model = st.secrets["models"]["summarizer_model"]  # e.g., "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model=summarizer_model)
    return summarizer

@st.cache_resource
def load_generator():
    """
    Load the text generation pipeline.
    """
    generator_model = st.secrets["models"]["generator_model"]  # e.g., "gpt2"
    generator = pipeline("text-generation", model=generator_model, tokenizer=generator_model)
    return generator

def summarize_paragraph(paragraph, summarizer):
    """
    Summarize a paragraph using the local summarization model.
    """
    try:
        summary = summarizer(paragraph, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        st.error(f"Summarization Error: {e}")
        return "Summary not available."

############################################
# 2) File Parsing
############################################
def parse_pdf(uploaded_pdf):
    """
    Extract text from PDF.
    """
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_pdf)
        for page in reader.pages:
            extracted_text = page.extract_text()
            text += extracted_text if extracted_text else ""
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

def parse_image(uploaded_image):
    """
    Extract text from images using OCR.
    """
    text = ""
    try:
        image = Image.open(uploaded_image)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error reading image: {e}")
    return text

def parse_multiple_files(uploaded_files):
    """
    Parse multiple uploaded files and combine their content.
    """
    combined_text = ""
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            combined_text += parse_pdf(file) + "\n"
        elif file.name.endswith(".docx"):
            combined_text += parse_docx(file) + "\n"
        elif file.name.endswith((".png", ".jpg", ".jpeg")):
            combined_text += parse_image(file) + "\n"
    return combined_text

############################################
# 3) Multi-Step Flow with Resume Capability
############################################
def main():
    st.set_page_config(page_title="Advanced Lesson Builder", layout="wide")

    # Initialize session state for navigation and data
    if "page" not in st.session_state:
        st.session_state["page"] = 0
    if "screens_df" not in st.session_state:
        st.session_state["screens_df"] = pd.DataFrame(columns=["Screen Title", "Text", "Estimated Duration", "Interactive Element"])
    if "metadata" not in st.session_state:
        st.session_state["metadata"] = {}
    if "textbook_text" not in st.session_state:
        st.session_state["textbook_text"] = ""
    if "sme_text" not in st.session_state:
        st.session_state["sme_text"] = ""

    # Navigation and step execution
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
    elif page == 5:
        page_export()
    else:
        st.write("Invalid page index!")

def show_sidebar_progress():
    """
    Display the sidebar for progress tracking and navigation.
    """
    st.sidebar.title("Lesson Steps Progress")
    steps = ["Metadata", "Content", "Analyze", "Generate", "Refine", "Export"]
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
        "Course and Title": st.text_input("Course # and Title", st.session_state["metadata"].get("Course and Title", "")),
        "Module and Title": st.text_input("Module # and Title", st.session_state["metadata"].get("Module and Title", "")),
        "Unit and Title": st.text_input("Unit # and Title", st.session_state["metadata"].get("Unit and Title", "")),
        "Lesson and Title": st.text_input("Lesson # and Title", st.session_state["metadata"].get("Lesson and Title", "")),
        "Lesson Objective": st.text_input("Lesson Objective", st.session_state["metadata"].get("Lesson Objective", "")),
        "Lesson Type": st.selectbox("Lesson Type", ["Core Learning Lesson", "Practice Lesson", "Other"], index=0)
    }

    if st.button("Next"):
        if not all(metadata_inputs.values()):
            st.error("Please fill in all metadata fields.")
        else:
            st.session_state["metadata"] = metadata_inputs
            st.session_state["page"] = 1

def page_content():
    """
    Step 2: Upload and parse content files or enter text directly.
    """
    st.title("Lesson Builder: Step 2 - Content")
    st.write("Upload textbook and SME files or enter text directly.")

    # Textbook content upload and manual text input
    st.subheader("Textbook Content")
    uploaded_textbook_files = st.file_uploader("Upload files for Textbook Content", type=["pdf", "docx", "png", "jpg", "jpeg"], key="textbook_files", accept_multiple_files=True)
    textbook_manual_text = st.text_area("Or enter Textbook Content manually", st.session_state.get("textbook_text", ""))

    # SME content upload and manual text input
    st.subheader("SME Content")
    uploaded_sme_files = st.file_uploader("Upload files for SME Content", type=["pdf", "docx", "png", "jpg", "jpeg"], key="sme_files", accept_multiple_files=True)
    sme_manual_text = st.text_area("Or enter SME Content manually", st.session_state.get("sme_text", ""))

    # Combine parsed content and manual text
    textbook_parsed = parse_multiple_files(uploaded_textbook_files) if uploaded_textbook_files else ""
    sme_parsed = parse_multiple_files(uploaded_sme_files) if uploaded_sme_files else ""
    combined_textbook_content = textbook_parsed + "\n" + textbook_manual_text
    combined_sme_content = sme_parsed + "\n" + sme_manual_text

    # Button to proceed to the next step
    if st.button("Next"):
        if not (combined_textbook_content.strip() or combined_sme_content.strip()):
            st.error("Please upload at least one content file or enter text manually.")
        else:
            st.session_state["textbook_text"] = combined_textbook_content
            st.session_state["sme_text"] = combined_sme_content
            st.session_state["page"] = 2

def page_analyze():
    """
    Step 3: Analyze chunks for alignment and summarization.
    """
    st.title("Lesson Builder: Step 3 - Analyze")

    # Check if content exists
    if not st.session_state.get("textbook_text") and not st.session_state.get("sme_text"):
        st.error("No content found. Please go back to Step 2.")
        return

    combined_text = st.session_state.get("textbook_text", "") + "\n" + st.session_state.get("sme_text", "")
    summarizer = load_summarizer()

    if st.button("Analyze Now"):
        paragraphs = [p for p in combined_text.split("\n") if p.strip()]
        data = []
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        objective = st.session_state["metadata"].get("Lesson Objective", "")
        objective_emb = embedding_model.encode(objective, convert_to_tensor=True)

        for i, para in enumerate(paragraphs):
            para_emb = embedding_model.encode(para, convert_to_tensor=True)
            sim_score = float(util.pytorch_cos_sim(para_emb, objective_emb)[0][0])
            summary = summarize_paragraph(para, summarizer)
            data.append({"Chunk ID": i + 1, "Paragraph": para, "Summary": summary, "Alignment Score": round(sim_score, 3)})

        df_analysis = pd.DataFrame(data)
        st.session_state["analysis_df"] = df_analysis
        st.success("Analysis completed!")

    if "analysis_df" in st.session_state:
        st.dataframe(st.session_state["analysis_df"])

############################################
# Final Steps
############################################
if __name__ == "__main__":
    main()
