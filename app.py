import streamlit as st
import pandas as pd
import PyPDF2
from docx import Document
import pytesseract
from PIL import Image
from docx import Document as WordDocument
import io

############################################
# Helper Functions for File Parsing
############################################
def parse_pdf(uploaded_pdf):
    """Extract text from PDF."""
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
    """Extract text from DOCX files."""
    text = ""
    try:
        doc = Document(uploaded_docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return text

def parse_image(uploaded_image):
    """Extract text from images using OCR."""
    text = ""
    try:
        image = Image.open(uploaded_image)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error reading image: {e}")
    return text

def parse_multiple_files(uploaded_files):
    """Parse multiple uploaded files and combine their content."""
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
# Multi-Step Workflow
############################################
def main():
    st.set_page_config(page_title="Advanced Lesson Builder", layout="wide")

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state["page"] = 0
    if "textbook_text" not in st.session_state:
        st.session_state["textbook_text"] = ""
    if "sme_text" not in st.session_state:
        st.session_state["sme_text"] = ""
    if "metadata" not in st.session_state:
        st.session_state["metadata"] = {}
    if "screens_df" not in st.session_state:
        st.session_state["screens_df"] = pd.DataFrame(columns=["Screen Title", "Text", "Estimated Duration", "Interactive Element"])

    show_sidebar_progress()
    page = st.session_state["page"]

    if page == 0:
        page_metadata()
    elif page == 1:
        page_content()
    elif page == 2:
        page_storyboard()
    elif page == 3:
        page_export()
    else:
        st.error("Invalid page index!")

############################################
# Sidebar Navigation
############################################
def show_sidebar_progress():
    """Display the sidebar for navigation and progress tracking."""
    st.sidebar.title("Lesson Builder Steps")
    steps = ["Metadata", "Content Upload", "Storyboard", "Export"]
    for i, step in enumerate(steps):
        if i == st.session_state["page"]:
            st.sidebar.write(f"▶️ {step} - In Progress")
        elif i < st.session_state["page"]:
            st.sidebar.write(f"✅ {step} - Done")
        else:
            st.sidebar.write(f"⬜ {step} - Pending")

############################################
# Step 1: Metadata
############################################
def page_metadata():
    """Step 1: Collect metadata."""
    st.title("Step 1: Metadata")
    st.write("Enter metadata for the lesson.")

    metadata_inputs = {
        "Course Title": st.text_input("Course Title", st.session_state["metadata"].get("Course Title", "")),
        "Module Title": st.text_input("Module Title", st.session_state["metadata"].get("Module Title", "")),
        "Unit Title": st.text_input("Unit Title", st.session_state["metadata"].get("Unit Title", "")),
        "Lesson Title": st.text_input("Lesson Title", st.session_state["metadata"].get("Lesson Title", "")),
        "Lesson Objective": st.text_input("Lesson Objective", st.session_state["metadata"].get("Lesson Objective", ""))
    }

    if st.button("Next"):
        if all(metadata_inputs.values()):
            st.session_state["metadata"] = metadata_inputs
            st.session_state["page"] = 1
        else:
            st.error("Please fill in all fields.")

############################################
# Step 2: Content Upload
############################################
def page_content():
    """Step 2: Upload textbook and SME content."""
    st.title("Step 2: Content Upload")
    st.write("Upload textbook and SME files or enter content manually.")

    # Textbook content
    st.subheader("Textbook Content")
    uploaded_textbook_files = st.file_uploader("Upload textbook files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)
    textbook_manual_text = st.text_area("Or enter textbook content manually", st.session_state.get("textbook_text", ""))

    # SME content
    st.subheader("SME Content")
    uploaded_sme_files = st.file_uploader("Upload SME files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)
    sme_manual_text = st.text_area("Or enter SME content manually", st.session_state.get("sme_text", ""))

    if st.button("Next"):
        # Combine uploaded and manual content
        st.session_state["textbook_text"] = parse_multiple_files(uploaded_textbook_files) + "\n" + textbook_manual_text
        st.session_state["sme_text"] = parse_multiple_files(uploaded_sme_files) + "\n" + sme_manual_text

        if st.session_state["textbook_text"].strip() or st.session_state["sme_text"].strip():
            st.session_state["page"] = 2
        else:
            st.error("Please upload or enter content.")

############################################
# Step 3: Storyboard
############################################
def page_storyboard():
    """Step 3: Create storyboard."""
    st.title("Step 3: Storyboard")
    st.write("Generate storyboard based on uploaded content.")

    # Generate storyboard data
    content = st.session_state["textbook_text"] + "\n" + st.session_state["sme_text"]
    st.subheader("Raw Content")
    st.text_area("Preview Raw Content", content, height=300)

    if st.button("Generate Storyboard"):
        screens = create_storyboard(content, st.session_state["metadata"])
        st.session_state["screens_df"] = pd.DataFrame(screens)
        st.success("Storyboard generated!")

    if not st.session_state["screens_df"].empty:
        st.subheader("Generated Storyboard")
        st.dataframe(st.session_state["screens_df"])

############################################
# Storyboard Creation Logic
############################################
def create_storyboard(content, metadata):
    """Create storyboard data from content."""
    chunks = content.split("\n\n")  # Split into chunks by double newlines
    screens = []
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            screens.append({
                "Screen Title": f"Screen {i + 1}",
                "Text": chunk.strip(),
                "Estimated Duration": round(len(chunk.split()) / 140, 2),  # Assume 140 words per minute
                "Interactive Element": "None"  # Placeholder
            })
    return screens

############################################
# Step 4: Export
############################################
def page_export():
    """Step 4: Export storyboard."""
    st.title("Step 4: Export")
    if st.session_state["screens_df"].empty:
        st.error("No storyboard available to export.")
        return

    st.write("Export the generated storyboard.")

    # Export to Word
    if st.button("Export to Word"):
        export_to_word(st.session_state["screens_df"], st.session_state["metadata"])
        st.success("Word document exported!")

############################################
# Export Logic
############################################
def export_to_word(screens_df, metadata):
    """Export storyboard to Word."""
    doc = WordDocument()
    doc.add_heading(metadata["Lesson Title"], level=1)
    for _, row in screens_df.iterrows():
        doc.add_heading(row["Screen Title"], level=2)
        doc.add_paragraph(row["Text"])
        doc.add_paragraph(f"Estimated Duration: {row['Estimated Duration']} minutes")
        doc.add_paragraph(f"Interactive Element: {row['Interactive Element']}")
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    st.download_button("Download Word Document", buffer, file_name="storyboard.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

############################################
# Run the App
############################################
if __name__ == "__main__":
    main()
