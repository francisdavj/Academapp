import streamlit as st
import pandas as pd
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer, util
import pytesseract
from PIL import Image
import io
import openpyxl  # For exporting to Excel

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
# AI Model Setup
############################################
@st.cache_resource
def load_embedding_model():
    """Load the SentenceTransformer model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
        page_analyze()
    elif page == 3:
        page_storyboard()
    elif page == 4:
        page_refine()
    elif page == 5:
        page_export()
    else:
        st.error("Invalid page index!")

############################################
# Sidebar Navigation
############################################
def show_sidebar_progress():
    """Display the sidebar for navigation and progress tracking."""
    st.sidebar.title("Lesson Builder Steps")
    steps = ["Metadata", "Content Upload", "Analyze Content", "Storyboard", "Refine Storyboard", "Export"]
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
# Step 3: Analyze Content
############################################
def page_analyze():
    """Step 3: Analyze content for alignment with lesson objectives."""
    st.title("Step 3: Analyze Content")

    # Check if metadata and content exist
    if "metadata" not in st.session_state or not st.session_state["metadata"].get("Lesson Objective", "").strip():
        st.error("Lesson Objective is missing. Please go back to Step 1: Metadata.")
        return

    if not st.session_state.get("textbook_text") and not st.session_state.get("sme_text"):
        st.error("No content found. Please go back to Step 2: Content.")
        return

    # Combine content
    combined_text = st.session_state["textbook_text"] + "\n" + st.session_state["sme_text"]
    st.text_area("Preview Combined Content", combined_text, height=300)

    embedding_model = load_embedding_model()
    objective = st.session_state["metadata"]["Lesson Objective"]
    objective_emb = embedding_model.encode(objective, convert_to_tensor=True)

    # Analyze chunks
    paragraphs = combined_text.split("\n\n")
    data = []
    for i, para in enumerate(paragraphs):
        if para.strip():
            para_emb = embedding_model.encode(para, convert_to_tensor=True)
            sim_score = float(util.pytorch_cos_sim(para_emb, objective_emb)[0][0])
            data.append({
                "Chunk ID": i + 1,
                "Paragraph": para.strip(),
                "Alignment Score": round(sim_score, 3)
            })

    df_analysis = pd.DataFrame(data)
    st.session_state["analysis_df"] = df_analysis
    st.write("Alignment Results:")
    st.dataframe(df_analysis)

############################################
# Step 4: Storyboard Creation with Interactive Elements
############################################
def page_storyboard():
    """Step 4: Create storyboard with AI-generated interactive elements."""
    st.title("Step 4: Storyboard")
    
    # Check if analysis results are available
    if "analysis_df" not in st.session_state or st.session_state["analysis_df"].empty:
        st.error("No analysis results available. Please complete Step 3: Analyze Content first.")
        return

    # Extract the analysis data into a structured format
    analysis_df = st.session_state["analysis_df"]
    
    # Create structured storyboard with interactive elements
    screens = []
    for i, row in analysis_df.iterrows():
        screen_title = f"Screen {i + 1}"  # Automatically generated screen titles
        paragraph = row["Paragraph"]
        alignment_score = row["Alignment Score"]
        estimated_duration = round(len(paragraph.split()) / 140, 2)  # Assuming 140 words = 1 minute
        
        # AI-based decision for adding interactive elements based on content
        if alignment_score > 0.8:  # For high alignment, suggest a quiz or reflection
            interactive_element = "Quiz"  # Example: AI suggests a quiz
        elif alignment_score > 0.5:  # For moderate alignment, suggest a reflection question
            interactive_element = "Reflection Question"
        else:  # For low alignment, suggest an activity or task
            interactive_element = "Activity"

        # Add each screen's data to the storyboard
        screens.append({
            "Screen Title": screen_title,
            "Text": paragraph,
            "Estimated Duration": estimated_duration,
            "Interactive Element": interactive_element
        })

    # Convert list of screens to DataFrame
    storyboard_df = pd.DataFrame(screens)

    # Store the storyboard in session state
    st.session_state["screens_df"] = storyboard_df
    
    # Display the structured storyboard with AI-generated interactive elements
    st.subheader("Generated Storyboard with Interactive Elements")
    st.dataframe(st.session_state["screens_df"])

    # Add a button to proceed to the refinement phase
    if st.button("Proceed to Refine"):
        st.session_state["page"] = 4

############################################
# Step 5: Refine Storyboard
############################################
def page_refine():
    """Step 5: Refine storyboard."""
    st.title("Step 5: Refine Storyboard")

    # Check if storyboard data is available
    if "screens_df" not in st.session_state or st.session_state["screens_df"].empty:
        st.error("No storyboard available to refine. Please complete Step 4: Storyboard first.")
        return

    st.write("Refine the storyboard by editing screen content or adding activities.")

    # Editable DataFrame for storyboard refinement
    st.write("### Editable Storyboard:")
    edited_df = st.data_editor(
        st.session_state["screens_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="refine_editor"
    )

    # Save refined storyboard back to session state
    if st.button("Save Refinements"):
        st.session_state["screens_df"] = edited_df
        st.success("Refinements saved successfully!")

    # Debug: Display refined storyboard
    st.subheader("Refined Storyboard Preview")
    st.dataframe(st.session_state["screens_df"])

############################################
# Step 6: Export to Word and Excel
############################################
def page_export():
    """Step 6: Export storyboard."""
    st.title("Step 6: Export")
    if st.session_state["screens_df"].empty:
        st.error("No storyboard available to export.")
        return

    if st.button("Export to Word"):
        export_to_word(st.session_state["screens_df"], st.session_state["metadata"])
        
    if st.button("Export to Excel"):
        export_to_excel(st.session_state["screens_df"], st.session_state["metadata"])

############################################
# Export to Word
############################################
def export_to_word(screens_df, metadata):
    """Export storyboard to Word."""
    doc = WordDocument()
    doc.add_heading(metadata["Lesson Title"], level=1)
    for _, row in screens_df.iterrows():
        doc.add_heading(f"Screen {row['Chunk ID']}", level=2)
        doc.add_paragraph(row["Paragraph"])
        doc.add_paragraph(f"Estimated Duration: {row['Estimated Duration']} minutes")
        doc.add_paragraph(f"Interactive Element: {row['Interactive Element']}")
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    st.download_button("Download Word Document", buffer, file_name="storyboard.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

############################################
# Export to Excel
############################################
def export_to_excel(screens_df,
