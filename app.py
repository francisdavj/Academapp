import streamlit as st
import pandas as pd
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer, util
import pytesseract
from PIL import Image
from docx import Document as WordDocument
import io
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import concurrent.futures

############################################
# Helper Functions for File Parsing
############################################
def parse_pdf(uploaded_pdf):
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
    text = ""
    try:
        doc = Document(uploaded_docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return text

def parse_image(uploaded_image):
    text = ""
    try:
        image = Image.open(uploaded_image)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error reading image: {e}")
    return text

def parse_multiple_files(uploaded_files):
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
    """Load the SentenceTransformer model once and cache it for faster reuse."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_gpt_model():
    """Load a GPT-based model for content generation"""
    model_name = "EleutherAI/gpt-neo-2.7B"  # Using GPT-Neo for interactivity suggestions
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

def generate_interactivity(content_chunk):
    """Generate interactivity suggestion using GPT model"""
    generator = load_gpt_model()
    prompt = f"Given this content: {content_chunk}, suggest an appropriate interactive element such as a quiz, mind map, or timeline, and provide a short description of how it should work."
    
    try:
        response = generator(prompt, max_new_tokens=50, num_return_sequences=1, truncation=True, pad_token_id=50256)
        if response:
            return response[0]['generated_text']
        else:
            return "No suggestion generated."
    except Exception as e:
        return f"Error generating interactivity: {e}"

############################################
# Multi-Step Workflow
############################################
def main():
    st.set_page_config(page_title="Advanced Lesson Builder", layout="wide")

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
# Sidebar Progress
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
    st.title("Step 1: Metadata")
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
    st.title("Step 2: Content Upload")
    uploaded_textbook_files = st.file_uploader("Upload textbook files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)
    textbook_manual_text = st.text_area("Or enter textbook content manually", st.session_state.get("textbook_text", ""))
    uploaded_sme_files = st.file_uploader("Upload SME files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)
    sme_manual_text = st.text_area("Or enter SME content manually", st.session_state.get("sme_text", ""))

    if st.button("Next"):
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
    st.title("Step 3: Analyze Content")

    if "metadata" not in st.session_state or not st.session_state["metadata"].get("Lesson Objective", "").strip():
        st.error("Lesson Objective is missing. Please go back to Step 1: Metadata.")
        return

    if not st.session_state.get("textbook_text") and not st.session_state.get("sme_text"):
        st.error("No content found. Please go back to Step 2: Content.")
        return

    combined_text = st.session_state["textbook_text"] + "\n" + st.session_state["sme_text"]
    st.text_area("Preview Combined Content", combined_text, height=300)

    embedding_model = load_embedding_model()
    objective = st.session_state["metadata"]["Lesson Objective"]
    objective_emb = embedding_model.encode(objective, convert_to_tensor=True)

    paragraphs = combined_text.split("\n\n")
    data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_para = {executor.submit(process_paragraph, para, objective_emb): para for para in paragraphs}
        for future in concurrent.futures.as_completed(future_to_para):
            result = future.result()
            if result:
                data.append(result)

    df_analysis = pd.DataFrame(data)
    st.session_state["analysis_df"] = df_analysis
    st.write("Alignment Results:")
    st.dataframe(df_analysis)

    if st.button("Next"):
        if not df_analysis.empty:
            st.session_state["page"] = 3
            st.success("Proceeding to Step 4: Storyboard.")
        else:
            st.error("Please make sure the analysis is complete before moving to the next step.")

def process_paragraph(para, objective_emb):
    embedding_model = load_embedding_model()
    para_emb = embedding_model.encode(para, convert_to_tensor=True)
    sim_score = float(util.pytorch_cos_sim(para_emb, objective_emb)[0][0])
    return {"Chunk ID": para, "Paragraph": para.strip(), "Alignment Score": round(sim_score, 3)}

############################################
# Step 4: Storyboard Creation
############################################
def page_storyboard():
    st.title("Step 4: Storyboard Creation with AI-Driven Interactivity")

    if "analysis_df" not in st.session_state or st.session_state["analysis_df"].empty:
        st.error("No analysis results available. Please complete the content analysis first.")
        return

    st.session_state["screens_df"] = st.session_state["analysis_df"]

    for index, row in st.session_state["screens_df"].iterrows():
        st.write(f"### Screen {index + 1}: {row['Paragraph'][:50]}...")  # Display snippet of the content

        interactivity_suggestion = generate_interactivity(row['Paragraph'])
        st.write(f"AI Suggested Interactivity: {interactivity_suggestion}")

        st.session_state["screens_df"].at[index, "Interactive Element"] = interactivity_suggestion

    if st.button("Save Storyboard"):
        st.success("Storyboard with interactivities saved successfully!")
        st.write(st.session_state["screens_df"])

# Continue with Steps 5 and 6...
############################################
# Step 5: Refine Storyboard
############################################
def page_refine():
    st.title("Step 5: Refine Storyboard")

    # Check if storyboard data is available
    if "screens_df" not in st.session_state or st.session_state["screens_df"].empty:
        st.error("No storyboard available to refine. Please complete Step 4: Storyboard first.")
        return

    edited_df = st.data_editor(
        st.session_state["screens_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="refine_editor"
    )

    # Allow for the modification of interactive elements in the refined storyboard
    st.write("### Add/Modify Interactive Elements:")
    for index, row in edited_df.iterrows():
        st.write(f"#### Screen {index + 1}: {row['Paragraph'][:50]}...")  # Show snippet of content
        interactivity_type = st.selectbox(
            f"Choose an interactive element for Screen {index + 1}:",
            ["None", "Quiz", "Reflection Question", "Accordion", "Tab"],
            index=["None", "Quiz", "Reflection Question", "Accordion", "Tab"].index(row.get("Interactive Element", "None")),
            key=f"interactive_{index}"
        )
        edited_df.at[index, "Interactive Element"] = interactivity_type

    # Save the refined storyboard back to session state
    if st.button("Save Refinements"):
        st.session_state["screens_df"] = edited_df
        st.success("Refinements saved successfully!")

    # Display the refined storyboard preview
    st.subheader("Refined Storyboard Preview")
    st.dataframe(st.session_state["screens_df"])

############################################
# Step 6: Export
############################################
def page_export():
    st.title("Step 6: Export")

    if st.session_state["screens_df"].empty:
        st.error("No storyboard available to export.")
        return

    if st.button("Export to Word"):
        export_to_word(st.session_state["screens_df"], st.session_state["metadata"])

############################################
# Export Logic
############################################
def export_to_word(screens_df, metadata):
    doc = WordDocument()
    doc.add_heading(metadata["Lesson Title"], level=1)
    
    for _, row in screens_df.iterrows():
        doc.add_heading(f"Screen {row['Chunk ID']}", level=2)
        doc.add_paragraph(row["Paragraph"])
        doc.add_paragraph(f"Estimated Duration: {row.get('Estimated Duration', 'Not Provided')} minutes")
        doc.add_paragraph(f"Interactive Element: {row.get('Interactive Element', 'None')}")
    
    # Save the file to memory and allow download
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    st.download_button("Download Word Document", buffer, file_name="storyboard.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# Run the App
if __name__ == "__main__":
    main()
