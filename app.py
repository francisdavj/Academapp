import streamlit as st
import mysql.connector
import datetime
import pandas as pd
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer, util
import pytesseract
from PIL import Image
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

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
        changes_str = json.dumps(log_entry.get("changes", {}))
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
        old_val = old_row.get(col, "")
        new_val = new_row.get(col, "")
        if old_val != new_val:
            changes[col] = {"old": old_val, "new": new_val}

    if changes:
        log_entry = {"screen_index": screen_index, "changes": changes}
        log_usage_to_db(log_entry)

############################################
# 2) Advanced AI Setup
############################################
@st.cache_resource
def load_summarizer():
    """
    Load the summarization pipeline.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

@st.cache_resource
def load_generator():
    """
    Load the text generation pipeline.
    """
    generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")
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

def generate_lesson_screens(aligned_paragraphs, objective, generator):
    """
    Generate lesson screens based on aligned paragraphs and objectives.
    """
    try:
        # Concatenate aligned paragraphs
        combined_text = "\n\n".join(aligned_paragraphs)
        prompt = (
            f"Create a structured lesson based on the following objective:\n\n"
            f"Objective: {objective}\n\n"
            f"Content:\n{combined_text}\n\n"
            f"Generate lesson screens with titles, text, estimated durations, and placeholders for interactive elements like quizzes and reflections.\n\n"
            f"Format the output as a JSON array where each item represents a screen with the following fields: 'Screen Title', 'Text', 'Estimated Duration', 'Interactive Element'."
        )
        # Generate text
        generated_text = generator(prompt, max_length=2000, num_return_sequences=1)[0]['generated_text']
        
        # Attempt to extract JSON from the generated text
        start_idx = generated_text.find('[')
        end_idx = generated_text.rfind(']') + 1
        json_str = generated_text[start_idx:end_idx]
        
        screens = json.loads(json_str)
        return screens
    except Exception as e:
        st.error(f"Generation Error: {e}")
        return []

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
        for page_num, page in enumerate(reader.pages, start=1):
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
            else:
                # Handle scanned images in PDF using OCR
                if '/XObject' in page['/Resources']:
                    xObject = page['/Resources']['/XObject'].get_object()
                    for obj in xObject:
                        if xObject[obj]['/Subtype'] == '/Image':
                            data = xObject[obj]._data
                            img = Image.open(io.BytesIO(data))
                            text += pytesseract.image_to_string(img) + "\n"
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
        text = pytesseract.image_to_string(image) + "\n"
    except Exception as e:
        st.error(f"Error reading image: {e}")
    return text

############################################
# 4) Multi-Step Flow
############################################
def main():
    st.set_page_config(page_title="Advanced Lesson Builder", layout="wide")

    if "page" not in st.session_state:
        st.session_state["page"] = 0
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
        page_generate()
    elif page == 4:
        page_refine()
    elif page == 5:
        page_export()
    else:
        st.write("Invalid page index!")

def show_sidebar_progress():
    """
    Display the sidebar for progress tracking.
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
        "Course and Title": st.text_input("Course # and Title", ""),
        "Module and Title": st.text_input("Module # and Title", ""),
        "Unit and Title": st.text_input("Unit # and Title", ""),
        "Lesson and Title": st.text_input("Lesson # and Title", ""),
        "Lesson Objective": st.text_input("Lesson Objective", ""),
        "Lesson Type": st.selectbox("Lesson Type", ["Core Learning Lesson", "Practice Lesson", "Other"])
    }

    if st.button("Next"):
        # Basic validation
        if not all(metadata_inputs.values()):
            st.error("Please fill in all metadata fields.")
            return
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
        if not textbook_parsed and not sme_parsed:
            st.error("Please upload at least one content file.")
            return
        st.session_state["textbook_text"] = textbook_parsed
        st.session_state["sme_text"] = sme_parsed
        st.session_state["page"] = 2

def handle_file_upload(section, key):
    """
    Helper function to handle file uploads and parsing.
    """
    st.subheader(section)
    uploaded_file = st.file_uploader(f"Upload a file for {section}", type=["pdf", "docx", "png", "jpg", "jpeg"], key=key)
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            return parse_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            return parse_docx(uploaded_file)
        elif uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
            return parse_image(uploaded_file)
    return ""

def page_analyze():
    """
    Step 3: Analyze chunks for alignment and summarization.
    """
    st.title("Lesson Builder: Step 3 - Analyze")
    objective = st.session_state.get("metadata", {}).get("Lesson Objective", "")
    combined_text = st.session_state.get("textbook_text", "") + "\n" + st.session_state.get("sme_text", "")

    if not combined_text.strip():
        st.warning("No content found. Please go back to Step 2.")
        return

    summarizer = load_summarizer()

    if st.button("Analyze Now"):
        paragraphs = [p for p in combined_text.split("\n") if p.strip()]
        df_analysis = analyze_chunks(paragraphs, objective, summarizer)
        st.session_state["analysis_df"] = df_analysis

    if "analysis_df" in st.session_state:
        st.subheader("Alignment Analysis")
        st.dataframe(st.session_state["analysis_df"])

def analyze_chunks(paragraphs, objective, summarizer):
    """
    Analyze alignment of content to objective and summarize.
    """
    data = []
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    objective_emb = embedding_model.encode(objective, convert_to_tensor=True)
    for i, para in enumerate(paragraphs):
        para_emb = embedding_model.encode(para, convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(para_emb, objective_emb)[0][0])
        summary = summarize_paragraph(para, summarizer)
        data.append({
            "Chunk ID": i + 1,
            "Paragraph": para,
            "Summary": summary,
            "Alignment Score": round(sim_score, 3)
        })
    return pd.DataFrame(data)

def page_generate():
    """
    Step 4: Generate lesson screens.
    """
    st.title("Lesson Builder: Step 4 - Generate")
    if "analysis_df" not in st.session_state:
        st.warning("Please complete Step 3: Analyze first.")
        return

    analysis_df = st.session_state["analysis_df"]
    objective = st.session_state["metadata"]["Lesson Objective"]

    generator = load_generator()

    if st.button("Generate Lesson Screens"):
        aligned_paragraphs = analysis_df[analysis_df["Alignment Score"] >= 0.7]["Paragraph"].tolist()
        if not aligned_paragraphs:
            st.error("No sufficiently aligned content found to generate lesson screens.")
            return
        lesson_screens = generate_lesson_screens(aligned_paragraphs, objective, generator)
        if lesson_screens:
            screens_df = pd.DataFrame(lesson_screens)
            # Ensure all necessary columns are present
            for col in ["Screen Title", "Text", "Estimated Duration", "Interactive Element"]:
                if col not in screens_df.columns:
                    screens_df[col] = ""
            st.session_state["screens_df"] = screens_df
            st.success("Lesson screens generated successfully!")
        else:
            st.error("Failed to generate lesson screens.")

    if "screens_df" in st.session_state and not st.session_state["screens_df"].empty:
        st.subheader("Generated Lesson Screens")
        st.dataframe(st.session_state["screens_df"])

def page_refine():
    """
    Step 5: Refine lesson screens.
    """
    st.title("Lesson Builder: Step 5 - Refine")
    if "screens_df" not in st.session_state or st.session_state["screens_df"].empty:
        st.warning("Please complete Step 4: Generate first.")
        return

    screens_df = st.session_state["screens_df"]

    edited_screens = []
    for index, row in screens_df.iterrows():
        st.markdown(f"### Screen {index + 1}")
        with st.form(key=f"screen_form_{index}"):
            title = st.text_input("Screen Title", value=row["Screen Title"], key=f"title_{index}")
            text = st.text_area("Text", value=row["Text"], key=f"text_{index}")
            duration = st.number_input("Estimated Duration (minutes)", value=row["Estimated Duration"] if pd.notnull(row["Estimated Duration"]) else 1, min_value=1, key=f"duration_{index}")
            interactive = st.selectbox("Add Interactive Element", ["None", "Quiz", "Reflection", "Video"], index=["None", "Quiz", "Reflection", "Video"].index(row["Interactive Element"]) if row["Interactive Element"] in ["None", "Quiz", "Reflection", "Video"] else 0, key=f"interactive_{index}")
            submit = st.form_submit_button("Save Changes")
            if submit:
                edited_screens.append({
                    "Screen Title": title,
                    "Text": text,
                    "Estimated Duration": duration,
                    "Interactive Element": interactive
                })
                # Log changes
                old_row = row.to_dict()
                new_row = {
                    "Screen Title": title,
                    "Text": text,
                    "Estimated Duration": duration
                }
                log_user_change_db(index + 1, old_row, new_row)

    if edited_screens:
        st.session_state["screens_df"] = pd.DataFrame(edited_screens)
        st.success("All changes saved!")

    st.subheader("Refined Lesson Screens")
    st.dataframe(st.session_state["screens_df"])

def page_export():
    """
    Step 6: Export lesson data.
    """
    st.title("Lesson Builder: Step 6 - Export")
    if "screens_df" not in st.session_state or st.session_state["screens_df"].empty:
        st.warning("No lesson data to export. Please complete the previous steps first.")
        return

    screens_df = st.session_state["screens_df"]

    export_format = st.selectbox("Select Export Format", ["CSV", "JSON"])

    if export_format == "CSV":
        csv = screens_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='lesson_screens.csv',
            mime='text/csv',
        )
    elif export_format == "JSON":
        json_data = screens_df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name='lesson_screens.json',
            mime='application/json',
        )

    st.success("Export completed successfully!")

############################################
# Final Steps and Output Generation
############################################
if __name__ == "__main__":
    main()
