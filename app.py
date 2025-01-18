def page_content():
    st.write("DEBUG: Entered page_content() function!")
    ...


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
    Connect to your Hostinger MySQL DB.
    Update these placeholders with your actual credentials.
    """
    host = "127.0.0.1"  # or "localhost" or "mysql-xxx.hostinger.com"
    user = "u628260032_francisdavid"
    password = "Chennai@202475"
    database = "u628260032_academapp"
    port = 3306  # typical MySQL port

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

def log_usage_to_db(log_entry):
    """
    Insert usage logs into usage_logs table (id, log_time, screen_index, changes TEXT).
    """
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        changes_str = str(log_entry.get("changes", {}))  # or json.dumps(...) if you prefer JSON
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
    Compare old vs. new row for columns ["Screen Title", "Text", "Estimated Duration"].
    If there's a difference, log it to usage_logs in MySQL.
    """
    changes = {}
    for col in ["Screen Title", "Text", "Estimated Duration"]:
        old_val = old_row[col]
        new_val = new_row[col]
        if old_val != new_val:
            changes[col] = {"old": old_val, "new": new_val}

    if changes:
        log_entry = {
            "screen_index": screen_index,
            "changes": changes
        }
        log_usage_to_db(log_entry)


############################################
# 2) Advanced AI Setup
############################################

def load_embedding_model():
    """
    Load a small sentence-transformer for chunk alignment checks.
    """
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Optionally set up an LLM for summarization (placeholder).
# import openai
# openai.api_key = "YOUR_OPENAI_KEY"

def call_llm_for_summary(paragraph):
    """
    Stub for LLM-based summarization. 
    Must ensure no new facts are introduced, so do a diff check if you implement for real.
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

# If you want OCR for images, define parse_image_ocr here
# def parse_image_ocr(uploaded_image):
#     from PIL import Image
#     import pytesseract
#     image = Image.open(uploaded_image)
#     text = pytesseract.image_to_string(image)
#     return text


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
    st.write("Upload PDF/DOCX/image or paste text for each category. No new content will be invented—only refined or reorganized if needed.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Textbook Content")
        tb_file = st.file_uploader("Upload PDF, DOCX, or image for textbook content",
                                   type=["pdf","docx","png","jpg","jpeg"],
                                   key="textbook_file")
        tb_text_fallback = st.text_area("Or paste textbook text below", height=150)

        # parse if file
        textbook_parsed = ""
        if tb_file is not None:
            fname = tb_file.name.lower()
            if fname.endswith(".pdf"):
                textbook_parsed = parse_pdf(tb_file)
            elif fname.endswith(".docx"):
                textbook_parsed = parse_docx(tb_file)
            elif fname.endswith((".png",".jpg",".jpeg")):
                # placeholder for OCR
                textbook_parsed = "(Placeholder) OCR for textbook not implemented."
            else:
                st.warning("Unsupported file type for textbook content.")
        final_textbook = textbook_parsed.strip() or tb_text_fallback.strip()

    with col2:
        st.subheader("SME Content")
        sme_file = st.file_uploader("Upload PDF, DOCX, or image for SME content",
                                    type=["pdf","docx","png","jpg","jpeg"],
                                    key="sme_file")
        sme_text_fallback = st.text_area("Or paste SME text below", height=150)

        sme_parsed = ""
        if sme_file is not None:
            fname2 = sme_file.name.lower()
            if fname2.endswith(".pdf"):
                sme_parsed = parse_pdf(sme_file)
            elif fname2.endswith(".docx"):
                sme_parsed = parse_docx(sme_file)
            elif fname2.endswith((".png",".jpg",".jpeg")):
                sme_parsed = "(Placeholder) OCR for SME not implemented."
            else:
                st.warning("Unsupported file type for SME content.")
        final_sme = sme_parsed.strip() or sme_text_fallback.strip()

    include_video = st.checkbox("Include Concept Teaching Video?")

    if st.button("Next"):
        st.session_state["textbook_text"] = final_textbook
        st.session_state["sme_text"] = final_sme
        st.session_state["include_video"] = include_video

        st.session_state["page"] = 2

############################################
# Page 2: Analyze (Chunk + Summaries)
############################################
def page_analyze():
    st.title("Lesson Builder: Step 3 - Analyze (Chunk + Summaries)")

    metadata = st.session_state.get("metadata", {})
    objective = metadata.get("Lesson Objective","")
    textbook = st.session_state.get("textbook_text","")
    sme = st.session_state.get("sme_text","")

    combined_text = (textbook + "\n" + sme).strip()
    if not combined_text:
        st.warning("No combined content found. Please go back to Step 2.")
        return

    st.write("We'll chunk the combined text, measure alignment with objective, and do optional LLM summarization. No new content is introduced—just reorganizing or summarizing existing text.")

    if st.button("Analyze Now"):
        paragraphs = [p.strip() for p in combined_text.split("\n") if p.strip()]
        df_analysis = analyze_chunks_with_llm(paragraphs, objective)
        st.session_state["analysis_df"] = df_analysis
        st.success("Analysis complete! Scroll down to view results.")

    if "analysis_df" in st.session_state:
        df_show = st.session_state["analysis_df"]
        st.dataframe(df_show)
        total_words = df_show["word_count"].sum()
        est_minutes = total_words / 140.0
        st.write(f"Total words: {total_words}, approx {est_minutes:.1f} minutes.")
        if est_minutes < 10:
            st.warning("Might be under 15 minutes. Possibly need more SME content.")
        elif est_minutes > 15:
            st.warning("Likely exceeds 15 minutes. Consider trimming or splitting content.")

        if st.button("Next: Generate Outline"):
            st.session_state["page"] = 3

def analyze_chunks_with_llm(paragraphs, objective):
    """
    For each paragraph:
      - compute alignment score to the objective (embedding-based).
      - LLM summary stub (no new facts).
    """
    data = []
    objective_emb = embedding_model.encode(objective, convert_to_tensor=True)
    for i, para in enumerate(paragraphs):
        para_emb = embedding_model.encode(para, convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(para_emb, objective_emb)[0][0])
        word_count = len(para.split())

        # Summarize with LLM (placeholder)
        summary = call_llm_for_summary(para)

        data.append({
            "chunk_id": i+1,
            "original_paragraph": para,
            "alignment_score": round(sim_score, 3),
            "word_count": word_count,
            "llm_summary": summary
        })
    return pd.DataFrame(data)

############################################
# Page 3: Generate Outline
############################################
def page_generate():
    st.title("Lesson Builder: Step 4 - Generate Outline")
    metadata = st.session_state.get("metadata", {})
    textbook_text = st.session_state.get("textbook_text","")
    sme_text = st.session_state.get("sme_text","")
    include_video = st.session_state.get("include_video", False)

    if st.button("Generate Screens"):
        df_screens = generate_screens(metadata, textbook_text, sme_text, include_video)
        st.session_state["screens_df"] = df_screens
        st.success("Lesson screens generated. Scroll down to preview.")

    df_screens = st.session_state.get("screens_df", pd.DataFrame())
    if not df_screens.empty:
        st.dataframe(df_screens)
        if st.button("Next: Refine & Finalize"):
            st.session_state["page"] = 4

def generate_screens(metadata, textbook_text, sme_text, include_video):
    """
    Build ~8-10 screens referencing user text, no new content invented.
    """
    screens = []
    total_duration = 0

    # Intro
    screens.append({
        "Screen Number": 1,
        "Screen Title": "Introduction / Hook",
        "Screen Type": "Text and Graphic",
        "Template": "Canvas",
        "Estimated Duration": "2 minutes",
        "Text": f"Welcome to {metadata.get('Lesson Type','')}! Objective: {metadata.get('Lesson Objective','')}\n\nHook scenario might go here.",
        "Content Source": "Placeholder"
    })
    total_duration += 2

    # Key Concept #1 (Textbook)
    screens.append({
        "Screen Number": 2,
        "Screen Title": "Key Concept 1",
        "Screen Type": "Text and Graphic",
        "Template": "Accordion",
        "Estimated Duration": "2 minutes",
        "Text": textbook_text or "No textbook content provided.",
        "Content Source": "User Provided"
    })
    total_duration += 2

    # Key Concept #2 (SME)
    screens.append({
        "Screen Number": 3,
        "Screen Title": "Key Concept 2",
        "Screen Type": "Text and Graphic",
        "Template": "Canvas",
        "Estimated Duration": "2 minutes",
        "Text": sme_text or "No SME content provided.",
        "Content Source": "User Provided"
    })
    total_duration += 2

    # Practice #1
    screens.append({
        "Screen Number": 4,
        "Screen Title": "Check Your Understanding #1",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "Short scenario or question derived from user text. No new info.",
        "Content Source": "Placeholder"
    })
    total_duration += 1

    # Animation
    screens.append({
        "Screen Number": 5,
        "Screen Title": "Concept Animation",
        "Screen Type": "Animation Placeholder",
        "Template": "Animation",
        "Estimated Duration": "2 minutes",
        "Text": "If you have an animation, link it here.",
        "Content Source": "Placeholder"
    })
    total_duration += 2

    # optional teaching video
    screen_index = 6
    if include_video:
        screens.append({
            "Screen Number": screen_index,
            "Screen Title": "Concept Teaching Video",
            "Screen Type": "Video Placeholder",
            "Template": "Video",
            "Estimated Duration": "2 minutes",
            "Text": "Include a user-chosen video with no new content from AI.",
            "Content Source": "Placeholder"
        })
        total_duration += 2
        screen_index += 1

    # advanced organizer
    screens.append({
        "Screen Number": screen_index,
        "Screen Title": "Advanced Organizer #1",
        "Screen Type": "Text and Graphic",
        "Template": "Complex Illustration",
        "Estimated Duration": "1 minute",
        "Text": "A diagram or infographic summarizing the lesson so far.",
        "Content Source": "Placeholder"
    })
    total_duration += 1
    screen_index += 1

    # second practice
    screens.append({
        "Screen Number": screen_index,
        "Screen Title": "Check Your Understanding #2",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "Another short interactive scenario or question. No new info.",
        "Content Source": "Placeholder"
    })
    total_duration += 1
    screen_index += 1

    # reflection
    screens.append({
        "Screen Number": screen_index,
        "Screen Title": "Reflection / Think About This",
        "Screen Type": "Text and Graphic",
        "Template": "Reflection",
        "Estimated Duration": "1 minute",
        "Text": "Helps learners synthesize and apply concepts to real-world.",
        "Content Source": "Placeholder"
    })
    total_duration += 1

    st.write(f"Approx total duration: ~{total_duration} minutes.")
    return pd.DataFrame(screens)

############################################
# PAGE 4: REFINE
############################################
def page_refine():
    st.title("Step 5: Refine & Finalize")
    df = st.session_state.get("screens_df", pd.DataFrame())
    if df.empty:
        st.write("No screens to refine. Please go back and generate first.")
        return

    st.write("Below are your generated screens. Editing text/duration logs changes to MySQL usage_logs.")

    updated_rows = []
    for i, row in df.iterrows():
        with st.expander(f"Screen {row['Screen Number']}: {row['Screen Title']}"):
            new_title = st.text_input("Screen Title", row["Screen Title"], key=f"title_{i}")
            new_text = st.text_area("Text", row["Text"], key=f"text_{i}")
            new_duration = st.text_input("Estimated Duration", row["Estimated Duration"], key=f"duration_{i}")
            updated_rows.append((i, new_title, new_text, new_duration))

    if st.button("Apply Changes"):
        for (idx, t, x, d) in updated_rows:
            old_row = df.loc[idx].copy()
            df.at[idx, "Screen Title"] = t
            df.at[idx, "Text"] = x
            df.at[idx, "Estimated Duration"] = d
            # log difference
            log_user_change_db(idx, old_row, df.loc[idx])

        st.session_state["screens_df"] = df
        st.success("Changes applied & usage logs stored in MySQL!")
        st.dataframe(df)

    if st.button("Finish"):
        st.write("Lesson workflow complete!")
        st.balloons()
        # optional CSV download
        csv_data = df.to_csv(index=False)
        st.download_button("Download Final Screens as CSV", csv_data, "final_screens.csv", "text/csv")


if __name__ == "__main__":
    main()
