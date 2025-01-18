import streamlit as st
import mysql.connector
import datetime
import pandas as pd

# For chunk alignment and optional LLM usage
import torch
from sentence_transformers import SentenceTransformer, util
# For PDF/DOCX parsing
import PyPDF2
from docx import Document
# (Optional) for images with OCR
# import pytesseract
# from PIL import Image
# etc.

###################################
# 1. DB & AI Setup
###################################

def get_db_connection():
    """
    Connect to your Hostinger MySQL database.
    UPDATE these credentials with your actual values:
      host, user, password, database
    If your domain is '127.0.0.1' or 'localhost' with port 3306, use those.
    """
    host = "127.0.0.1"  # or "localhost"
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

def log_usage_to_db(log_entry):
    """
    Insert a usage log row into usage_logs.
    Make sure usage_logs table exists:
    CREATE TABLE usage_logs (
      id INT AUTO_INCREMENT PRIMARY KEY,
      log_time TIMESTAMP NOT NULL,
      screen_index INT,
      changes TEXT
    );
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
    Compare old vs. new, store diffs as usage logs in MySQL.
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

# Load embedding model (for alignment checks)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

###################################
# 2. Utility parse functions
###################################

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

def parse_image_ocr(uploaded_image):
    """
    If you want OCR on images, install pytesseract & PIL, then un-comment.
    """
    # from PIL import Image
    # import pytesseract
    # image = Image.open(uploaded_image)
    # text = pytesseract.image_to_string(image)
    # return text
    return "(Placeholder) OCR not implemented."

###################################
# 3. Main Multi-Page App
###################################

def main():
    st.set_page_config(page_title="Advanced Lesson Builder", layout="wide")

    # Initialize multi-step page index
    if "page" not in st.session_state:
        st.session_state["page"] = 0

    # Where we'll store final screens DataFrame
    if "screens_df" not in st.session_state:
        st.session_state["screens_df"] = pd.DataFrame()

    show_progress_in_sidebar()

    # Routing logic
    if st.session_state["page"] == 0:
        page_metadata()
    elif st.session_state["page"] == 1:
        page_content()
    elif st.session_state["page"] == 2:
        page_chunk_check()
    elif st.session_state["page"] == 3:
        page_generate()
    elif st.session_state["page"] == 4:
        page_refine()
    else:
        st.write("Invalid page index!")

def show_progress_in_sidebar():
    st.sidebar.title("Lesson Steps Progress")
    steps = [
        "Metadata",
        "Content Collection",
        "Analyze Alignment",
        "Generate Screens",
        "Refine & Finalize"
    ]
    current_page = st.session_state["page"]
    for i, step_name in enumerate(steps):
        if i < current_page:
            st.sidebar.write(f"✅ {step_name} - Done")
        elif i == current_page:
            st.sidebar.write(f"▶️ {step_name} - In Progress")
        else:
            st.sidebar.write(f"⬜ {step_name} - Pending")

###################################
# Page 0: Metadata
###################################
def page_metadata():
    st.title("Lesson Builder: Step 1 - Metadata")
    st.write("Enter your high-level metadata for this lesson.")

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

###################################
# Page 1: Content
###################################
def page_content():
    st.title("Lesson Builder: Step 2 - Content Collection")
    st.write("You can upload PDF/DOCX/images or paste text for both Textbook and SME content. No new content will be invented—only reorganized/refined if needed.")

    col1, col2 = st.columns(2)

    # Textbook
    with col1:
        st.subheader("Textbook Content")
        tb_file = st.file_uploader(
            "Upload PDF, DOCX, or image for textbook content",
            type=["pdf","docx","png","jpg","jpeg"],
            key="tb_file"
        )
        tb_text_fallback = st.text_area("Or paste textbook text below", height=150)
        textbook_parsed = ""

        if tb_file is not None:
            if tb_file.name.lower().endswith(".pdf"):
                textbook_parsed = parse_pdf(tb_file)
            elif tb_file.name.lower().endswith(".docx"):
                textbook_parsed = parse_docx(tb_file)
            elif tb_file.name.lower().endswith((".png",".jpg",".jpeg")):
                textbook_parsed = parse_image_ocr(tb_file)
            else:
                st.warning("Unsupported file type for textbook content.")

        final_textbook_text = textbook_parsed.strip() or tb_text_fallback.strip()

    # SME
    with col2:
        st.subheader("SME Content")
        sme_file = st.file_uploader(
            "Upload PDF, DOCX, or image for SME content",
            type=["pdf","docx","png","jpg","jpeg"],
            key="sme_file"
        )
        sme_text_fallback = st.text_area("Or paste SME text below", height=150)
        sme_parsed = ""

        if sme_file is not None:
            if sme_file.name.lower().endswith(".pdf"):
                sme_parsed = parse_pdf(sme_file)
            elif sme_file.name.lower().endswith(".docx"):
                sme_parsed = parse_docx(sme_file)
            elif sme_file.name.lower().endswith((".png",".jpg",".jpeg")):
                sme_parsed = parse_image_ocr(sme_file)
            else:
                st.warning("Unsupported file type for SME content.")

        final_sme_text = sme_parsed.strip() or sme_text_fallback.strip()

    # Option for concept teaching video
    include_video = st.checkbox("Include Concept Teaching Video?")

    if st.button("Next"):
        st.session_state["textbook_text"] = final_textbook_text
        st.session_state["sme_text"] = final_sme_text
        st.session_state["include_video"] = include_video

        st.session_state["page"] = 2

###################################
# Page 2: Chunk + Alignment Check
###################################
def page_chunk_check():
    st.title("Lesson Builder: Step 3 - Analyze Alignment")
    st.write("We'll chunk your text, check alignment with the lesson objective, and do a rough word count for duration.")

    metadata = st.session_state.get("metadata", {})
    objective = metadata.get("Lesson Objective", "")
    textbook_text = st.session_state.get("textbook_text", "")
    sme_text = st.session_state.get("sme_text", "")

    combined_text = (textbook_text + "\n" + sme_text).strip()
    if not combined_text:
        st.warning("No content found. Please go back and provide content.")
        return

    if st.button("Analyze"):
        paragraphs = [p.strip() for p in combined_text.split("\n") if p.strip()]
        results_df = analyze_chunks(paragraphs, objective)
        st.session_state["analysis_df"] = results_df
        st.success("Analysis complete!")

    if "analysis_df" in st.session_state:
        df = st.session_state["analysis_df"]
        st.dataframe(df)
        total_words = df["word_count"].sum()
        minutes_est = total_words / 140.0
        st.write(f"Total word count: {total_words}, approx. {minutes_est:.1f} minutes.")
        if minutes_est < 10:
            st.warning("Might be less than 15 minutes. Consider adding more content.")
        elif minutes_est > 15:
            st.warning("Likely more than 15 minutes. You may need to split or shorten content.")

        if st.button("Next: Generate Screens"):
            st.session_state["page"] = 3

def analyze_chunks(paragraphs, objective):
    data = []
    obj_emb = embedding_model.encode(objective, convert_to_tensor=True)
    for i, para in enumerate(paragraphs):
        para_emb = embedding_model.encode(para, convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(para_emb, obj_emb)[0][0])
        word_count = len(para.split())
        data.append({
            "chunk_id": i+1,
            "paragraph": para,
            "alignment_score": round(sim_score, 3),
            "word_count": word_count
        })
    return pd.DataFrame(data)

###################################
# Page 3: Generate
###################################
def page_generate():
    st.title("Lesson Builder: Step 4 - Generate Screens")
    st.write("We’ll build an 8–10 screen outline referencing your content, then you can refine it.")

    metadata = st.session_state.get("metadata", {})
    combined_text = (st.session_state.get("textbook_text","") + "\n" + st.session_state.get("sme_text","")).strip()
    include_video = st.session_state.get("include_video", False)

    if st.button("Generate Outline"):
        screens_df = generate_screens(metadata, combined_text, include_video)
        st.session_state["screens_df"] = screens_df
        st.success("Screens generated. See below.")

    df_screens = st.session_state.get("screens_df", pd.DataFrame())
    if not df_screens.empty:
        st.dataframe(df_screens)
        if st.button("Next: Refine & Finalize"):
            st.session_state["page"] = 4

def generate_screens(metadata, combined_text, include_video):
    """
    Basic function to produce 8-10 screens referencing the combined text.
    """
    screens = []
    total_duration = 0

    # 1) Intro
    screens.append({
        "Screen Number": 1,
        "Screen Title": "Introduction / Hook",
        "Screen Type": "Text and Graphic",
        "Template": "Canvas",
        "Estimated Duration": "2 minutes",
        "Text": f"Welcome to {metadata.get('Lesson Type','')}! Objective: {metadata.get('Lesson Objective','')}\n\nHook scenario could go here.",
        "Content Source": "Placeholder"
    })
    total_duration += 2

    # For demonstration, split combined_text in half
    splitted = combined_text.split()
    mid = len(splitted)//2
    text_part1 = " ".join(splitted[:mid])
    text_part2 = " ".join(splitted[mid:])

    # 2) Key Concept 1
    screens.append({
        "Screen Number": 2,
        "Screen Title": "Key Concept 1",
        "Screen Type": "Text and Graphic",
        "Template": "Accordion",
        "Estimated Duration": "2 minutes",
        "Text": text_part1 or "No content provided.",
        "Content Source": "User Content"
    })
    total_duration += 2

    # 3) Key Concept 2
    screens.append({
        "Screen Number": 3,
        "Screen Title": "Key Concept 2",
        "Screen Type": "Text and Graphic",
        "Template": "Canvas",
        "Estimated Duration": "2 minutes",
        "Text": text_part2 or "No content provided (second half).",
        "Content Source": "User Content"
    })
    total_duration += 2

    # 4) Practice Interactive
    screens.append({
        "Screen Number": 4,
        "Screen Title": "Check Your Understanding #1",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "Placeholder quiz. No new info from AI.",
        "Content Source": "Placeholder"
    })
    total_duration += 1

    # 5) Concept Animation
    screens.append({
        "Screen Number": 5,
        "Screen Title": "Concept Animation",
        "Screen Type": "Animation Placeholder",
        "Template": "Animation",
        "Estimated Duration": "2 minutes",
        "Text": "Link an animation here if you have one.",
        "Content Source": "Placeholder"
    })
    total_duration += 2

    # Optional video
    next_num = 6
    if include_video:
        screens.append({
            "Screen Number": 6,
            "Screen Title": "Concept Teaching Video",
            "Screen Type": "Video Placeholder",
            "Template": "Video",
            "Estimated Duration": "2 minutes",
            "Text": "User-chosen concept video. No new content added.",
            "Content Source": "Placeholder"
        })
        total_duration += 2
        next_num = 7

    # 6) Advanced Organizer
    screens.append({
        "Screen Number": next_num,
        "Screen Title": "Advanced Organizer #1",
        "Screen Type": "Text and Graphic",
        "Template": "Complex Illustration",
        "Estimated Duration": "1 minute",
        "Text": "Infographic summarizing the lesson so far.",
        "Content Source": "Placeholder"
    })
    total_duration += 1
    next_num += 1

    # 7) Another Quiz
    screens.append({
        "Screen Number": next_num,
        "Screen Title": "Check Your Understanding #2",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "Another short quiz. No new info introduced.",
        "Content Source": "Placeholder"
    })
    total_duration += 1
    next_num += 1

    # 8) Reflection
    screens.append({
        "Screen Number": next_num,
        "Screen Title": "Reflection / Think About This",
        "Screen Type": "Text and Graphic",
        "Template": "Reflection",
        "Estimated Duration": "1 minute",
        "Text": "End with a reflection screen to help learners apply concepts.",
        "Content Source": "Placeholder"
    })
    total_duration += 1

    st.write(f"Approx total lesson duration: ~{total_duration} minutes")
    return pd.DataFrame(screens)

###################################
# Page 4: Refine
###################################
def page_refine():
    st.title("Lesson Builder: Step 5 - Refine & Finalize")
    df = st.session_state.get("screens_df", pd.DataFrame())
    if df.empty:
        st.write("No screens found. Go back and generate first.")
        return

    st.write("Review each screen. Only use your provided content, no new info from AI.")
    updated_rows = []
    for i, row in df.iterrows():
        with st.expander(f"Screen {row['Screen Number']}: {row['Screen Title']}"):
            new_title = st.text_input("Screen Title", row["Screen Title"], key=f"title_{i}")
            new_text = st.text_area("Text", row["Text"], key=f"text_{i}")
            new_duration = st.text_input("Estimated Duration", row["Estimated Duration"], key=f"dur_{i}")
            updated_rows.append((i, new_title, new_text, new_duration))

    if st.button("Apply Changes"):
        # Compare old vs. new, log in MySQL
        for idx, t, x, d in updated_rows:
            old_row = df.loc[idx].copy()
            df.at[idx, "Screen Title"] = t
            df.at[idx, "Text"] = x
            df.at[idx, "Estimated Duration"] = d
            log_user_change_db(idx, old_row, df.loc[idx])
        st.session_state["screens_df"] = df
        st.success("Refinements applied & usage logs saved to DB!")
        st.dataframe(df)

    if st.button("Finish"):
        st.write("Lesson creation complete!")
        st.balloons()

        # Optionally let user download final screens as CSV
        csv_data = df.to_csv(index=False)
        st.download_button(
            "Download Final Screens CSV",
            data=csv_data,
            file_name="final_lesson_screens.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
