######################
# app.py
######################
import streamlit as st
import mysql.connector
import datetime
import pandas as pd

import PyPDF2
from docx import Document

# For embeddings + alignment checks
from sentence_transformers import SentenceTransformer, util

# (Optional) If you integrate LLM summarization, e.g., openai
# import openai

############################
# 1) MySQL CONFIG
############################
def get_db_connection():
    """
    Connect to your Hostinger MySQL.
    Update the credentials (host, user, password, database).
    """
    # If you want to store them in st.secrets, do that. 
    # For demonstration, we'll just put placeholders here:

    host = "127.0.0.1"            # or "localhost" if that works
    user = "u628260032_francisdavid"
    password = "Chennai@202475"
    database = "u628260032_academapp"
    port = 3306                   # default MySQL port

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
    Insert usage logs into the usage_logs table:
    usage_logs( id INT AUTO_INCREMENT,
                log_time TIMESTAMP,
                screen_index INT,
                changes TEXT,
                PRIMARY KEY(id) )
    """
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        changes_str = str(log_entry.get("changes", {}))  # or use json.dumps(...) if you prefer
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
    Compare old vs. new row for columns (Screen Title, Text, Estimated Duration).
    If there's a difference, log it to usage_logs.
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

############################
# 2) AI CONFIG
############################

def load_embedding_model():
    """
    Load a small sentence-transformer for chunk alignment checks.
    """
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# If you want LLM summarization, you'd configure openai or another API:
# openai.api_key = "YOUR_OPENAI_KEY"

def call_llm_for_summary(paragraph):
    """
    Example stub for LLM-based summarization. 
    You can integrate GPT or other providers here, 
    always verifying no new facts are introduced.
    """
    # prompt = f"Summarize this paragraph in 1-2 lines without adding new info:\n\n{paragraph}"
    # response = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=prompt,
    #     max_tokens=100,
    #     temperature=0
    # )
    # summary = response["choices"][0]["text"].strip()
    # return summary

    return "LLM summary placeholder (no new content)."


############################
# 3) PARSE FILES
############################

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
    """Extract text from a .docx file using python-docx."""
    text = ""
    try:
        doc = Document(uploaded_docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return text

# def parse_image_ocr(uploaded_image):
#     """If you want to do OCR on images, do something like pytesseract here."""
#     from PIL import Image
#     import pytesseract
#     image = Image.open(uploaded_image)
#     text = pytesseract.image_to_string(image)
#     return text

############################
# 4) MULTI-STEP WORKFLOW
############################

def main():
    st.set_page_config(page_title="Advanced Lesson Builder", layout="wide")

    if "page" not in st.session_state:
        st.session_state["page"] = 0
    if "screens_df" not in st.session_state:
        st.session_state["screens_df"] = pd.DataFrame()

    show_progress_in_sidebar()

    # 0: Metadata, 1: Content, 2: Analyze, 3: Generate, 4: Refine
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

def show_progress_in_sidebar():
    st.sidebar.title("Lesson Steps Progress")
    steps = [
        "Metadata",
        "Content (Upload/Paste)",
        "Analyze (Chunk + Summaries)",
        "Generate Outline",
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

############################
# PAGE 0: METADATA
############################
def page_metadata():
    st.title("Step 1: Lesson Metadata")

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

############################
# PAGE 1: CONTENT
############################
def page_content():
    st.title("Step 2: Content Collection")
    st.write("Upload PDF/DOCX or an image, OR paste text. No new content will be invented, only reorganized if needed.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Textbook Content")
        tb_file = st.file_uploader("Upload PDF, DOCX, or image for textbook", 
                                   type=["pdf","docx","png","jpg","jpeg"], 
                                   key="textbook_file")
        tb_text_fallback = st.text_area("Or paste textbook text here", height=150)

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
                st.warning("Unsupported file type for textbook.")
        final_textbook = textbook_parsed.strip() or tb_text_fallback.strip()

    with col2:
        st.subheader("SME Content")
        sme_file = st.file_uploader("Upload PDF, DOCX, or image for SME", 
                                    type=["pdf","docx","png","jpg","jpeg"], 
                                    key="sme_file")
        sme_text_fallback = st.text_area("Or paste SME text here", height=150)

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
                st.warning("Unsupported file type for SME.")
        final_sme = sme_parsed.strip() or sme_text_fallback.strip()

    include_video = st.checkbox("Include Concept Teaching Video?")

    if st.button("Next"):
        st.session_state["textbook_text"] = final_textbook
        st.session_state["sme_text"] = final_sme
        st.session_state["include_video"] = include_video
        st.session_state["page"] = 2

############################
# PAGE 2: ANALYZE (CHUNK + SUMMARIES)
############################
def page_analyze():
    st.title("Step 3: Analyze Content (Chunk + LLM Summaries)")

    metadata = st.session_state.get("metadata", {})
    objective = metadata.get("Lesson Objective","")
    textbook = st.session_state.get("textbook_text","")
    sme = st.session_state.get("sme_text","")

    if not objective.strip():
        st.warning("No lesson objective found. Please go back to Step 1.")
        return

    combined_text = (textbook + "\n" + sme).strip()
    if not combined_text:
        st.warning("No content found. Please go back to Step 2.")
        return

    if st.button("Analyze Now"):
        paragraphs = [p.strip() for p in combined_text.split("\n") if p.strip()]
        df_analysis = analyze_chunks_with_llm(paragraphs, objective)
        st.session_state["analysis_df"] = df_analysis
        st.success("Analysis complete. See below.")

    if "analysis_df" in st.session_state:
        df_show = st.session_state["analysis_df"]
        st.dataframe(df_show)
        total_words = df_show["word_count"].sum()
        est_minutes = total_words / 140.0
        st.write(f"Total words: {total_words}, approx {est_minutes:.1f} minutes.")
        if est_minutes < 10:
            st.warning("Might be under 15 minutes. Possibly need more content or SME input.")
        elif est_minutes > 15:
            st.warning("Likely exceeds 15 minutes. Consider splitting or removing redundancies.")

        if st.button("Next: Generate Outline"):
            st.session_state["page"] = 3

def analyze_chunks_with_llm(paragraphs, objective):
    """
    For each paragraph:
      - embedding alignment to objective
      - optional LLM summary
    """
    data = []
    objective_emb = embedding_model.encode(objective, convert_to_tensor=True)
    for i, para in enumerate(paragraphs):
        para_emb = embedding_model.encode(para, convert_to_tensor=True)
        sim_score = float(util.pytorch_cos_sim(para_emb, objective_emb)[0][0])
        word_count = len(para.split())

        # Summarize with LLM (placeholder)
        summary = call_llm_for_summary(para)
        # If you want to do a diff check that summary doesn't add new facts,
        # you'd implement that here.

        data.append({
            "chunk_id": i+1,
            "original_paragraph": para,
            "alignment_score": round(sim_score, 3),
            "word_count": word_count,
            "llm_summary": summary
        })
    return pd.DataFrame(data)

############################
# PAGE 3: GENERATE OUTLINE
############################
def page_generate():
    st.title("Step 4: Generate Lesson Outline")

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
        if st.button("Next: Refine + Finalize"):
            st.session_state["page"] = 4

def generate_screens(metadata, textbook_text, sme_text, include_video):
    """
    Create ~8-10 screens for a ~15-min lesson, referencing user text
    without inventing new content.
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
        "Text": f"Welcome to {metadata.get('Lesson Type','')}! Objective: {metadata.get('Lesson Objective','')}\n\nHook or scenario here.",
        "Content Source": "Placeholder"
    })
    total_duration += 2

    # 2) Key Concept #1 from textbook
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

    # 3) Key Concept #2 from SME
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

    # 4) Practice Interactive #1
    screens.append({
        "Screen Number": 4,
        "Screen Title": "Check Your Understanding #1",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "Placeholder quiz from user text (no new info).",
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
        "Text": "Link an animation here if you have it. No new content.",
        "Content Source": "Placeholder"
    })
    total_duration += 2

    # optional concept video
    screen_num = 6
    if include_video:
        screens.append({
            "Screen Number": screen_num,
            "Screen Title": "Concept Teaching Video",
            "Screen Type": "Video Placeholder",
            "Template": "Video",
            "Estimated Duration": "2 minutes",
            "Text": "User-chosen video. No new content from AI.",
            "Content Source": "Placeholder"
        })
        total_duration += 2
        screen_num += 1

    # advanced organizer #1
    screens.append({
        "Screen Number": screen_num,
        "Screen Title": "Advanced Organizer #1",
        "Screen Type": "Text and Graphic",
        "Template": "Complex Illustration",
        "Estimated Duration": "1 minute",
        "Text": "Diagram or infographic summarizing the lesson so far.",
        "Content Source": "Placeholder"
    })
    total_duration += 1
    screen_num += 1

    # practice #2
    screens.append({
        "Screen Number": screen_num,
        "Screen Title": "Check Your Understanding #2",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "Another short interactive scenario or question. No new content.",
        "Content Source": "Placeholder"
    })
    total_duration += 1
    screen_num += 1

    # reflection
    screens.append({
        "Screen Number": screen_num,
        "Screen Title": "Reflection / Think About This",
        "Screen Type": "Text and Graphic",
        "Template": "Reflection",
        "Estimated Duration": "1 minute",
        "Text": "Help learners synthesize concepts and apply them to real-world scenarios.",
        "Content Source": "Placeholder"
    })
    total_duration += 1

    st.write(f"Approx. total duration: ~{total_duration} minutes.")
    return pd.DataFrame(screens)

############################
# PAGE 4: REFINE
############################
def page_refine():
    st.title("Step 5: Refine & Finalize")

    df = st.session_state.get("screens_df", pd.DataFrame())
    if df.empty:
        st.write("No screens to refine. Please go back and generate first.")
        return

    st.write("Review each screen. If you edit text/duration, changes get logged to MySQL (usage_logs).")

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
            # Log the difference
            log_user_change_db(idx, old_row, df.loc[idx])

        st.session_state["screens_df"] = df
        st.success("Refinements applied & usage logs stored in MySQL!")
        st.dataframe(df)

    if st.button("Finish"):
        st.write("Lesson creation workflow complete!")
        st.balloons()
        # optional: allow CSV/Excel download
        csv_data = df.to_csv(index=False)
        st.download_button("Download Final Screens as CSV", csv_data, "final_screens.csv", "text/csv")


if __name__ == "__main__":
    main()
