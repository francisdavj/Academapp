import streamlit as st
import pandas as pd

def main():
    # Basic Streamlit page config
    st.set_page_config(page_title="Lesson Builder", layout="wide")

    # Initialize page index and DataFrame if not in session_state
    if "page" not in st.session_state:
        st.session_state["page"] = 0

    if "screens_df" not in st.session_state:
        st.session_state["screens_df"] = pd.DataFrame()

    # Render sidebar progress menu
    show_progress_in_sidebar()

    # Display current page
    if st.session_state["page"] == 0:
        page_metadata()
    elif st.session_state["page"] == 1:
        page_content()
    elif st.session_state["page"] == 2:
        page_generate()
    elif st.session_state["page"] == 3:
        page_refine()
    else:
        st.write("Invalid page index!")  

def show_progress_in_sidebar():
    """
    Renders a sidebar section with the steps of the workflow.
    Indicates completion status based on st.session_state["page"].
    """
    st.sidebar.title("Lesson Steps Progress")

    steps = ["Metadata", "Content", "Generate", "Refine"]
    current_page = st.session_state["page"]

    for i, step_name in enumerate(steps):
        if i < current_page:
            # Step is completed
            st.sidebar.write(f"✅ {step_name} - Done")
        elif i == current_page:
            # Step is in progress
            st.sidebar.write(f"▶️ {step_name} - In Progress")
        else:
            # Future steps
            st.sidebar.write(f"⬜ {step_name} - Pending")

def page_metadata():
    st.title("Lesson Builder: Step 1 - Metadata")
    st.write("Enter your high-level metadata for the lesson.")

    # Create text inputs for metadata
    course_title = st.text_input("Course # and Title", "")
    module_title = st.text_input("Module # and Title", "")
    unit_title = st.text_input("Unit # and Title", "")
    lesson_title = st.text_input("Lesson # and Title", "")
    lesson_objective = st.text_input("Lesson Objective", "")
    lesson_type = st.selectbox("Lesson Type", ["Core Learning Lesson", "Practice Lesson", "Other"])

    if st.button("Next"):
        # Store metadata in session_state
        st.session_state["metadata"] = {
            "Course and Title": course_title,
            "Module and Title": module_title,
            "Unit and Title": unit_title,
            "Lesson and Title": lesson_title,
            "Lesson Objective": lesson_objective,
            "Lesson Type": lesson_type
        }
        # Move to next page
        st.session_state["page"] = 1

def page_content():
    st.title("Lesson Builder: Step 2 - Content Collection")
    st.write("Paste or type your textbook and SME content below. No new content will be invented, only refined if needed.")

    # Collect text from user
    textbook_text = st.text_area("Textbook Content", height=200)
    sme_text = st.text_area("SME Content", height=200)

    # Toggle for concept teaching video
    include_video = st.checkbox("Include Concept Teaching Video?")

    if st.button("Next"):
        # Store content in session_state
        st.session_state["textbook_text"] = textbook_text
        st.session_state["sme_text"] = sme_text
        st.session_state["include_video"] = include_video

        st.session_state["page"] = 2

def page_generate():
    st.title("Lesson Builder: Step 3 - Generate Lesson Screens")
    st.write("When you click 'Generate Screens,' we'll build ~8-10 placeholders for a 15-minute lesson. Then you can refine them.")

    # Get data from session_state
    metadata = st.session_state.get("metadata", {})
    textbook_text = st.session_state.get("textbook_text", "")
    sme_text = st.session_state.get("sme_text", "")
    include_video = st.session_state.get("include_video", False)

    if st.button("Generate Screens"):
        # Create or call your AI logic to generate the screens
        df = generate_screens(metadata, textbook_text, sme_text, include_video)
        st.session_state["screens_df"] = df
        st.success("Lesson screens generated! Scroll down to preview.")

    if "screens_df" in st.session_state and not st.session_state["screens_df"].empty:
        st.write("Preview of generated lesson screens:")
        st.dataframe(st.session_state["screens_df"])

        if st.button("Next: Refine Screens"):
            st.session_state["page"] = 3

def page_refine():
    st.title("Lesson Builder: Step 4 - Refine Screens")
    st.write("Review and edit each screen. Make sure to only use the content you provided (no new info).")

    df = st.session_state.get("screens_df", pd.DataFrame())
    if df.empty:
        st.write("No screens to refine. Please go back and generate first.")
        return

    updated_rows = []
    for i, row in df.iterrows():
        with st.expander(f"Screen {row['Screen Number']}: {row['Screen Title']}"):
            # Provide text fields for editable columns
            new_title = st.text_input("Screen Title", row["Screen Title"], key=f"title_{i}")
            new_text = st.text_area("Text", row["Text"], key=f"text_{i}")
            new_duration = st.text_input("Estimated Duration", row["Estimated Duration"], key=f"duration_{i}")

            updated_rows.append((i, new_title, new_text, new_duration))

    if st.button("Apply Refinements"):
        for (i, t, x, d) in updated_rows:
            df.at[i, "Screen Title"] = t
            df.at[i, "Text"] = x
            df.at[i, "Estimated Duration"] = d

        st.session_state["screens_df"] = df
        st.success("Refinements applied! Check updated screens below.")
        st.dataframe(df)

    if st.button("Finish"):
        st.write("Lesson creation workflow complete!")
        st.balloons()

def generate_screens(metadata, textbook_text, sme_text, include_video):
    """
    Basic function that simulates the AI building out ~8-10 screens for a 15-minute lesson,
    referencing user-provided content. No new content is introduced.
    """
    screens = []
    total_duration = 0

    # 1. Introduction
    screens.append({
        "Screen Number": 1,
        "Screen Title": "Introduction / Hook",
        "Screen Type": "Text and Graphic",
        "Template": "Canvas",
        "Estimated Duration": "2 minutes",
        "Text": (f"Welcome to {metadata.get('Lesson Type','')}! "
                 f"This lesson covers: {metadata.get('Lesson Objective','')}\n\n"
                 "Hook or scenario from your content might go here."),
        "Content Source": "AI Placeholder"
    })
    total_duration += 2

    # 2. Key Concept 1 (using textbook text)
    screens.append({
        "Screen Number": 2,
        "Screen Title": "Key Concept 1",
        "Screen Type": "Text and Graphic",
        "Template": "Accordion",
        "Estimated Duration": "2 minutes",
        "Text": textbook_text or "No textbook content was provided.",
        "Content Source": "Textbook"
    })
    total_duration += 2

    # 3. Key Concept 2 (using sme_text)
    screens.append({
        "Screen Number": 3,
        "Screen Title": "Key Concept 2",
        "Screen Type": "Text and Graphic",
        "Template": "Canvas",
        "Estimated Duration": "2 minutes",
        "Text": sme_text or "No SME content was provided.",
        "Content Source": "SME"
    })
    total_duration += 2

    # 4. Practice Interactive #1
    screens.append({
        "Screen Number": 4,
        "Screen Title": "Check Your Understanding #1",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "Short scenario or question derived from your content. (No new info)",
        "Content Source": "AI Placeholder"
    })
    total_duration += 1

    # 5. Concept Animation
    screens.append({
        "Screen Number": 5,
        "Screen Title": "Concept Animation",
        "Screen Type": "Animation Placeholder",
        "Template": "Animation",
        "Estimated Duration": "2 minutes",
        "Text": "You can link a short animation here if available. No new content introduced.",
        "Content Source": "Placeholder"
    })
    total_duration += 2

    # Optional concept video
    if include_video:
        screens.append({
            "Screen Number": 6,
            "Screen Title": "Concept Teaching Video",
            "Screen Type": "Video Placeholder",
            "Template": "Video",
            "Estimated Duration": "2 minutes",
            "Text": "If the ID decided to have a video, place it here. No new content from AI.",
            "Content Source": "Placeholder"
        })
        total_duration += 2

    # Next placeholders (advanced organizers, second quiz, reflection)
    next_screen = 6 if not include_video else 7
    screens.append({
        "Screen Number": next_screen,
        "Screen Title": "Advanced Organizer #1",
        "Screen Type": "Text and Graphic",
        "Template": "Complex Illustration",
        "Estimated Duration": "1 minute",
        "Text": "A complex infographic or chart summarizing the lesson so far.",
        "Content Source": "Placeholder"
    })
    total_duration += 1

    next_screen += 1
    screens.append({
        "Screen Number": next_screen,
        "Screen Title": "Check Your Understanding #2",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "A second short practice or scenario-based question.",
        "Content Source": "AI Placeholder"
    })
    total_duration += 1

    next_screen += 1
    screens.append({
        "Screen Number": next_screen,
        "Screen Title": "Reflection / Think About This",
        "Screen Type": "Text and Graphic",
        "Template": "Reflection",
        "Estimated Duration": "1 minute",
        "Text": "End with a reflection screen to help learners apply these concepts to real-world scenarios.",
        "Content Source": "Placeholder"
    })
    total_duration += 1

    st.write(f"Approximate total lesson duration: ~{total_duration} minutes")
    return pd.DataFrame(screens)

if __name__ == "__main__":
    main()
