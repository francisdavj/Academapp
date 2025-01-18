import streamlit as st
import pandas as pd

def main():
    st.set_page_config(page_title="Lesson Builder", layout="wide")

    # Keep track of which "page" or step the user is on
    if "page" not in st.session_state:
        st.session_state["page"] = 0
    
    # If we don’t already have a DataFrame for screens, initialize an empty one
    if "screens_df" not in st.session_state:
        st.session_state["screens_df"] = pd.DataFrame()

    # Render the current page
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


def page_metadata():
    """
    Page 0: Collect high-level lesson metadata.
    """
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
    """
    Page 1: Collect textbook and SME content.
    """
    st.title("Lesson Builder: Step 2 - Content Collection")
    st.write("Paste or type your textbook and SME content below:")

    # Collect text from user
    textbook_text = st.text_area("Textbook Content", height=200)
    sme_text = st.text_area("SME Content", height=200)

    # Optional toggle for concept teaching video
    include_video = st.checkbox("Include Concept Teaching Video?")

    if st.button("Next"):
        st.session_state["textbook_text"] = textbook_text
        st.session_state["sme_text"] = sme_text
        st.session_state["include_video"] = include_video
        st.session_state["page"] = 2


def page_generate():
    """
    Page 2: Generate an outline of lesson screens based on the user's input.
    """
    st.title("Lesson Builder: Step 3 - Generate Lesson Screens")

    # Retrieve stored data
    metadata = st.session_state.get("metadata", {})
    textbook_text = st.session_state.get("textbook_text", "")
    sme_text = st.session_state.get("sme_text", "")
    include_video = st.session_state.get("include_video", False)

    if st.button("Generate Screens"):
        # Build the lesson screens DataFrame
        df = generate_screens(metadata, textbook_text, sme_text, include_video)
        st.session_state["screens_df"] = df
        st.success("Lesson screens generated! Scroll down to preview.")

    if "screens_df" in st.session_state and not st.session_state["screens_df"].empty:
        st.write("Preview of generated lesson screens:")
        st.dataframe(st.session_state["screens_df"])

        if st.button("Next: Refine"):
            st.session_state["page"] = 3


def page_refine():
    """
    Page 3: Let the user refine/edit the generated screens.
    """
    st.title("Lesson Builder: Step 4 - Refine Screens")

    df = st.session_state.get("screens_df", pd.DataFrame())
    if df.empty:
        st.write("No screens to refine. Go back and generate first.")
        return

    # Display each row in an expander for editing
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
        st.success("Refinements applied. Check below for updated screens.")
        st.dataframe(df)

    if st.button("Finish"):
        st.write("Lesson creation workflow complete!")
        st.balloons()


def generate_screens(metadata, textbook_text, sme_text, include_video):
    """
    Example logic that generates an 8-10 screen structure for a ~15 minute lesson
    based on the guidelines we discussed. You can customize to your needs.
    """
    # Basic approach to building a list of screen dictionaries
    screens = []
    total_duration = 0

    # 1) Introduction
    screens.append({
        "Screen Number": 1,
        "Screen Title": "Introduction / Hook",
        "Screen Type": "Text and Graphic",
        "Template": "Canvas",
        "Estimated Duration": "2 minutes",
        "Text": f"Welcome to {metadata.get('Lesson Type','')}! "
                f"This lesson covers: {metadata.get('Lesson Objective','')}\n\n"
                "Hook scenario or question goes here.",
        "Content Source": "AI Generated"
    })
    total_duration += 2

    # 2) Key Concept 1 from textbook
    screens.append({
        "Screen Number": 2,
        "Screen Title": "Key Concept 1",
        "Screen Type": "Text and Graphic",
        "Template": "Accordion",
        "Estimated Duration": "2 minutes",
        "Text": textbook_text or "No textbook content provided yet.",
        "Content Source": "Textbook"
    })
    total_duration += 2

    # 3) Key Concept 2 (some filler text)
    screens.append({
        "Screen Number": 3,
        "Screen Title": "Key Concept 2",
        "Screen Type": "Text and Graphic",
        "Template": "Canvas",
        "Estimated Duration": "2 minutes",
        "Text": "Additional details or second concept elaboration.",
        "Content Source": "AI Generated"
    })
    total_duration += 2

    # 4) 1st Practice Interactive
    screens.append({
        "Screen Number": 4,
        "Screen Title": "Check Your Understanding #1",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "It’s time to apply what you learned with a short scenario or MCQ.",
        "Content Source": "AI Generated"
    })
    total_duration += 1

    # 5) Concept Animation
    screens.append({
        "Screen Number": 5,
        "Screen Title": "Concept Animation",
        "Screen Type": "Animation Placeholder",
        "Template": "Animation",
        "Estimated Duration": "2 minutes",
        "Text": "Introduce the animation concept, how to play it, etc.",
        "Content Source": "AI Generated"
    })
    total_duration += 2

    # (Optional) Concept Video
    if include_video:
        screens.append({
            "Screen Number": 6,
            "Screen Title": "Concept Teaching Video",
            "Screen Type": "Video Placeholder",
            "Template": "Video Player",
            "Estimated Duration": "2 minutes",
            "Text": "Introduction to the video content, instructions to watch, etc.",
            "Content Source": "AI Generated"
        })
        total_duration += 2

    # Next screens, advanced organizers, 2nd practice interactive, reflection...
    # We'll add a few more placeholders:

    # Advanced Organizer #1
    screens.append({
        "Screen Number": len(screens)+1,
        "Screen Title": "Advanced Organizer #1",
        "Screen Type": "Text and Graphic",
        "Template": "Complex Illustration",
        "Estimated Duration": "1 minute",
        "Text": "A diagram or infographic summarizing key concepts.",
        "Content Source": "AI Generated"
    })
    total_duration += 1

    # 2nd Practice Interactive
    screens.append({
        "Screen Number": len(screens)+1,
        "Screen Title": "Check Your Understanding #2",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "Another short interactive to reinforce learning.",
        "Content Source": "AI Generated"
    })
    total_duration += 1

    # Reflection
    screens.append({
        "Screen Number": len(screens)+1,
        "Screen Title": "Reflection / Think About This",
        "Screen Type": "Text and Graphic",
        "Template": "Reflection",
        "Estimated Duration": "1 minute",
        "Text": "Help the learner apply these concepts to real-world scenarios.",
        "Content Source": "AI Generated"
    })
    total_duration += 1

    # Summarize total
    st.write(f"Approximate total lesson duration: ~{total_duration} minutes")

    return pd.DataFrame(screens)


if __name__ == "__main__":
    main()
