import streamlit as st
import pandas as pd

def main():
    """
    Entry point for the multi-step lesson builder.
    We've expanded the idea of 'AI Build -> Refine' with placeholders.
    """
    # Basic Streamlit page setup
    st.set_page_config(page_title="Lesson Builder", layout="wide")

    # Keep track of which "page" or step the user is on
    if "page" not in st.session_state:
        st.session_state["page"] = 0

    # If we don’t already have a DataFrame for screens, initialize an empty one
    if "screens_df" not in st.session_state:
        st.session_state["screens_df"] = pd.DataFrame()

    # Choose which page to display based on the step
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
    Page 0: Collect high-level lesson metadata (Course, Module, etc.).
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
    st.write("Paste or type your textbook and SME content below. We will not add new material—only refine or reorganize existing text where needed.")

    # Collect text from user
    textbook_text = st.text_area("Textbook Content", height=200)
    sme_text = st.text_area("SME Content", height=200)

    # Optional toggle for a concept teaching video
    include_video = st.checkbox("Include Concept Teaching Video?")

    if st.button("Next"):
        # Store content in session_state
        st.session_state["textbook_text"] = textbook_text
        st.session_state["sme_text"] = sme_text
        st.session_state["include_video"] = include_video

        # Move to the AI (or code logic) build step
        st.session_state["page"] = 2

def page_generate():
    """
    Page 2: Automatically build the lesson screens from the inputs (AI Build).
    The ID can see a preview and then refine in the next step.
    """
    st.title("Lesson Builder: Step 3 - AI/Code Generated Lesson Screens")

    # Get data from session_state
    metadata = st.session_state.get("metadata", {})
    textbook_text = st.session_state.get("textbook_text", "")
    sme_text = st.session_state.get("sme_text", "")
    include_video = st.session_state.get("include_video", False)

    st.write("When you click 'Generate Screens,' the system (or an AI) will create an initial outline of ~8-10 screens.")
    st.write("No new content will be invented—only the text you provided will be used or rearranged if necessary.")

    if st.button("Generate Screens"):
        # In a real scenario, you might call an AI API or run code that organizes your content.
        # Here, we use a local function that sets up a standard 8-10 screen structure.
        df = generate_screens(metadata, textbook_text, sme_text, include_video)
        st.session_state["screens_df"] = df
        st.success("Lesson screens generated! Scroll down to preview.")

    # Show the generated screens if available
    if "screens_df" in st.session_state and not st.session_state["screens_df"].empty:
        st.write("Preview of generated lesson screens:")
        st.dataframe(st.session_state["screens_df"])

        if st.button("Next: Refine Screens"):
            st.session_state["page"] = 3

def page_refine():
    """
    Page 3: The ID refines the auto-generated screens—manually editing or removing any extra text.
    """
    st.title("Lesson Builder: Step 4 - Refine Screens")

    df = st.session_state.get("screens_df", pd.DataFrame())
    if df.empty:
        st.write("No screens to refine. Go back and generate first.")
        return

    st.write("Below are the screens we generated. You can edit their titles, text, and duration. Only use the content from your textbook/SME if you need to fill in details. No new content from AI is allowed.")

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
        st.success("Refinements applied! The updated screens are below.")
        st.dataframe(df)

    if st.button("Finish"):
        st.write("Lesson creation workflow complete!")
        st.balloons()

def generate_screens(metadata, textbook_text, sme_text, include_video):
    """
    Basic function that simulates the AI building out ~8-10 screens for a 15-minute lesson.
    We only reorganize the user-provided text and insert placeholders for additional screens.
    """
    screens = []
    total_duration = 0

    # Screen 1: Introduction
    screens.append({
        "Screen Number": 1,
        "Screen Title": "Introduction / Hook",
        "Screen Type": "Text and Graphic",
        "Template": "Canvas",
        "Estimated Duration": "2 minutes",
        "Text": (f"Welcome to {metadata.get('Lesson Type','a lesson')}! "
                 f"This lesson covers: {metadata.get('Lesson Objective','')}\n\n"
                 "Hook or scenario from your content might go here."),
        "Content Source": "AI/Code Generated"
    })
    total_duration += 2

    # Screen 2: Key Concept 1 (using textbook_text)
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

    # Screen 3: Key Concept 2 (we can do minimal additions but not new content)
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

    # Screen 4: Practice Interactive #1
    screens.append({
        "Screen Number": 4,
        "Screen Title": "Check Your Understanding #1",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "A short scenario or question derived from your content. (No new info)",
        "Content Source": "AI Placeholder"
    })
    total_duration += 1

    # Screen 5: Concept Animation
    screens.append({
        "Screen Number": 5,
        "Screen Title": "Concept Animation",
        "Screen Type": "Animation Placeholder",
        "Template": "Animation",
        "Estimated Duration": "2 minutes",
        "Text": "You can link a short animation here if you have one from the SME or a library.",
        "Content Source": "Placeholder"
    })
    total_duration += 2

    # (Optional) Concept Teaching Video
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

    # Let's add more placeholders for advanced organizers & second quiz

    next_screen_number = 6 if not include_video else 7
    screens.append({
        "Screen Number": next_screen_number,
        "Screen Title": "Advanced Organizer #1",
        "Screen Type": "Text and Graphic",
        "Template": "Complex Illustration",
        "Estimated Duration": "1 minute",
        "Text": "A complex infographic or chart summarizing the lesson so far.",
        "Content Source": "Placeholder"
    })
    total_duration += 1

    next_screen_number += 1
    screens.append({
        "Screen Number": next_screen_number,
        "Screen Title": "Check Your Understanding #2",
        "Screen Type": "Practice Interactive",
        "Template": "Quiz",
        "Estimated Duration": "1 minute",
        "Text": "A second short practice or scenario-based question.",
        "Content Source": "AI Placeholder"
    })
    total_duration += 1

    next_screen_number += 1
    screens.append({
        "Screen Number": next_screen_number,
        "Screen Title": "Reflection / Think About This",
        "Screen Type": "Text and Graphic",
        "Template": "Reflection",
        "Estimated Duration": "1 minute",
        "Text": "End the lesson with a reflection screen to help learners synthesize the concepts.",
        "Content Source": "Placeholder"
    })
    total_duration += 1

    # Summarize total
    st.write(f"Approximate total lesson duration: ~{total_duration} minutes")

    # Return the final DataFrame
    return pd.DataFrame(screens)


if __name__ == "__main__":
    main()
