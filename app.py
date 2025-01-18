import streamlit as st
import os
import pytesseract
from PIL import Image

# Set page configuration at the very top
st.set_page_config(page_title="Advanced Lesson Builder", layout="wide")

# Main function
def main():
    st.title("Welcome to the Advanced Lesson Builder!")
    st.write("Upload your files and build interactive lessons easily.")

    # Sidebar for file upload
    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file:
        # Display uploaded file
        st.image(uploaded_file, caption="Uploaded File", use_column_width=True)

        # Save file temporarily for OCR processing
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the uploaded file
        process_file(os.path.join("temp", uploaded_file.name))

# Function to process uploaded file
def process_file(file_path):
    st.subheader("Extracted Text")
    try:
        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(Image.open(file_path))
        st.text_area("Extracted Text", text, height=300)

        # Optional: Add functionality to save the extracted text
        if st.button("Save Text"):
            save_text(file_path, text)
            st.success("Text saved successfully!")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Function to save extracted text
def save_text(file_path, text):
    base_name = os.path.basename(file_path)
    output_file = os.path.splitext(base_name)[0] + "_extracted.txt"
    with open(output_file, "w") as f:
        f.write(text)

# Run the app
if __name__ == "__main__":
    # Create temp directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")
    main()
