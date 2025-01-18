# Advanced Lesson Builder

**Advanced Lesson Builder** is an AI-powered Streamlit application designed to streamline and enhance the instructional design process for creating digital learning content.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Features

- **Automated Lesson Design:** Extract, analyze, and structure content automatically.
- **Content Integration:** Upload and parse textbook content, SME inputs, and more.
- **Alignment with Objectives:** AI-based analysis to ensure content aligns with lesson objectives.
- **Structured Output Generation:** Create lesson screens with titles, text, durations, and interactive elements.
- **Interactive Elements:** Add quizzes, reflections, and multimedia content.
- **Export Functionality:** Export lessons in CSV or JSON formats for LMS integration.

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- [Streamlit](https://streamlit.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

### System-Level Dependencies

For Debian/Ubuntu systems:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils
