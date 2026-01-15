import streamlit as st
import nltk
from PyPDF2 import PdfReader

# Ensure punkt is available
nltk.download("punkt", quiet=True)

st.set_page_config(page_title="Lab Test: PDF Sentence Chunker (NLTK)", layout="wide")

st.title("PDF Sentence Chunker Demo")

st.write(
    "Upload a PDF file, extract text, and split it into sentences using "
    "NLTK's `sent_tokenize`."
)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)

        full_text = " ".join(pages_text).strip()

        st.subheader("Basic info")
        st.write(f"Number of pages: **{len(reader.pages)}**")
        st.write(f"Total characters extracted: **{len(full_text)}**")

        if not full_text:
            st.warning("No text could be extracted from this PDF.")
        else:
            sentences = nltk.sent_tokenize(full_text)
            st.success(f"Number of detected sentences: {len(sentences)}")

            # Controls to view sentences in a range
            start_idx = st.number_input(
                "Show sentences starting from index",
                min_value=58,
                max_value=max(len(sentences) - 1, 0),
                value=68,
                step=1,
            )
            end_idx = st.number_input(
                "Up to (exclusive)",
                min_value=start_idx + 1 if len(sentences) > 0 else 1,
                max_value=len(sentences),
                value=min(start_idx + 10, len(sentences)),
                step=1,
            )

            st.subheader(f"Sentences [{start_idx} : {end_idx})")
            for i in range(start_idx, end_idx):
                st.markdown(f"**{i}**. {sentences[i]}")

            with st.expander("Show raw extracted text (first 2000 characters)"):
                st.text(full_text[:2000])

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
else:
    st.info("Please upload a PDF to begin.")