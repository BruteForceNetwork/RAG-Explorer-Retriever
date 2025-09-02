import os
import shutil
import pathlib
import streamlit as st
from Rag_explorer_retriever.RAGexplorer import answer_query

# Streamlit UI Dashboard 
st.title("RAG Explorer")
st.subheader("Benchmark retrieval from Wikipedia vs. uploaded PDFs")

# keep track of user's choice
if "data_source" not in st.session_state:
    st.session_state.data_source = ""

# reset docs folder every session
DOCS_DIR = pathlib.Path().absolute() / "docs"
if DOCS_DIR.exists():
    shutil.rmtree(DOCS_DIR)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# selection widget
temp_box = st.empty()
selected_option = temp_box.selectbox(
    key="source_select",
    label="Choose a knowledge source:",
    options=("Wikipedia", "Research Paper"),
)

if st.button("Confirm Source"):
    st.session_state.data_source = selected_option


# ----------------------#
#   Input Handlers      #
# ----------------------#
def handle_wiki_input() -> str | None:
    """Collects a Wikipedia query string from the user."""
    query = st.text_input(
        label="Enter your search topic:",
        max_chars=200,
        key="wiki_query",
    )
    if st.button("Search Wikipedia"):
        return query


def handle_pdf_input() -> str | None:
    """Allows uploading a PDF and a query against it."""
    with st.form(key="pdf_form", clear_on_submit=False):
        file_upload = st.file_uploader(
            label="Upload a PDF document",
            type=["pdf"],
        )
        query = st.text_input(
            label="What do you want to ask about this file?",
            max_chars=200,
            key="pdf_query",
        )
        submit_btn = st.form_submit_button("Process Document")

    if submit_btn and file_upload:
        file_path = DOCS_DIR / file_upload.name
        with open(file_path, "wb") as f:
            f.write(file_upload.getbuffer())
        return query


# ----------------------#
#   Main Application    #
# ----------------------#
def run_app(selection: str):
    """Runs the RAG workflow based on the chosen source."""
    if selection == "Wikipedia":
        query = handle_wiki_input()
        if query:
            with st.spinner("Querying Wikipedia..."):
                result = answer_query(selection, query)
            st.success("Done!")
            st.write(result["result"])

    elif selection == "Research Paper":
        query = handle_pdf_input()
        if query:
            with st.spinner("Analyzing your document..."):
                result = answer_query(selection, query)
            st.success("Done!")
            st.write(result["result"])


# entry point
if __name__ == "__main__":
    run_app(st.session_state.data_source)
