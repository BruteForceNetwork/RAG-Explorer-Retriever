import os
from os import listdir
from os.path import isfile, join
from typing import Literal, get_args

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

# define supported content providers
SourceType = Literal["Wikipedia", "Research Paper"]
AVAILABLE_SOURCES = get_args(SourceType)

# load secrets from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# initialize core components
chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
embed_model = OpenAIEmbeddings()

# path for incoming PDFs
DOCS_PATH = "./docs/"


def build_dataset(source: SourceType, query: str) -> DocArrayInMemorySearch:
    """
    Collects and prepares text data for retrieval.
    - If Wikipedia is chosen, pulls results via API.
    - If 'Research Paper' is chosen, reads the first PDF in the docs folder.
    The text is then split into overlapping chunks and indexed into a vector store.
    """
    if source not in AVAILABLE_SOURCES:
        raise ValueError(f"Unsupported source: {source}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    if source == "Wikipedia":
        wiki_runner = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        wiki_output = wiki_runner.run(query)
        chunks = [Document(page_content=line) for line in wiki_output.splitlines()]
    else:
        files = [f for f in listdir(DOCS_PATH) if isfile(join(DOCS_PATH, f))]
        if not files:
            raise FileNotFoundError("No PDF found in docs folder.")
        pdf_path = join(DOCS_PATH, files[0])
        print(f"Loading file: {pdf_path}")

        pdf_loader = PyPDFLoader(pdf_path)
        pdf_docs = pdf_loader.load()
        chunks = splitter.split_documents(pdf_docs)

    return DocArrayInMemorySearch.from_documents(documents=chunks, embedding=embed_model)


def run_retrieval(source: SourceType, dataset: DocArrayInMemorySearch, query: str) -> dict:
    """
    Executes retrieval-augmented question answering.
    Wraps the retriever with a LangChain QA chain and queries it with the prompt.
    """
    if source not in AVAILABLE_SOURCES:
        raise ValueError(f"Unsupported source: {source}")

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=dataset.as_retriever(),
        chain_type="stuff",
        verbose=True,
    )

    return qa_chain.invoke(query)


def answer_query(source: SourceType, question: str) -> dict:
    """
    High-level helper function:
    - Builds the dataset (Wikipedia or PDF)
    - Runs the retriever
    - Returns the model’s answer
    """
    if source not in AVAILABLE_SOURCES:
        raise ValueError(f"Unsupported source: {source}")

    dataset = build_dataset(source, question)
    return run_retrieval(source, dataset, question)
