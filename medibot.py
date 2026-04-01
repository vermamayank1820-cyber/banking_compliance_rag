import os
import shutil
import streamlit as st
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load secrets: Streamlit Cloud uses st.secrets, local uses .env
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please add it to your Streamlit secrets or .env file.")
    st.stop()

DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = Path("data")

@st.cache_resource
def get_vectorstore():
    if not Path(DB_FAISS_PATH).exists():
        raise FileNotFoundError("Vector store not found. Upload PDF files and rebuild the knowledge base first.")
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def get_pdf_files():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    return sorted(DATA_PATH.glob("*.pdf"))


def rebuild_vectorstore():
    pdf_files = get_pdf_files()
    if not pdf_files:
        raise ValueError("No PDFs found in the data directory.")

    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    get_vectorstore.clear()


def save_uploaded_files(uploaded_files, replace_existing):
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    if replace_existing:
        for existing_pdf in DATA_PATH.glob("*.pdf"):
            existing_pdf.unlink()
        if Path(DB_FAISS_PATH).exists():
            shutil.rmtree(DB_FAISS_PATH)

    saved_files = []
    for uploaded_file in uploaded_files:
        destination = DATA_PATH / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_files.append(destination.name)

    rebuild_vectorstore()
    return saved_files


def answer_question(prompt):
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    custom_prompt_template = """
            You are a banking compliance assistant.
            Use only the information provided in the context to answer the user's question.
            If the answer is not in the context, say that you do not know.
            Do not make up requirements, regulations, or guidance.
            Keep the response concise and compliance-focused.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
            """

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the vector store")
            return

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.0,
                groq_api_key=groq_api_key,
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
        )

        response = qa_chain.invoke({'query': prompt})

        result = response["result"]
        source_documents = response["source_documents"]
        sources_to_show = "\n".join(
            f"- {doc.metadata.get('source', 'Unknown source')} (page {doc.metadata.get('page', 'n/a')})"
            for doc in source_documents
        )
        result_to_show = result + "\n\nSources:\n" + sources_to_show
        st.chat_message('assistant').markdown(result_to_show)
        st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

    except Exception as e:
        st.error(f"Error: {str(e)}")


def main():
    st.markdown(
        """
        <style>
        div[data-testid="stChatInput"] {display: none;}
        section.main > div.block-container {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding-bottom: 2rem;
        }
        div[data-testid="stVerticalBlock"] div[data-testid="stForm"] {
            background: #242530;
            border: 1px solid #3a3d4d;
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
        }
        div[data-testid="stVerticalBlock"] div[data-testid="stForm"] input {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        div[data-testid="stVerticalBlock"] div[data-testid="stForm"] button[kind="secondary"] {
            border-radius: 999px;
            min-height: 2.5rem;
            padding: 0 0.85rem;
        }
        div[data-testid="stVerticalBlock"] div[data-testid="stForm"] button[kind="primary"] {
            border-radius: 999px;
            min-width: 2.75rem;
            min-height: 2.75rem;
            padding: 0;
        }
        div[data-testid="stVerticalBlock"] div[data-testid="stForm"] {
            background: transparent;
            border: none;
            padding: 0;
        }
        .composer-push {
            margin-top: auto;
            min-height: 10rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Banking Compliance RAG Assistant")
    st.caption("Ask questions grounded in your banking compliance documents.")
    pdf_files = get_pdf_files()
    if pdf_files:
        st.caption("Loaded PDFs: " + ", ".join(pdf_file.name for pdf_file in pdf_files))
    else:
        st.caption("Loaded PDFs: none")

    if not pdf_files:
        st.warning("No source PDFs found in the data directory. Use ＋ Add PDFs to upload banking compliance PDFs.")
    elif any("medicine" in pdf_file.name.lower() or "medical" in pdf_file.name.lower() for pdf_file in pdf_files):
        st.warning("The current knowledge base still appears to include medical content. Upload banking compliance PDFs and choose Replace existing PDFs.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    st.markdown('<div class="composer-push"></div>', unsafe_allow_html=True)
    left_col, right_col = st.columns([1.15, 6.85], vertical_alignment="bottom")
    with left_col:
        with st.popover("＋", use_container_width=True):
            uploaded_files = st.file_uploader(
                "Upload one or more PDF files",
                type="pdf",
                accept_multiple_files=True,
                key="inline_pdf_uploader",
            )
            replace_existing = st.checkbox("Replace existing PDFs", value=False, key="inline_replace_existing")
            submitted_upload = st.button("Save and Rebuild", use_container_width=True, key="inline_save_rebuild")
    with right_col:
        with st.form("chat_composer", clear_on_submit=True):
            input_col, send_col = st.columns([6.0, 0.9], vertical_alignment="center")
            with input_col:
                prompt = st.text_input(
                    "Ask a banking compliance question",
                    placeholder="Ask a banking compliance question",
                    label_visibility="collapsed",
                )
            with send_col:
                submitted_chat = st.form_submit_button("➤", use_container_width=True)

    if submitted_upload:
        if not uploaded_files:
            st.warning("Select at least one PDF to continue.")
        else:
            with st.spinner("Saving PDFs and rebuilding the knowledge base..."):
                try:
                    saved_files = save_uploaded_files(uploaded_files, replace_existing)
                    st.session_state.messages = []
                    st.success(f"Indexed {len(saved_files)} PDF(s): {', '.join(saved_files)}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    if submitted_chat and prompt.strip():
        answer_question(prompt.strip())

if __name__ == "__main__":
    main()
