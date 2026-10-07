# DAY 19 - CLEAN STREAMLIT APP
# Save this as: day16_rag_streamlit_app.py
#
# IMPORTANT:
# This Streamlit file uses the reusable RAG logic from rag_core.py.
# Make sure rag_core.py returns "retrieved_chunks" in answer_question().

import re
import hashlib
import time

import streamlit as st

from rag_core import (
    answer_question,
    load_embedding_model,
    load_chroma_client,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)

# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = "uploaded_documents"


# ============================================================
# DATABASE
# ============================================================

def get_collection():
    client = load_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def reset_collection():
    """Delete all stored chunks and recreate an empty collection."""

    client = load_chroma_client()

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# CHUNK CREATION
# ============================================================

def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    chunks = []

    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += step

    return chunks


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_chunks(pdf_file):
    import fitz

    pdf_document = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf",
    )

    all_chunks = []

    try:
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            text = clean_text(page.get_text())

            if not text:
                continue

            chunks = create_chunks(text)

            for chunk_index, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "filename": pdf_file.name,
                            "page": page_number + 1,
                            "chunk": chunk_index,
                        },
                    }
                )
    finally:
        pdf_document.close()

    return all_chunks


# ============================================================
# STABLE CHUNK ID
# ============================================================

def create_chunk_id(chunk):
    content = (
        chunk["metadata"]["filename"]
        + "_"
        + str(chunk["metadata"]["page"])
        + "_"
        + str(chunk["metadata"]["chunk"])
        + "_"
        + chunk["text"]
    )

    return hashlib.md5(
        content.encode("utf-8")
    ).hexdigest()


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_documents(uploaded_files):
    collection = reset_collection()
    embedding_model = load_embedding_model()

    all_chunks = []

    for pdf_file in uploaded_files:
        chunks = extract_pdf_chunks(pdf_file)
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0, 0

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedding_model.encode(texts).tolist()

    ids = [
        create_chunk_id(chunk)
        for chunk in all_chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in all_chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    embedding_dimension = len(embeddings[0])

    return len(all_chunks), embedding_dimension


# ============================================================
# HEADER
# ============================================================

st.title(
    "📚 RAG-Based Intelligent Document Assistant"
)

st.write(
    "Upload PDF documents, process them, and ask questions "
    "using Retrieval-Augmented Generation."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    st.write(
        f"**Embedding Model:** all-MiniLM-L6-v2"
    )
    st.write(
        f"**LLM:** llama3.2:3b"
    )
    st.write(
        f"**Chunk Size:** {CHUNK_SIZE}"
    )
    st.write(
        f"**Chunk Overlap:** {CHUNK_OVERLAP}"
    )
    st.write(
        f"**Top K:** {TOP_K}"
    )

    st.divider()

    debug_mode = st.checkbox(
        "🔧 Debug Mode",
        value=False,
        help="Show retrieved chunks and detailed performance information.",
    )


# ============================================================
# UPLOAD DOCUMENTS
# ============================================================

st.header("📄 1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDF documents.", 
)

if uploaded_files:
    st.write(
        f"📄 **{len(uploaded_files)} PDF(s) uploaded**"
    )

    for file in uploaded_files:
        st.write(f"• {file.name}")

    if st.button(
        "🚀 Process Documents",
        use_container_width=True,
    ):
        with st.spinner(
            "Extracting, chunking and embedding documents..."
        ):
            try:
                total_chunks, embedding_dimension = process_documents(
                    uploaded_files
                )

                if total_chunks == 0:
                    st.warning(
                        "No text could be extracted from the uploaded PDF files."
                    )
                else:
                    st.success(
                        "Documents processed successfully!"
                    )
                    st.info(
                        f"Stored chunks: **{total_chunks}**"
                    )
                    st.info(
                        f"Embedding dimension: **{embedding_dimension}**"
                    )
                    st.session_state["documents_processed"] = True
                    st.rerun()

            except Exception as error:
                st.error(
                    "Error while processing documents."
                )
                if debug_mode:
                    st.exception(error)


# ============================================================
# DATABASE STATUS
# ============================================================

st.header("🗄️ 2. Database Status")

try:
    collection = get_collection()
    count = collection.count()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Stored Chunks", count)

    with col2:
        st.metric("Top K", TOP_K)

    st.subheader("🗄️ Database Management")

    if count > 0:
        st.warning(
            f"Database currently contains {count} chunks."
        )
    else:
        st.info(
            "Database is empty. Upload and process PDFs to add documents."
        )

    if st.button(
        "🗑️ Reset Database",
        use_container_width=True,
    ):
        try:
            reset_collection()
            st.session_state["documents_processed"] = False
            st.success(
                "Database reset successfully. All stored document chunks were deleted."
            )
            st.rerun()
        except Exception as error:
            st.error("Failed to reset database.")
            if debug_mode:
                st.exception(error)

except Exception as error:
    st.warning("Database is not initialized yet.")
    if debug_mode:
        st.exception(error)


# ============================================================
# ASK QUESTION
# ============================================================

st.header("💬 3. Ask a Question")

question = st.text_input(
    "Enter your question about the uploaded documents:",
    max_chars=1000,
)

if st.button(
    "🔍 Ask Question",
    use_container_width=True,
):
    question = question.strip()

    if not question:
        st.warning("Please enter a question.")

    elif len(question) < 3:
        st.warning(
            "Question is too short. Please enter a meaningful question."
        )

    else:
        try:
            collection = get_collection()

            if collection.count() == 0:
                st.warning(
                    "The database is empty. Please upload and process PDF documents first."
                )
                st.stop()

            total_start = time.time()

            with st.spinner(
                "Retrieving documents and generating answer..."
            ):
                result = answer_question(question)

            total_time = time.time() - total_start

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            retrieved_chunks = result.get("retrieved_chunks", [])

            embedding_time = result.get("embedding_time", 0.0)
            retrieval_time = result.get("retrieval_time", 0.0)
            llm_time = result.get("llm_time", 0.0)

            # ------------------------------------------------
            # RETRIEVAL PERFORMANCE
            # ------------------------------------------------

            st.subheader("⏱️ Retrieval Performance")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Embedding Time",
                    f"{embedding_time:.2f} s",
                )

            with col2:
                st.metric(
                    "ChromaDB Retrieval Time",
                    f"{retrieval_time:.2f} s",
                )

            # ------------------------------------------------
            # DEBUG RETRIEVED CHUNKS
            # ------------------------------------------------

            if debug_mode and retrieved_chunks:
                st.subheader(
                    "🔎 Retrieved Context — Debug View"
                )

                st.info(
                    "Inspect these chunks before evaluating the generated answer."
                )

                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1,
                ):
                    metadata = chunk.get("metadata", {})
                    distance = chunk.get("distance", 0.0)

                    with st.expander(
                        f"Chunk {i} — "
                        f"{metadata.get('filename', 'Unknown')} | "
                        f"Page {metadata.get('page', 'Unknown')} | "
                        f"Distance {distance:.4f}"
                    ):
                        st.markdown(f"### Chunk {i}")
                        st.write("**Retrieved Text:**")
                        st.write(chunk.get("text", ""))
                        st.write(f"**Distance:** `{distance:.4f}`")
                        st.write(
                            "**Filename:** "
                            f"`{metadata.get('filename', 'Unknown')}`"
                        )
                        st.write(
                            "**Page:** "
                            f"`{metadata.get('page', 'Unknown')}`"
                        )
                        st.write(
                            "**Chunk:** "
                            f"`{metadata.get('chunk', 'Unknown')}`"
                        )

            elif debug_mode:
                st.warning(
                    "No retrieved chunk details were returned by rag_core."
                )

            # ------------------------------------------------
            # GENERATED ANSWER
            # ------------------------------------------------

            st.subheader("🧠 Generated Answer")

            if answer:
                st.write(answer)
            else:
                st.warning(
                    "The system did not generate an answer."
                )

            # ------------------------------------------------
            # PERFORMANCE BREAKDOWN
            # ------------------------------------------------

            if debug_mode:
                st.subheader("⏱️ Performance Breakdown")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Embedding",
                        f"{embedding_time:.2f}s",
                    )

                with col2:
                    st.metric(
                        "Retrieval",
                        f"{retrieval_time:.2f}s",
                    )

                with col3:
                    st.metric(
                        "LLM",
                        f"{llm_time:.2f}s",
                    )

                with col4:
                    st.metric(
                        "Total",
                        f"{total_time:.2f}s",
                    )

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            if sources:
                st.subheader("📚 Retrieved Sources")

                displayed_sources = set()

                for index, source in enumerate(
                    sources,
                    start=1,
                ):
                    filename = source.get(
                        "filename", "Unknown"
                    )
                    page = source.get(
                        "page", "Unknown"
                    )
                    

                    source_key = (filename, page)

                    if source_key in displayed_sources:
                        continue

                    displayed_sources.add(source_key)

                    st.write(
                        f"{index}. **{filename}** — "
                        f"Page **{page}** "
                        
                    )

            # ------------------------------------------------
            # RAG Evaluation & Diagnostics EVALUATION
            # ------------------------------------------------
      
            if debug_mode:
                st.subheader("🧪 RAG Evaluation & Diagnostics")

                st.write("1️⃣ **Retrieval Relevance**")
                st.write(
                    "Are the retrieved chunks actually about the question?"
                )

                st.write("2️⃣ **Answer Relevance**")
                st.write(
                    "Does the generated answer directly answer the question?"
                )

                st.write("3️⃣ **Faithfulness / Groundedness**")
                st.write(
                    "Can the answer be supported by the retrieved chunks above?"
                )

                st.write("4️⃣ **Source Correctness**")
                st.write(
                    "Can you locate the information in the displayed source?"
                )

        except Exception as error:
            st.error(
                "An error occurred while processing your question."
            )

            if debug_mode:
                st.exception(error)
            else:
                st.info(
                    "Turn on Debug Mode to see detailed error information."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RAG-Based Intelligent Document Assistant | "
    "Streamlit + ChromaDB + Sentence Transformers + Ollama"
)    
#& ".venv\Scripts\python.exe" -m streamlit run ".\day16_rag_streamlit_app.py" --server.port 8502