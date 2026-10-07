# 📚 RAG-Based Intelligent Document Assistant

A Retrieval-Augmented Generation (RAG) system that allows users
to upload PDF documents and ask questions about their contents.

The system extracts text from PDF documents, divides the text into
overlapping chunks, converts the chunks into semantic embeddings,
stores them in ChromaDB, retrieves relevant chunks for a user
question, and uses Llama 3.2 through Ollama to generate a
document-grounded answer.

---

## 🚀 Features

- PDF document upload
- Multi-PDF processing
- PDF text extraction using PyMuPDF
- Text cleaning
- Overlapping text chunking
- Semantic embeddings using Sentence Transformers
- ChromaDB vector database
- Top-K semantic retrieval
- Context construction for LLM generation
- Local Llama 3.2 inference through Ollama
- Source and page information
- Retrieval Debug Mode
- Retrieval and generation performance monitoring
- Streamlit user interface
- FastAPI REST API
- Database reset functionality
- RAG evaluation and diagnostics

---

## 🏗️ Architecture

```text
                 PDF Documents
                       │
                       ▼
               PyMuPDF Extraction
                       │
                       ▼
                 Text Cleaning
                       │
                       ▼
              Chunking + Overlap
                       │
                       ▼
              Sentence Transformers
                       │
                       ▼
                  Embeddings
                       │
                       ▼
                   ChromaDB
                       │
                       │
                       │
              User Question
                       │
                       ▼
             Question Embedding
                       │
                       ▼
              ChromaDB Retrieval
                       │
                       ▼
              Top-K Relevant Chunks
                       │
                       ▼
               Context Construction
                       │
                       ▼
             Grounded Prompt
                       │
                       ▼
            Llama 3.2 via Ollama
                       │
                       ▼
              Answer + Sources
```

---

## 🧠 How RAG Works in This Project

The system separates information retrieval from answer generation.

### 1. Document Processing

The user uploads one or more PDF documents.

PyMuPDF extracts the text from each page.

### 2. Text Cleaning

Extracted text is cleaned to remove unnecessary whitespace and
prepare it for processing.

### 3. Chunking

Large document text is divided into smaller chunks.

The current configuration uses:

- Chunk Size: 500
- Chunk Overlap: 100

The overlap helps preserve context between neighboring chunks.

### 4. Embedding Generation

Each document chunk is converted into a numerical vector using:

`all-MiniLM-L6-v2`

These vectors represent the semantic meaning of the text.

### 5. Vector Storage

The embeddings and their document metadata are stored in ChromaDB.

Metadata includes information such as:

- Filename
- Page number
- Chunk number

### 6. Question Retrieval

When a user asks a question, the question is converted into an
embedding.

ChromaDB searches for the most semantically relevant document chunks.

The current Top-K value is:

`5`

### 7. Context Construction

The retrieved chunks are combined into a document context containing
the source filename, page number, chunk number, and text.

### 8. Answer Generation

The retrieved context and user question are passed to Llama 3.2
through Ollama.

The model generates an answer based on the retrieved document
context.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Semantic embeddings |
| ChromaDB | Vector database and retrieval |
| Ollama | Local LLM runtime |
| Llama 3.2 | Local language model |
| Streamlit | Interactive web interface |
| FastAPI | REST API |
| Uvicorn | FastAPI server |
| Requests | Communication with Ollama |
| Pydantic | API request validation |

---

## 📁 Project Structure

```text
RAG-Based-Intelligent-Document-Assistant/
│
├── day16_rag_streamlit_app.py
├── rag_core.py
├── rag_api.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── chroma_db_day16/        # Local vector database
├── documents/              # Local uploaded documents
└── .venv/                  # Local virtual environment
```

> `chroma_db_day16/`, `documents/`, and `.venv/` should not be
> committed to GitHub.

---

## ⚙️ Configuration

The current RAG configuration is:

```text
Embedding Model: all-MiniLM-L6-v2
LLM: llama3.2:3b
Chunk Size: 500
Chunk Overlap: 100
Top K: 5
```

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG-Based-Intelligent-Document-Assistant
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.venv\Scripts\activate
```

### 4. Install Python dependencies

```powershell
pip install -r requirements.txt
```

---

## 🦙 Install and Run Ollama

Install Ollama on your computer and download the required model:

```powershell
ollama pull llama3.2:3b
```

Make sure Ollama is running before asking questions.

The application communicates with the local Ollama API.

---

## ▶️ Run the Streamlit Application

Run:

```powershell
python -m streamlit run .\day16_rag_streamlit_app.py --server.port 8502
```

The Streamlit interface allows you to:

1. Upload PDF documents
2. Process documents
3. Store document embeddings
4. Ask questions
5. Inspect retrieved chunks
6. View source information
7. Monitor retrieval and generation performance

---

## 🔌 Run the FastAPI Application

Start the API using:

```powershell
uvicorn rag_api:app --reload
```

The API provides programmatic access to the RAG system.

FastAPI's interactive documentation can be accessed through:

```text
/docs
```

---

## 🔍 Debug Mode

The application provides a Debug Mode for inspecting the retrieval
process.

Debug information includes:

- Retrieved document chunks
- Filename
- Page number
- Chunk number
- ChromaDB distance
- Embedding time
- Retrieval time
- LLM generation time
- Total processing time

This makes it possible to determine whether a problem occurs during
retrieval or answer generation.

---

## 🧪 RAG Evaluation & Diagnostics

The application evaluates the RAG pipeline using four main criteria:

### 1. Retrieval Relevance

Are the retrieved chunks actually related to the question?

### 2. Answer Relevance

Does the generated answer directly answer the question?

### 3. Faithfulness / Groundedness

Can the generated answer be supported by the retrieved document
context?

### 4. Source Correctness

Can the information in the answer be located in the retrieved
document source?

---

## 📊 Performance Monitoring

The application records:

- Embedding time
- ChromaDB retrieval time
- LLM generation time
- Total processing time

Performance can vary depending on hardware, model loading state,
document size, and local inference speed.

---

## 🔐 Privacy

The RAG pipeline is designed to support local document processing
and local LLM inference through Ollama.

Uploaded documents and the local ChromaDB database should not be
committed to the public GitHub repository.

---

## 🎯 Project Goals

The project was developed to understand and implement the main
components of a Retrieval-Augmented Generation system, including:

- Document processing
- Text chunking
- Embeddings
- Vector databases
- Semantic retrieval
- Context construction
- Grounded LLM generation
- RAG evaluation
- API development
- Application debugging

---

## 👨‍💻 Author

**Nandan Raj**

AI/ML Engineering Student

---

## 📌 Future Improvements

Possible future improvements include:

- Improved retrieval ranking
- Hybrid keyword + semantic retrieval
- Reranking models
- Better evaluation datasets
- Multiple LLM provider support
- Authentication for the API
- Cloud deployment
- Conversation history
- More advanced document formats