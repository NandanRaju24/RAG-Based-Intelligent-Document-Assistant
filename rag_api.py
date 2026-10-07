from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from rag_core import answer_question


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RAG Document Assistant API",
    description="API for asking questions about uploaded documents",
    version="1.1"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question to ask about the uploaded documents"
    )


# ============================================================
# SOURCE MODEL
# ============================================================

class SourceResponse(BaseModel):

    filename: str
    page: int
    chunk: int
    distance: float


# ============================================================
# RESPONSE MODEL
# ============================================================

class AnswerResponse(BaseModel):

    question: str
    answer: str
    sources: List[SourceResponse]

    embedding_time: float
    retrieval_time: float
    llm_time: float
    total_time: float


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "RAG Document Assistant API is running",
        "version": "1.1"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask_question(request: QuestionRequest):

    try:

        # Remove unnecessary spaces
        question = request.question.strip()

        # Reject whitespace-only questions
        if not question:

            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        # Reject extremely short questions
        if len(question) < 3:

            raise HTTPException(
                status_code=400,
                detail="Question is too short. Please enter a meaningful question."
            )

        # Run shared RAG pipeline
        result = answer_question(question)

        return result

    except HTTPException:

        raise

    except Exception as error:

        print(f"RAG API ERROR: {error}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the question."
        )