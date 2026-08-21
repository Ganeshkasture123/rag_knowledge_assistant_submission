from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag import answer


app = FastAPI(
    title="Enterprise RAG Knowledge Assistant"
)


class ConversationMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    category: str | None = None
    conversation: list[ConversationMessage] = Field(
        default_factory=list
    )


@app.get("/")
def home():
    return {
        "message": "Enterprise RAG Knowledge Assistant is running",
        "status": "healthy",
        "endpoint": "POST /ask",
        "docs": "/docs",
    }


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        conversation = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in req.conversation
        ]

        return answer(
            question=question,
            category=req.category,
            conversation=conversation,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline failed: {exc}",
        )