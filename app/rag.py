from azure.search.documents.models import VectorizedQuery
import time

from .clients import openai_client, search_client
from .config import (
    EMBEDDING_DEPLOYMENT,
    CHAT_DEPLOYMENT,
    TOP_K,
    RETRIEVAL_K,
    MAX_SUBQUERIES,
    PER_QUERY_RESULTS,
    SEMANTIC_CONFIG,
    MIN_RERANKER_SCORE,
)

SYSTEM_PROMPT = """You are an enterprise knowledge assistant.
Answer ONLY from the provided knowledge-base context.
Do not use outside knowledge or make assumptions.
If the answer is not in the context, say:
"I couldn't find sufficient information in the knowledge base."
Cite factual statements using [1], [2], etc.
Keep answers concise.
"""


def create_embedding(text: str):
    response = openai_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text,
    )
    return response.data[0].embedding


def format_conversation(conversation, max_messages=6):
    if not conversation:
        return "No previous conversation."

    lines = []
    for msg in conversation[-max_messages:]:
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{msg.get('role', 'user').upper()}: {content}")

    return "\n".join(lines) or "No previous conversation."


def resolve_conversation_query(question: str, conversation=None):
    if not conversation:
        return question

    history = format_conversation(conversation)

    response = openai_client.responses.create(
        model=CHAT_DEPLOYMENT,
        input=[
            {
                "role": "system",
                "content": """Convert the current question into a standalone
search query using conversation only when needed.
Resolve pronouns and follow-up questions.
Do not answer the question.
Return ONLY the standalone search query.""",
            },
            {
                "role": "user",
                "content": f"Conversation:\n{history}\n\nQuestion:\n{question}",
            },
        ],
        max_output_tokens=100,
    )

    return response.output_text.strip() or question


def check_ambiguity(question: str, conversation=None):
    history = format_conversation(conversation or [])

    response = openai_client.responses.create(
        model=CHAT_DEPLOYMENT,
        input=[
            {
                "role": "system",
                "content": """Check whether the question is clear enough
for retrieval. Use conversation context when available.
Return exactly:
STATUS: CLEAR
or
STATUS: AMBIGUOUS
CLARIFICATION: <short question>
Do not answer the question.""",
            },
            {
                "role": "user",
                "content": f"Conversation:\n{history}\n\nQuestion:\n{question}",
            },
        ],
        max_output_tokens=80,
    )

    output = response.output_text.strip()

    if "STATUS: CLEAR" in output.upper():
        return {"status": "clear", "clarification": None}

    clarification = "Could you please clarify what you mean?"
    for line in output.splitlines():
        if line.upper().startswith("CLARIFICATION:"):
            clarification = line.split(":", 1)[1].strip()
            break

    return {"status": "ambiguous", "clarification": clarification}


def decompose_query(question: str):
    response = openai_client.responses.create(
        model=CHAT_DEPLOYMENT,
        input=[
            {
                "role": "system",
                "content": """Break the question into 2 to 4 focused
search queries. For comparisons, create queries for each entity.
Preserve important terms. Return one query per line.
Do not answer the question.""",
            },
            {"role": "user", "content": question},
        ],
        max_output_tokens=150,
    )

    queries = []
    for line in response.output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > 2 and line[0].isdigit() and line[1] in ".-)":
            line = line[2:].strip()
        queries.append(line)

    return queries[:MAX_SUBQUERIES] or [question]


def retrieve(question: str, category: str | None = None):
    vector_query = VectorizedQuery(
        vector=create_embedding(question),
        k_nearest_neighbors=RETRIEVAL_K,
        fields="contentVector",
    )

    search_filter = None
    if category:
        safe_category = category.replace("'", "''")
        search_filter = f"category eq '{safe_category}'"

    return list(
        search_client.search(
            search_text=question,
            vector_queries=[vector_query],
            query_type="semantic",
            semantic_configuration_name=SEMANTIC_CONFIG,
            filter=search_filter,
            top=PER_QUERY_RESULTS,
            select=[
                "content",
                "title",
                "category",
                "source",
                "page",
                "sheet",
                "chunk_id",
            ],
        )
    )


def merge_results(result_sets):
    merged = {}

    for results in result_sets:
        for result in results:
            key = f"{result.get('source', '')}:{result.get('chunk_id', 0)}"

            if key not in merged:
                merged[key] = result
                continue

            current = result.get("@search.reranker_score")
            existing = merged[key].get("@search.reranker_score")

            if current is not None and (
                existing is None or current > existing
            ):
                merged[key] = result

    return list(merged.values())


def select_results(result_sets, merged_results):
    selected = []

    # Keep representation from each sub-query.
    for results in result_sets:
        for result in results[:2]:
            key = (result.get("source", ""), result.get("chunk_id", 0))
            if not any(
                (x.get("source", ""), x.get("chunk_id", 0)) == key
                for x in selected
            ):
                selected.append(result)

    # Fill remaining slots.
    for result in merged_results:
        if len(selected) >= TOP_K:
            break

        key = (result.get("source", ""), result.get("chunk_id", 0))
        if not any(
            (x.get("source", ""), x.get("chunk_id", 0)) == key
            for x in selected
        ):
            selected.append(result)

    return selected[:TOP_K]


def retrieve_for_complex_question(question: str, category=None):
    queries = decompose_query(question)
    result_sets = [retrieve(q, category) for q in queries]
    merged = merge_results(result_sets)
    return select_results(result_sets, merged)


def check_retrieval_confidence(results):
    scores = [
        float(r["@search.reranker_score"])
        for r in results
        if r.get("@search.reranker_score") is not None
    ]

    if not scores:
        return False, 0.0

    best_score = max(scores)
    return best_score >= MIN_RERANKER_SCORE, best_score


def check_answerability(question, results):
    if not results:
        return False

    context = "\n\n".join(r.get("content", "") for r in results)

    response = openai_client.responses.create(
        model=CHAT_DEPLOYMENT,
        input=[
            {
                "role": "system",
                "content": """Check whether the context contains enough
information to answer the question.
Return ONLY SUFFICIENT or INSUFFICIENT.
Do not answer the question or use outside knowledge.""",
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}",
            },
        ],
        max_output_tokens=10,
    )

    return response.output_text.strip().upper().startswith("SUFFICIENT")


def answer(question: str, category=None, conversation=None):
    start_time = time.perf_counter()
    question = question.strip()
    conversation = conversation or []

    if not question:
        return {
            "status": "error",
            "answer": "Please enter a question.",
            "citations": [],
        }

    # 1. Check ambiguity.
    ambiguity = check_ambiguity(question, conversation)
    if ambiguity["status"] == "ambiguous":
        return {
            "status": "clarification_needed",
            "answer": ambiguity["clarification"],
            "citations": [],
        }

    # 2. Resolve follow-up question.
    search_question = resolve_conversation_query(
        question, conversation
    )

    # 3. Retrieve documents.
    results = retrieve_for_complex_question(
        search_question, category
    )

    if not results:
        return {
            "status": "insufficient_evidence",
            "answer": "I couldn't find sufficient information in the knowledge base.",
            "citations": [],
            "retrieval_confidence": 0.0,
        }

    # 4. Check retrieval confidence.
    retrieval_ok, best_score = check_retrieval_confidence(results)

    if not retrieval_ok:
        return {
            "status": "insufficient_evidence",
            "answer": "I couldn't find sufficient information in the knowledge base to answer this question.",
            "citations": [],
            "retrieval_confidence": round(best_score, 3),
        }

    # 5. Check whether retrieved evidence can answer the question.
    if not check_answerability(search_question, results):
        return {
            "status": "insufficient_evidence",
            "answer": "I couldn't find sufficient information in the knowledge base to answer this question.",
            "citations": [],
            "retrieval_confidence": round(best_score, 3),
        }

    # 6. Build context and citations.
    context_parts = []
    citations = []

    for i, result in enumerate(results, 1):
        page = result.get("page", 0)
        sheet = result.get("sheet", "")

        if page:
            location = f"page {page}"
        elif sheet:
            location = f"sheet {sheet}"
        else:
            location = f"chunk {result.get('chunk_id', 0)}"

        context_parts.append(
            f"[{i}]\n"
            f"Source: {result.get('source', '')}\n"
            f"Title: {result.get('title', '')}\n"
            f"Category: {result.get('category', '')}\n"
            f"Location: {location}\n"
            f"Content:\n{result.get('content', '')}"
        )

        citations.append({
            "id": i,
            "source": result.get("source", ""),
            "title": result.get("title", ""),
            "category": result.get("category", ""),
            "page": page,
            "sheet": sheet,
            "score": result.get("@search.score"),
            "reranker_score": result.get("@search.reranker_score"),
        })

    context = "\n\n".join(context_parts)

    # 7. Generate final grounded answer.
    response = openai_client.responses.create(
        model=CHAT_DEPLOYMENT,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Context:
{context}

Question:
{question}

Resolved search query:
{search_question}""",
            },
        ],
        max_output_tokens=700,
    )

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    return {
        "status": "answer",
        "answer": response.output_text,
        "citations": citations,
        "retrieval_confidence": round(best_score, 3),
        "metrics": {
            "latency_ms": round(
                (time.perf_counter() - start_time) * 1000, 2
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }