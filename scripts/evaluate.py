import json
import time
from pathlib import Path

from app.rag import answer
from app.clients import openai_client
from app.config import CHAT_DEPLOYMENT


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "evaluation"
    / "dataset.json"
)

RESULTS_DIR = (
    BASE_DIR
    / "evaluation"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# DOCUMENT MATCH
# ============================================================

def normalize_document(
    value
):

    if not value:
        return ""

    return Path(
        value
    ).name.lower().strip()


def document_hit(
    expected_documents,
    citations,
):

    expected = {
        normalize_document(
            doc
        )
        for doc in expected_documents
    }

    retrieved = {
        normalize_document(
            citation.get(
                "source",
                ""
            )
        )
        for citation in citations
    }

    if not expected:

        return True

    return bool(
        expected.intersection(
            retrieved
        )
    )


# ============================================================
# RETRIEVAL RELEVANCE
# ============================================================

def calculate_retrieval_relevance(
    expected_documents,
    citations,
):

    if not expected_documents:

        return 1.0

    expected = {
        normalize_document(
            doc
        )
        for doc in expected_documents
    }

    relevant = 0

    for citation in citations:

        source = normalize_document(
            citation.get(
                "source",
                ""
            )
        )

        if source in expected:

            relevant += 1

    if not citations:

        return 0.0

    return relevant / len(
        citations
    )


# ============================================================
# CITATION CORRECTNESS
# ============================================================

def citation_correctness(
    expected_documents,
    citations,
):

    if not expected_documents:

        return (
            1.0
            if not citations
            else 0.0
        )

    expected = {
        normalize_document(
            doc
        )
        for doc in expected_documents
    }

    if not citations:

        return 0.0

    correct = 0

    for citation in citations:

        source = normalize_document(
            citation.get(
                "source",
                ""
            )
        )

        if source in expected:

            correct += 1

    return correct / len(
        citations
    )


# ============================================================
# LLM JUDGE
# ============================================================

def judge_generation(
    question,
    expected_answer,
    actual_answer,
    citations,
    expected_documents,
):
    """
    Evaluate:

    - answer correctness
    - groundedness
    - hallucination
    """

    citation_text = "\n".join(

        f"[{c.get('id')}] "
        f"{c.get('source')} "
        f"{c.get('page') or c.get('sheet') or ''}"

        for c in citations
    )

    expected_docs = ", ".join(
        expected_documents
    )

    prompt = f"""
Evaluate the RAG response.

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

ACTUAL ANSWER:
{actual_answer}

EXPECTED DOCUMENTS:
{expected_docs}

CITATIONS:
{citation_text}

Return ONLY valid JSON:

{{
  "correctness": 0,
  "groundedness": 0,
  "hallucination": 0,
  "reason": ""
}}

Scoring:

correctness:
0 = incorrect
1 = partially correct
2 = fully correct

groundedness:
0 = unsupported
1 = partially grounded
2 = fully grounded

hallucination:
0 = no hallucination
1 = minor unsupported claim
2 = major hallucination

Important:
- Judge only against the expected answer
  and supplied evidence.
- Do not use outside knowledge.
"""

    response = openai_client.responses.create(

        model=CHAT_DEPLOYMENT,

        input=[
            {
                "role": "system",
                "content": (
                    "You are a strict RAG evaluator."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        max_output_tokens=200,
    )

    text = (
        response.output_text
        .strip()
    )

    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:

        return {
            "correctness": 0,
            "groundedness": 0,
            "hallucination": 2,
            "reason": (
                "Judge returned invalid JSON"
            ),
        }


# ============================================================
# SINGLE TEST
# ============================================================

def evaluate_case(case):

    question = case[
        "question"
    ]

    conversation = case.get(
        "conversation",
        []
    )

    expected_answer = case[
        "expected_answer"
    ]

    expected_documents = case.get(
        "expected_documents",
        []
    )

    start = time.perf_counter()

    result = answer(
        question=question,
        conversation=conversation,
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000

    actual_answer = result.get(
        "answer",
        ""
    )

    citations = result.get(
        "citations",
        []
    )

    # --------------------------------------------
    # Retrieval metrics
    # --------------------------------------------

    hit = document_hit(
        expected_documents,
        citations,
    )

    relevance = (
        calculate_retrieval_relevance(
            expected_documents,
            citations,
        )
    )

    citation_score = (
        citation_correctness(
            expected_documents,
            citations,
        )
    )

    # --------------------------------------------
    # Special cases
    # --------------------------------------------

    if case["type"] == "ambiguous":

        correctness = (
            2
            if result.get("status")
            == "clarification_required"
            else 0
        )

        groundedness = 2

        hallucination = (
            0
            if result.get("status")
            == "clarification_required"
            else 2
        )

        judge = {
            "correctness": correctness,
            "groundedness": groundedness,
            "hallucination": hallucination,
            "reason": (
                "Ambiguity handling evaluated "
                "by system status."
            ),
        }

    elif case["type"] == "no_answer":

        insufficient = (
            result.get("status")
            == "insufficient_evidence"
        )

        judge = {
            "correctness": (
                2 if insufficient else 0
            ),
            "groundedness": (
                2 if insufficient else 0
            ),
            "hallucination": (
                0 if insufficient else 2
            ),
            "reason": (
                "Missing-information "
                "handling."
            ),
        }

    else:

        judge = judge_generation(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            citations=citations,
            expected_documents=expected_documents,
        )

    metrics = result.get(
        "metrics",
        {}
    )

    return {
        "id": case["id"],
        "question": question,
        "type": case["type"],
        "difficulty": case["difficulty"],

        "expected_answer": expected_answer,

        "actual_answer": actual_answer,

        "status": result.get(
            "status",
            ""
        ),

        "retrieval_hit": hit,

        "retrieval_relevance": round(
            relevance,
            3,
        ),

        "citation_correctness": round(
            citation_score,
            3,
        ),

        "answer_correctness": judge[
            "correctness"
        ],

        "groundedness": judge[
            "groundedness"
        ],

        "hallucination": judge[
            "hallucination"
        ],

        "judge_reason": judge[
            "reason"
        ],

        "retrieval_confidence": result.get(
            "retrieval_confidence"
        ),

        "latency_ms": metrics.get(
            "latency_ms",
            round(
                elapsed_ms,
                2,
            ),
        ),

        "input_tokens": metrics.get(
            "input_tokens",
            0,
        ),

        "output_tokens": metrics.get(
            "output_tokens",
            0,
        ),

        "total_tokens": metrics.get(
            "total_tokens",
            0,
        ),

        "citations": citations,
    }


# ============================================================
# AGGREGATE METRICS
# ============================================================

def calculate_summary(
    results
):

    total = len(
        results
    )

    if not total:

        return {}

    hit_rate = sum(
        r["retrieval_hit"]
        for r in results
    ) / total

    relevance = sum(
        r["retrieval_relevance"]
        for r in results
    ) / total

    citation = sum(
        r["citation_correctness"]
        for r in results
    ) / total

    correctness = sum(
        r["answer_correctness"]
        for r in results
    ) / (
        total * 2
    )

    groundedness = sum(
        r["groundedness"]
        for r in results
    ) / (
        total * 2
    )

    hallucination_rate = sum(
        r["hallucination"] > 0
        for r in results
    ) / total

    avg_latency = sum(
        r["latency_ms"]
        for r in results
    ) / total

    total_tokens = sum(
        r["total_tokens"]
        for r in results
    )

    avg_tokens = (
        total_tokens / total
    )

    return {
        "total_questions": total,

        "retrieval_hit_rate": round(
            hit_rate,
            3,
        ),

        "retrieval_relevance": round(
            relevance,
            3,
        ),

        "answer_correctness": round(
            correctness,
            3,
        ),

        "groundedness": round(
            groundedness,
            3,
        ),

        "citation_correctness": round(
            citation,
            3,
        ),

        "hallucination_rate": round(
            hallucination_rate,
            3,
        ),

        "average_latency_ms": round(
            avg_latency,
            2,
        ),

        "average_tokens": round(
            avg_tokens,
            2,
        ),

        "total_tokens": total_tokens,
    }


# ============================================================
# RUN
# ============================================================

def main():

    dataset = load_dataset()

    results = []

    print(
        "\n========================================"
    )

    print(
        "RAG EVALUATION"
    )

    print(
        "========================================\n"
    )

    for index, case in enumerate(
        dataset,
        start=1,
    ):

        print(
            f"[{index}/{len(dataset)}] "
            f"{case['id']}: "
            f"{case['question']}"
        )

        result = evaluate_case(
            case
        )

        results.append(
            result
        )

        print(
            f"  Hit: {result['retrieval_hit']}"
        )

        print(
            f"  Relevance: "
            f"{result['retrieval_relevance']}"
        )

        print(
            f"  Correctness: "
            f"{result['answer_correctness']}"
        )

        print(
            f"  Groundedness: "
            f"{result['groundedness']}"
        )

        print(
            f"  Hallucination: "
            f"{result['hallucination']}"
        )

        print(
            f"  Latency: "
            f"{result['latency_ms']} ms\n"
        )

    summary = calculate_summary(
        results
    )

    output = {
        "summary": summary,
        "results": results,
    }

    output_file = (
        RESULTS_DIR
        / "improved_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "\n========================================"
    )

    print(
        "SUMMARY"
    )

    print(
        "========================================"
    )

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print(
        f"\nSaved to: {output_file}"
    )


if __name__ == "__main__":

    main()