import json
import time
from pathlib import Path

from app.clients import openai_client
from app.config import CHAT_DEPLOYMENT

from scripts.baseline_rag import baseline_answer


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
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# DOCUMENT NORMALIZATION
# ============================================================

def normalize_document(value):

    if not value:
        return ""

    return Path(
        value
    ).name.lower().strip()


# ============================================================
# DOCUMENT HIT
# ============================================================

def document_hit(
    expected_documents,
    citations
):

    if not expected_documents:

        return True

    expected = {
        normalize_document(doc)
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

    return bool(
        expected.intersection(
            retrieved
        )
    )


# ============================================================
# RETRIEVAL RELEVANCE
# ============================================================

def retrieval_relevance(
    expected_documents,
    citations
):

    if not expected_documents:

        return 1.0

    if not citations:

        return 0.0

    expected = {
        normalize_document(doc)
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

    return (
        relevant / len(citations)
    )


# ============================================================
# CITATION CORRECTNESS
# ============================================================

def citation_correctness(
    expected_documents,
    citations
):

    if not expected_documents:

        return (
            1.0
            if not citations
            else 0.0
        )

    if not citations:

        return 0.0

    expected = {
        normalize_document(doc)
        for doc in expected_documents
    }

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

    return (
        correct / len(citations)
    )


# ============================================================
# SIMPLE BASELINE JUDGE
# ============================================================

def judge_baseline(
    question,
    expected_answer,
    actual_answer,
    expected_documents,
    citations
):

    citation_text = "\n".join(

        f"[{c.get('id')}] "
        f"{c.get('source')} "
        f"{c.get('page', '')}"

        for c in citations
    )

    prompt = f"""
Evaluate this baseline RAG answer.

Question:
{question}

Expected answer:
{expected_answer}

Actual answer:
{actual_answer}

Expected documents:
{expected_documents}

Retrieved citations:
{citation_text}

Return ONLY JSON:

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

Do not use outside knowledge.
"""

    response = openai_client.responses.create(

        model=CHAT_DEPLOYMENT,

        input=[
            {
                "role": "system",
                "content": (
                    "You are a strict RAG evaluator."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_output_tokens=200
    )

    text = (
        response.output_text
        .strip()
    )

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        return {
            "correctness": 0,
            "groundedness": 0,
            "hallucination": 2,
            "reason": (
                "Invalid judge response"
            )
        }


# ============================================================
# EVALUATE ONE CASE
# ============================================================

def evaluate_case(case):

    question = case[
        "question"
    ]

    expected_answer = case[
        "expected_answer"
    ]

    expected_documents = case.get(
        "expected_documents",
        []
    )

    start = time.perf_counter()

    result = baseline_answer(
        question
    )

    latency_ms = (
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
        citations
    )

    relevance = retrieval_relevance(
        expected_documents,
        citations
    )

    citation_score = (
        citation_correctness(
            expected_documents,
            citations
        )
    )

    # --------------------------------------------
    # Ambiguous questions
    # --------------------------------------------

    if case["type"] == "ambiguous":

        is_clarification = (
            "clarif"
            in actual_answer.lower()
            or
            "which"
            in actual_answer.lower()
        )

        judge = {
            "correctness": (
                2 if is_clarification else 0
            ),
            "groundedness": 2,
            "hallucination": (
                0 if is_clarification else 2
            ),
            "reason": (
                "Baseline ambiguity handling"
            )
        }

    # --------------------------------------------
    # No-answer questions
    # --------------------------------------------

    elif case["type"] == "no_answer":

        unknown_words = [
            "don't know",
            "do not know",
            "not found",
            "insufficient",
            "cannot find",
            "not available"
        ]

        refused = any(
            word in actual_answer.lower()
            for word in unknown_words
        )

        judge = {
            "correctness": (
                2 if refused else 0
            ),
            "groundedness": (
                2 if refused else 0
            ),
            "hallucination": (
                0 if refused else 2
            ),
            "reason": (
                "Baseline missing-information "
                "handling"
            )
        }

    # --------------------------------------------
    # Normal questions
    # --------------------------------------------

    else:

        judge = judge_baseline(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            expected_documents=expected_documents,
            citations=citations
        )

    return {
        "id": case["id"],
        "question": question,
        "type": case["type"],
        "difficulty": case["difficulty"],

        "expected_answer": expected_answer,

        "actual_answer": actual_answer,

        "retrieval_hit": hit,

        "retrieval_relevance": round(
            relevance,
            3
        ),

        "citation_correctness": round(
            citation_score,
            3
        ),

        "answer_correctness": (
            judge["correctness"]
        ),

        "groundedness": (
            judge["groundedness"]
        ),

        "hallucination": (
            judge["hallucination"]
        ),

        "judge_reason": (
            judge["reason"]
        ),

        "latency_ms": round(
            latency_ms,
            2
        ),

        "total_tokens": 0,

        "citations": citations
    }


# ============================================================
# SUMMARY
# ============================================================

def calculate_summary(results):

    total = len(results)

    if total == 0:

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

    average_latency = sum(
        r["latency_ms"]
        for r in results
    ) / total

    return {
        "total_questions": total,

        "retrieval_hit_rate": round(
            hit_rate,
            3
        ),

        "retrieval_relevance": round(
            relevance,
            3
        ),

        "answer_correctness": round(
            correctness,
            3
        ),

        "groundedness": round(
            groundedness,
            3
        ),

        "citation_correctness": round(
            citation,
            3
        ),

        "hallucination_rate": round(
            hallucination_rate,
            3
        ),

        "average_latency_ms": round(
            average_latency,
            2
        ),

        "average_tokens": 0
    }


# ============================================================
# MAIN
# ============================================================

def main():

    dataset = load_dataset()

    results = []

    print()
    print("=" * 60)
    print("BASELINE RAG EVALUATION")
    print("=" * 60)

    for index, case in enumerate(
        dataset,
        start=1
    ):

        print(
            f"\n[{index}/{len(dataset)}] "
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
            f"  Hit Rate: "
            f"{result['retrieval_hit']}"
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
            f"{result['latency_ms']} ms"
        )

    summary = calculate_summary(
        results
    )

    output = {
        "summary": summary,
        "results": results
    }

    output_file = (
        RESULTS_DIR
        / "baseline_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print()
    print("=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print()
    print(
        f"Saved to: {output_file}"
    )


if __name__ == "__main__":

    main()