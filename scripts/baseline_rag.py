from azure.search.documents.models import VectorizedQuery

from app.clients import (
    openai_client,
    search_client,
)

from app.config import (
    EMBEDDING_DEPLOYMENT,
    CHAT_DEPLOYMENT,
)


def create_embedding(text):

    response = (
        openai_client
        .embeddings
        .create(
            model=EMBEDDING_DEPLOYMENT,
            input=text,
        )
    )

    return response.data[0].embedding


def baseline_answer(question):

    # --------------------------------------------
    # Only one query
    # --------------------------------------------

    vector = create_embedding(
        question
    )

    vector_query = VectorizedQuery(

        vector=vector,

        k_nearest_neighbors=5,

        fields="contentVector",
    )

    # --------------------------------------------
    # Vector-only retrieval
    # --------------------------------------------

    results = search_client.search(

        search_text=None,

        vector_queries=[
            vector_query
        ],

        top=5,

        select=[
            "content",
            "title",
            "source",
            "page",
            "sheet",
            "chunk_id",
        ],
    )

    results = list(
        results
    )

    if not results:

        return {
            "answer": (
                "I don't know."
            ),
            "citations": [],
        }

    context_parts = []

    citations = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        context_parts.append(
            f"""
[{index}]
Source: {result.get('source', '')}

{result.get('content', '')}
"""
        )

        citations.append(
            {
                "id": index,
                "source": result.get(
                    "source",
                    "",
                ),
                "page": result.get(
                    "page",
                    0,
                ),
            }
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------
    # Direct LLM generation
    # --------------------------------------------

    response = (
        openai_client
        .responses
        .create(

            model=CHAT_DEPLOYMENT,

            input=[
                {
                    "role": "system",
                    "content": """
Answer the question using the context.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Context:

{context}


Question:

{question}
"""
                },
            ],

            max_output_tokens=700,
        )
    )

    return {
        "answer": (
            response.output_text
        ),
        "citations": citations,
    }