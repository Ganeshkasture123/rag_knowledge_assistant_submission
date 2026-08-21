from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
)

from .config import (
    SEARCH_ENDPOINT,
    SEARCH_API_KEY,
    SEARCH_INDEX,
    SEMANTIC_CONFIG,
)


def create_index():
    client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
        ),

        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),

        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
        ),

        SearchableField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        SearchableField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        SimpleField(
            name="page",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),

        SearchableField(
            name="sheet",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        SimpleField(
            name="chunk_id",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),

        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="default-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="default-hnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="default-vector-profile",
                algorithm_configuration_name="default-hnsw",
            )
        ],
    )

    semantic = SemanticConfiguration(
        name=SEMANTIC_CONFIG,
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(
                field_name="title"
            ),
            content_fields=[
                SemanticField(
                    field_name="content"
                )
            ],
        ),
    )

    index = SearchIndex(
        name=SEARCH_INDEX,
        fields=fields,
        vector_search=vector_search,
        semantic_search={
            "configurations": [semantic]
        },
    )

    client.create_or_update_index(index)

    print(f"Index '{SEARCH_INDEX}' created successfully.")

    return SEARCH_INDEX