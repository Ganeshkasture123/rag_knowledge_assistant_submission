from app.ingestion import ingest_directory


if __name__ == "__main__":

    count = ingest_directory(
        "knowledge_base"
    )

    print(
        f"Ingestion completed. "
        f"Total chunks: {count}"
    )