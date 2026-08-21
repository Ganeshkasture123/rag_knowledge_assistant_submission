from app.index import create_index


if __name__ == "__main__":

    index_name = create_index()

    print(
        f"Created index: {index_name}"
    )