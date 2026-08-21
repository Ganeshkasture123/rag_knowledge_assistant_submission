from app.ingestion import split_text

def test_split_text():
    text = "Sentence one. " * 200
    chunks = split_text(text, size=100, overlap=20)
    assert chunks
