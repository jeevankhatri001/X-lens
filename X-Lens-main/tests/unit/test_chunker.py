from app.rag.chunker import chunk_text
def test_chunker_deduplicates():
    chunks=chunk_text('hello world '*100,'https://example.edu',size=20,overlap=0)
    assert len({c.content_hash for c in chunks})==len(chunks)
