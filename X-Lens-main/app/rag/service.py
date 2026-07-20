class RAGService:
    def __init__(self, retriever=None):
        self.retriever = retriever

    def context_for(self, query: str) -> tuple[str, list[dict]]:
        if not self.retriever:
            return "", []
        hits = self.retriever.retrieve(query)
        context = "\n\n".join(
            f"Source: {hit.get('source_url', 'unknown')}\n{hit['text']}" for hit in hits
        )
        return context, hits
