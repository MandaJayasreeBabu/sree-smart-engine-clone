from sklearn.metrics.pairwise import cosine_similarity

class SearchEngine:
    def __init__(self, tfidf_matrix, vectorizer, filenames, docs_path="data/docs/"):
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer
        self.filenames = filenames
        self.docs_path = docs_path

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_indices = similarities.argsort()[::-1][:top_k]
        results = []

        for i in ranked_indices:
            if similarities[i] > 0:
                filepath = f"{self.docs_path}{self.filenames[i]}"
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                snippet = self._extract_snippet(content, query)
                results.append({
                    "filename": self.filenames[i],
                    "score": similarities[i],
                    "snippet": snippet
                })
        return results

    def _extract_snippet(self, text, query):
        query = query.lower()
        text_lower = text.lower()
        idx = text_lower.find(query)
        if idx != -1:
            start = max(0, idx - 50)
            end = min(len(text), idx + 50)
            return "..." + text[start:end] + "..."
        else:
            return text[:100] + "..."
