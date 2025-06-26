import os
from sklearn.feature_extraction.text import TfidfVectorizer

class SearchIndexer:
    def __init__(self, docs_path="data/docs/"):
        self.docs_path = docs_path
        self.filenames = []
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

    def load_documents(self):
        for file in os.listdir(self.docs_path):
            if file.endswith(".txt"):
                path = os.path.join(self.docs_path, file)
                with open(path, 'r', encoding='utf-8') as f:
                    self.documents.append(f.read())
                    self.filenames.append(file)

    def build_index(self):
        self.load_documents()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def get_index_data(self):
        return self.tfidf_matrix, self.vectorizer, self.filenames
