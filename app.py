from flask import Flask, render_template, request
from engine.indexer import SearchIndexer
from engine.search import SearchEngine

app = Flask(__name__)

# Step 1: Build index when app starts
indexer = SearchIndexer()
indexer.build_index()
tfidf_matrix, vectorizer, filenames = indexer.get_index_data()
search_engine = SearchEngine(tfidf_matrix, vectorizer, filenames)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        results = search_engine.search(query)
    return render_template("index.html", results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
