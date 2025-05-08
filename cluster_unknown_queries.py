import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import nltk
from nltk.corpus import stopwords
import string
import os

# Download stopwords if not yet downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ========== Step 1: Load unknown queries ==========
with open('unknown_queries.json', 'r') as f:
    queries = json.load(f)

# Extract only user messages
messages = [entry['user_message'] for entry in queries]

# ========== Step 2: Preprocess text ==========
def preprocess(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return ' '.join(tokens)

processed_messages = [preprocess(msg) for msg in messages]

# ========== Step 3: Vectorize using TF-IDF ==========
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_messages)

# ========== Step 4: Cluster using DBSCAN ==========
# Use cosine similarity with DBSCAN
clustering = DBSCAN(eps=0.5, min_samples=1, metric='cosine')
labels = clustering.fit_predict(X)

# ========== Step 5: Group messages by cluster ==========
clustered = {}
for idx, label in enumerate(labels):
    cluster_key = f"cluster_{label}"
    if cluster_key not in clustered:
        clustered[cluster_key] = []
    clustered[cluster_key].append({
        "timestamp": queries[idx]["timestamp"],
        "user_message": queries[idx]["user_message"]
    })

# ========== Step 6: Save the result ==========
with open('cluster_unknown.json', 'w') as f:
    json.dump(clustered, f, indent=4)

print("Clustering complete. Output saved to 'cluster_unknown.json'.")
