import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import datasets

nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)


# Example questions
dataset = datasets.load_dataset(path="ai2_arc", name="ARC-Challenge")

questions = dataset["validation"]["question"]

# Preprocess questions
preprocessed_questions = [preprocess_text(question) for question in questions]

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_questions)

# Clustering using K-means
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Print clusters
for i in range(num_clusters):
    print(f"Cluster {i}:")
    for idx, label in enumerate(labels):
        if label == i:
            print(f" - {questions[idx]}")

# Visualization using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X.toarray())

# Plot the clusters
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
for i in range(num_clusters):
    points = X_tsne[labels == i]
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], label=f'Cluster {i}')
plt.legend()
plt.show()
