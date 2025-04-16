import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset from the specified location (Replace 'fjfj' with actual path)
dataset_path = "/content/Final_Expanded_1000_Words_dataset.csv"
df = pd.read_csv(dataset_path)

# Ensure only necessary columns exist
if "word" not in df.columns or "url" not in df.columns:
    raise ValueError("Dataset must contain 'word' and 'url' columns.")

# Drop missing values
df = df.dropna(subset=["word", "url"])

# Load BERT-based embedding model
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode words using BERT
word_embeddings = bert_model.encode(df["word"].tolist())

# Apply Agglomerative Clustering
num_clusters = 10  # Adjust based on dataset size
clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
df["cluster"] = clustering_model.fit_predict(word_embeddings)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    word_embeddings, df["cluster"], test_size=0.2, random_state=42, stratify=df["cluster"]
)

# Train CatBoost Classifier
catboost_model = CatBoostClassifier(iterations=300, depth=7, learning_rate=0.1, verbose=0)
catboost_model.fit(X_train, y_train)

# Predictions
y_pred = catboost_model.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Function to recommend similar words and their URLs
def recommend_similar_words(input_word, top_n=5):
    if input_word not in df["word"].tolist():
        return "Word not in vocabulary."

    # Get BERT embedding for input word (✅ Fix shape issue)
    input_embedding = bert_model.encode([input_word]).reshape(1, -1)

    # Predict cluster
    predicted_cluster = int(catboost_model.predict(input_embedding)[0])

    # Filter words from the predicted cluster
    cluster_df = df[df["cluster"] == predicted_cluster]
    if cluster_df.empty:
        return "No words found in the predicted cluster."

    cluster_df = cluster_df[["word", "url"]]

    # Compute cosine similarity
    cluster_embeddings = bert_model.encode(cluster_df["word"].tolist())
    similarities = cosine_similarity(input_embedding, cluster_embeddings).flatten()

    # Sort and return top-N similar words
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommendations = cluster_df.iloc[top_indices].to_dict(orient="records")

    return recommendations



# Example Recommendation
print(f"Recommended words for 'truck': {recommend_similar_words('truck')}")
