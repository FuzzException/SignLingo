#FINAL FOR LightGBM with full dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import hdbscan
from scipy.spatial.distance import cdist
import umap
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load and explore your dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")

    # Make sure the column names match what we need
    # If your CSV has different column names, rename them here
    if 'word' not in df.columns:
        if 'term' in df.columns:
            df.rename(columns={'term': 'word'}, inplace=True)
        elif 'Word' in df.columns:
            df.rename(columns={'Word': 'word'}, inplace=True)
        else:
            # Try to identify the column that contains words
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].str.isalpha().mean() > 0.8:
                    print(f"Renaming column '{col}' to 'word'")
                    df.rename(columns={col: 'word'}, inplace=True)
                    break

    # Similarly for video_link column
    if 'video_link' not in df.columns:
        if 'url' in df.columns:
            df.rename(columns={'url': 'video_link'}, inplace=True)
        elif 'URL' in df.columns:
            df.rename(columns={'URL': 'video_link'}, inplace=True)
        elif 'video_url' in df.columns:
            df.rename(columns={'video_url': 'video_link'}, inplace=True)
        else:
            # Try to identify the column that contains URLs
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].str.contains('http').mean() > 0.5:
                    print(f"Renaming column '{col}' to 'video_link'")
                    df.rename(columns={col: 'video_link'}, inplace=True)
                    break

    # Make sure we have the required columns
    if 'word' not in df.columns or 'video_link' not in df.columns:
        print("Warning: Required columns 'word' and 'video_link' not found. Please check your CSV file.")
        print(f"Available columns: {df.columns.tolist()}")

        # As a last resort, create default columns
        if 'word' not in df.columns:
            print("Creating a default 'word' column using the first text column")
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            if text_cols:
                df['word'] = df[text_cols[0]]

        if 'video_link' not in df.columns:
            print("Creating a placeholder 'video_link' column")
            df['video_link'] = [f"https://example.com/video_{i}" for i in range(len(df))]

    # Clean up the word column - lowercase, strip whitespace
    df['word'] = df['word'].str.lower().str.strip()

    # Remove duplicates if any
    df = df.drop_duplicates(subset=['word'])

    return df

# Step 2: Generate embeddings using a pre-trained model
def generate_embeddings(df):
    print("Generating word embeddings...")
    # Using Sentence Transformers for word embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for each word
    word_embeddings = {}
    for word in df['word'].unique():
        word_embeddings[word] = model.encode([word])[0]

    # Convert embeddings to a DataFrame for easier manipulation
    embeddings_list = list(word_embeddings.values())
    words_list = list(word_embeddings.keys())

    # Create a DataFrame with word and its embedding
    embeddings_df = pd.DataFrame({
        'word': words_list,
        'embedding': embeddings_list
    })

    # Merge with original DataFrame
    df_with_embeddings = df.merge(embeddings_df, on='word')

    return df_with_embeddings, word_embeddings, embeddings_list, words_list

# Step 3: Automatically categorize words using clustering
def auto_categorize_words(df, word_embeddings):
    print("Automatically categorizing words using clustering...")
    words = list(word_embeddings.keys())
    embeddings = np.array(list(word_embeddings.values()))

    # Option 1: K-Means clustering
    # We'll try to determine the optimal number of clusters using the elbow method
    inertias = []
    max_clusters = min(15, len(words) - 1)  # Don't try more clusters than we have words, minus 1

    if max_clusters > 1:
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)

        # Plot the elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), inertias, marker='o')
        plt.title('Elbow Method For Optimal Number of Clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.savefig('elbow_curve.png')
        plt.close()

        # Heuristic: Find the "elbow" point - where the rate of decrease sharply changes
        # Simple approach: look for the largest second derivative
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        if len(second_diffs) > 0:
            elbow_idx = np.argmax(second_diffs) + 2  # +2 because of the two diffs and 0-indexing
            optimal_k = elbow_idx + 2  # +2 because our range started at 2
        else:
            optimal_k = 3  # Default if we can't compute second derivatives
    else:
        optimal_k = 1

    print(f"Estimated optimal number of clusters: {optimal_k}")

    # Option 2: HDBSCAN for clustering (often better for text embeddings)
    # Reduce dimensionality first for better clustering
    umap_reducer = umap.UMAP(n_components=10, random_state=42)
    umap_embeddings = umap_reducer.fit_transform(embeddings)

    # Apply HDBSCAN clustering
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, len(words) // 20),
                                        min_samples=1,
                                        prediction_data=True)
    hdbscan_labels = hdbscan_clusterer.fit_predict(umap_embeddings)

    # Fall back to K-means if HDBSCAN didn't find good clusters
    if len(np.unique(hdbscan_labels)) < 2 or (len(np.unique(hdbscan_labels)) == 2 and -1 in hdbscan_labels):
        print("HDBSCAN didn't find clear clusters, falling back to K-means")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
    else:
        cluster_labels = hdbscan_labels
        # Relabel noise points (-1) to their nearest cluster
        if -1 in cluster_labels:
            noise_indices = np.where(cluster_labels == -1)[0]
            for idx in noise_indices:
                # Find the nearest cluster center
                if len(np.unique(cluster_labels)) > 1:  # Ensure we have clusters
                    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
                    cluster_points = [embeddings[cluster_labels == c].mean(axis=0) for c in unique_clusters]
                    distances = cdist([embeddings[idx]], cluster_points, 'euclidean')[0]
                    nearest_cluster = unique_clusters[np.argmin(distances)]
                    cluster_labels[idx] = nearest_cluster

    # Create a mapping from words to clusters
    word_to_cluster = {word: label for word, label in zip(words, cluster_labels)}

    # Add cluster labels to the dataframe
    df['auto_category'] = df['word'].map(word_to_cluster)

    # Visualize the clusters in 2D
    # Use t-SNE for dimensionality reduction to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Cluster')

    # Annotate some points with word labels (annotate a subset to avoid overcrowding)
    max_annotations = min(50, len(words))
    step = max(1, len(words) // max_annotations)
    for i in range(0, len(words), step):
        plt.annotate(words[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)

    plt.title('Word Clusters Visualization')
    plt.savefig('word_clusters.png')
    plt.close()

    # Print examples of words in each cluster
    print("\nWord categories discovered by clustering:")
    for cluster in sorted(np.unique(cluster_labels)):
        cluster_words = [words[i] for i in range(len(words)) if cluster_labels[i] == cluster]
        cluster_examples = cluster_words[:min(5, len(cluster_words))]
        print(f"Cluster {cluster}: {', '.join(cluster_examples)}{'...' if len(cluster_words) > 5 else ''}")

    return df, word_to_cluster

# Step 4: Calculate similarity scores between words
def calculate_similarities(word_embeddings):
    print("Calculating similarity scores between words...")
    words = list(word_embeddings.keys())
    embeddings = np.array(list(word_embeddings.values()))

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Create a DataFrame for the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix, index=words, columns=words)

    return similarity_df

# Step 5: Prepare data for LightGBM model
def prepare_training_data(df, similarity_df, word_embeddings):
    print("Preparing training data for LightGBM model...")
    # Create training data with features and target
    training_data = []

    words = list(word_embeddings.keys())

    for i, word1 in enumerate(words):
        category1 = df[df['word'] == word1]['auto_category'].iloc[0]

        for j, word2 in enumerate(words):
            if word1 != word2:
                category2 = df[df['word'] == word2]['auto_category'].iloc[0]
                similarity = similarity_df.loc[word1, word2]

                # Features: word embedding of word1 and word2
                embedding1 = word_embeddings[word1]
                embedding2 = word_embeddings[word2]

                # Combine features
                features = np.concatenate([embedding1, embedding2])

                # Target: 1 if same category, 0 otherwise
                target = 1 if category1 == category2 else 0

                training_data.append({
                    'word1': word1,
                    'word2': word2,
                    'features': features,
                    'similarity': similarity,
                    'target': target
                })

    # Convert to pandas DataFrame
    train_df = pd.DataFrame(training_data)

    # Split features into separate columns for LightGBM
    feature_columns = [f'feature_{i}' for i in range(len(train_df['features'].iloc[0]))]
    features_df = pd.DataFrame(train_df['features'].tolist(), columns=feature_columns)

    # Combine with training DataFrame
    train_df = pd.concat([train_df.drop('features', axis=1), features_df], axis=1)

    return train_df

# Step 6: Train LightGBM model
def train_lightgbm_model(train_df):
    print("Training LightGBM model...")
    # Split data into features and target
    feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
    X = train_df[feature_cols]
    y = train_df['target']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Train LightGBM model
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[valid_data],
        callbacks=[
        lgb.log_evaluation(period=100),  # This replaces verbose_eval
        # You can add early_stopping here too
        lgb.early_stopping(10)]
    )

    '''from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit only on training data
    X_test = scaler.transform(X_test)  # Transform test data without fitting'''


    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("Train R²:", r2_score(y_train, y_pred_train))
    print("Test R²:", r2_score(y_test, y_pred_test))  # Should be lower than train score

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return model, X_test, y_test, y_pred

# Step 7: Create the recommendation function
def recommend_words(input_word, df, similarity_df, model, word_embeddings, top_n=5):
    if input_word not in df['word'].unique():
        print(f"Word '{input_word}' not found in the dataset")
        return []

    # Get the category of the input word
    category = df[df['word'] == input_word]['auto_category'].iloc[0]

    # Get embedding of the input word
    input_embedding = word_embeddings[input_word]

    # Prepare data for model prediction
    recommendations = []

    for word in df['word'].unique():
        if word != input_word:
            # Get embedding of the candidate word
            candidate_embedding = word_embeddings[word]

            # Get category of candidate word
            candidate_category = df[df['word'] == word]['auto_category'].iloc[0]

            # Calculate similarity score from pre-computed similarity matrix
            similarity = similarity_df.loc[input_word, word]

            # Add to recommendations with all relevant information
            recommendations.append({
                'word': word,
                'category': candidate_category,
                'same_category': category == candidate_category,
                'similarity': similarity,
                'video_link': df[df['word'] == word]['video_link'].iloc[0]
            })

    # Sort by similarity score
    recommendations = sorted(recommendations, key=lambda x: x['similarity'], reverse=True)

    # Filter by category
    category_recommendations = [r for r in recommendations if r['same_category']]

    # If not enough same-category recommendations, add some from other categories
    if len(category_recommendations) < top_n:
        other_recommendations = [r for r in recommendations if not r['same_category']]
        category_recommendations.extend(other_recommendations[:top_n - len(category_recommendations)])

    # Return top N recommendations
    return category_recommendations[:top_n]

# Step 8: Visualize similarity scores
def visualize_similarity(similarity_df, df):
    print("Visualizing word similarities...")
    # Get a subset of words for visualization (to avoid overcrowding)
    categories = df['auto_category'].unique()
    sample_words = []

    for category in categories:
        category_words = df[df['auto_category'] == category]['word'].unique()
        if len(category_words) > 0:
            sample_words.extend(category_words[:min(3, len(category_words))])

    # Limit to a reasonable number for visualization
    max_words = min(20, len(sample_words))
    sample_words = sample_words[:max_words]

    # Create a subset of the similarity matrix
    subset_similarity = similarity_df.loc[sample_words, sample_words]

    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(subset_similarity, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Word Similarity Heatmap')
    plt.tight_layout()
    plt.savefig('word_similarity_heatmap.png')
    plt.close()

# Function to evaluate the recommendation system
def evaluate_recommendations(recommend_function, df, word_embeddings, n_samples=10):
    print("\nEvaluating recommendation system...")
    # Sample a few words to test
    test_words = np.random.choice(list(df['word'].unique()), size=min(n_samples, len(df['word'].unique())), replace=False)

    results = []
    for word in test_words:
        recommendations = recommend_function(word)
        avg_similarity = np.mean([rec['similarity'] for rec in recommendations]) if recommendations else 0
        same_category_count = sum(1 for rec in recommendations if rec['same_category']) if recommendations else 0

        results.append({
            'word': word,
            'category': df[df['word'] == word]['auto_category'].iloc[0],
            'avg_similarity': avg_similarity,
            'same_category_ratio': same_category_count / len(recommendations) if recommendations else 0,
            'num_recommendations': len(recommendations)
        })

    results_df = pd.DataFrame(results)

    print(f"Average similarity score across all recommendations: {results_df['avg_similarity'].mean():.4f}")
    print(f"Average same-category ratio: {results_df['same_category_ratio'].mean():.4f}")

    return results_df

# Main function to run the entire pipeline
def main(file_path):
    print("Starting word recommendation system with your data...")
    # Load data
    df = load_data(file_path)

    # Generate embeddings
    df_with_embeddings, word_embeddings, embeddings_list, words_list = generate_embeddings(df)

    # Automatically categorize words
    df_categorized, word_to_cluster = auto_categorize_words(df_with_embeddings, word_embeddings)

    # Calculate similarities
    similarity_df = calculate_similarities(word_embeddings)

    # Visualize similarity
    visualize_similarity(similarity_df, df_categorized)

    # Prepare training data
    train_df = prepare_training_data(df_categorized, similarity_df, word_embeddings)

    # Train LightGBM model
    model, X_test, y_test, y_pred = train_lightgbm_model(train_df)


    ##########

    # Create recommendation function
    def get_recommendations(word, top_n=5):
        return recommend_words(word, df_categorized, similarity_df, model, word_embeddings, top_n)

    # Test with a few examples
    print("\nTesting recommendation system with examples:")
    for example_word in np.random.choice(df['word'].unique(), size=min(3, len(df['word'].unique())), replace=False):
        recommendations = get_recommendations(example_word)

        print(f"\nRecommendations for '{example_word}':")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['word']} (Similarity: {rec['similarity']:.4f}, Same Category: {'Yes' if rec['same_category'] else 'No'})")
            print(f"   Video Link: {rec['video_link']}")

    # Evaluate recommendations
    eval_results = evaluate_recommendations(get_recommendations, df_categorized, word_embeddings)

    # Save the trained model and necessary data for later use
    model_data = {
        'word_embeddings': word_embeddings,
        'similarity_df': similarity_df,
        'categorized_df': df_categorized
    }

    # Return the recommendation function for interactive use
    return get_recommendations, model_data

# Interactive component for user input
def interactive_recommendations(recommend_function):
    while True:
        word = input("\nEnter a word for recommendations (or 'q' to quit): ")
        if word.lower() == 'q':
            break

        recommendations = recommend_function(word)

        if not recommendations:
            continue

        print(f"\nRecommendations for '{word}':")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['word']} (Similarity: {rec['similarity']:.4f}, Same Category: {'Yes' if rec['same_category'] else 'No'})")
            print(f"   Video Link: {rec['video_link']}")

# Run the system with your data
if __name__ == "__main__":
    file_path = "/Users/sarahgteerthan/Desktop/models/filtered_categories(1).csv"  # Your actual file path
    recommend_function, model_data = main(file_path)

    # Option to run interactive mode
    interactive_recommendations(recommend_function)