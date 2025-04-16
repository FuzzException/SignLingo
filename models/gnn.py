import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure Spacy Model is Installed
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading 'en_core_web_md' model...")
    import spacy.cli
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")


class GNNRecommender(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNRecommender, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class SignLingoGNNRecommender:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()

        # Prepare data and store additional features
        self._prepare_data()

        # Create graph data object
        self.graph_data = Data(x=self.X_tensor, edge_index=self.edge_index)

    def _prepare_data(self):
        print("Generating word embeddings...")

        # Generate word embeddings
        self.data['vector'] = self.data['word'].apply(lambda x: nlp(x).vector)
        self.word_embeddings = np.vstack(self.data['vector'].values)

        # Generate additional features
        self.additional_features = self._create_additional_features()

        # Combine word embeddings + additional features
        if self.additional_features is not None:
            self.X = np.hstack([self.word_embeddings, self.additional_features])
        else:
            self.X = self.word_embeddings

        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Convert to PyTorch tensor
        self.X_tensor = torch.tensor(self.X_scaled, dtype=torch.float)

        # Generate edges
        self.edge_index = self._create_edges()

    def _create_additional_features(self):
        features = []
        if 'difficulty' in self.data.columns:
            features.append(self.data['difficulty'].values.reshape(-1, 1))
        if 'views' in self.data.columns:
            views = np.log1p(self.data['views'].values).reshape(-1, 1)
            features.append(views)
        if 'rating' in self.data.columns:
            features.append(self.data['rating'].values.reshape(-1, 1))
        features.append(self.data['word'].str.len().values.reshape(-1, 1))
        return np.hstack(features) if features else None

    def _create_edges(self):
        sim_matrix = cosine_similarity(self.X_scaled)
        threshold = 0.6
        edge_index = []
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                if sim_matrix[i, j] > threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
        return torch.tensor(edge_index, dtype=torch.long).t()

    def _train_model(self, epochs=100, lr=0.01):
        print("Training GNN model...")

        # Define the GNN model
        self.hidden_channels = 128  # Define hidden layer size
        self.model = GNNRecommender(self.X_tensor.shape[1], self.hidden_channels, self.X_tensor.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.graph_data.x, self.graph_data.edge_index)
            mse_loss = F.mse_loss(out, self.graph_data.x)

            # Total loss (Adding optional classification loss for improvement)
            loss = mse_loss
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.final_embeddings = out.detach().numpy()

    def recommend(self, current_word, top_n=5):
        word_vector = nlp(current_word).vector.reshape(1, -1)

        additional_features = []
        if self.additional_features is not None:
            if 'difficulty' in self.data.columns:
                additional_features.append(self.data['difficulty'].mean())
            if 'views' in self.data.columns:
                additional_features.append(np.log1p(self.data['views'].mean()))
            if 'rating' in self.data.columns:
                additional_features.append(self.data['rating'].mean())
            additional_features.append(len(current_word))

            # Ensure additional features match trained dimensions
            while len(additional_features) < self.additional_features.shape[1]:
                additional_features.append(0)

        # Combine word vector and additional features
        if self.additional_features is not None:
            word_features = np.hstack([word_vector, np.array(additional_features).reshape(1, -1)])
        else:
            word_features = word_vector

        # Prevent data leakage by using fitted scaler
        try:
            word_features_scaled = self.scaler.transform(word_features)
        except ValueError:
            word_features_scaled = np.zeros_like(self.X_scaled[0])

        # Compute similarity scores
        similarities = cosine_similarity(word_features_scaled, self.X_scaled)[0]
        self.data['similarity_score'] = similarities
        recommendations = self.data.sort_values(by='similarity_score', ascending=False).head(top_n)

        return recommendations[['url', 'word', 'similarity_score']]

    def evaluate_model(self):
        self.model.eval()

        with torch.no_grad():
            predicted_embeddings = self.model(self.graph_data.x, self.graph_data.edge_index)

        y_true = self.X_scaled
        y_pred = predicted_embeddings.cpu().numpy()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"\nModel Evaluation:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")


# 🚀 Sample usage
sample_data = pd.read_csv("/Users/sarahgteerthan/Desktop/models/Final_Expanded_1000_Words_dataset.csv")
recommender = SignLingoGNNRecommender(sample_data)

# Train model
recommender._train_model()

# Get recommendations
print("\nRecommendations for 'car':")
print(recommender.recommend('car', top_n=5))

# Evaluate the model
recommender.evaluate_model()
