# Importing libraries
import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from collections import OrderedDict
import copy

# Suppress warnings
warnings.filterwarnings('ignore')

# Load datasets
def load_data():
    users = pd.read_csv('datasets/ml-1m/users.dat', sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    ratings = pd.read_csv('datasets/ml-1m/ratings.dat', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    movies = pd.read_csv('datasets/ml-1m/movies.dat', sep='::', engine='python', encoding='ISO-8859-1', names=['MovieID', 'Title', 'Genres'])
    return users, ratings, movies

# Display basic statistics
def display_statistics(users, ratings, movies):
    print("Users Data:")
    print(users.describe())
    print("\nMovies Data:")
    print(movies.describe())
    print("\nRatings Data:")
    print(ratings.describe())
    print("\nMissing Values in Users Data:")
    print(users.isnull().sum())
    print("\nMissing Values in Movies Data:")
    print(movies.isnull().sum())
    print("\nMissing Values in Ratings Data:")
    print(ratings.isnull().sum())

# Plotting functions
def plot_distributions(ratings, users, movies):
    plt.figure(figsize=(10, 6))
    plt.hist(ratings['Rating'], bins=5, edgecolor='black', align='mid', rwidth=0.2, color="gray")
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Distribution of Ratings')
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.countplot(data=users, x='Gender')
    plt.title('Gender Distribution')
    plt.show()

    ratings_per_user = ratings.groupby('UserID').size()
    plt.figure(figsize=(10, 6))
    plt.hist(ratings_per_user, bins=50, edgecolor='black', color="gray")
    plt.title('Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.show()

    all_genres = movies['Genres'].str.split('|', expand=True).stack().reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.barh(all_genres.value_counts().index, all_genres.value_counts(), edgecolor='black', color="gray")
    plt.title('Top Genres')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genre')
    plt.show()

# Prepare data for modeling
def preprocess_data(users, ratings, movies):
    # Drop duplicates
    users = users.drop_duplicates()
    movies = movies.drop_duplicates()
    ratings = ratings.drop_duplicates()

    # Encode features
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(movies['Genres'].str.split('|'))
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=movies.index)

    occupation_encoder = OneHotEncoder(sparse=False)
    occupations_encoded = occupation_encoder.fit_transform(users[['Occupation']])
    occupations_df = pd.DataFrame(occupations_encoded, columns=occupation_encoder.get_feature_names_out(['Occupation']), index=users.index)

    gender_encoder = LabelEncoder()
    users['Gender_encoded'] = gender_encoder.fit_transform(users['Gender'])

    age_encoder = LabelEncoder()
    users['Age_encoded'] = age_encoder.fit_transform(users['Age'])

    return users, genres_df, occupations_df, gender_encoder, age_encoder

# Dataset class
class ClientDataset(Dataset):
    def __init__(self, users, movies, ratings, mlb, occupation_encoder, genre_encoder, age_encoder):
        self.data = ratings.merge(users, on='UserID').merge(movies, on='MovieID')
        self.mlb = mlb
        self.occupation_encoder = occupation_encoder
        self.genre_encoder = genre_encoder
        self.age_encoder = age_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (
            row['UserID'],
            row['MovieID'],
            torch.tensor(self.mlb.transform([row['Genres'].split('|')])[0], dtype=torch.float),
            torch.tensor(self.occupation_encoder.transform([[row['Occupation']]])[0], dtype=torch.float),
            torch.tensor(row['Age_encoded'], dtype=torch.long),
            row['Gender_encoded'],
            row['Rating']
        )

# Model class
class ExpandedRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim, num_genres, num_occupations, num_ages, num_genders=2):
        super(ExpandedRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim)
        self.age_embedding = nn.Embedding(num_ages, embedding_dim)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim)
        self.genre_layer = nn.Linear(num_genres, embedding_dim)
        self.occupation_layer = nn.Linear(num_occupations, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, user_ids, movie_ids, genres, occupations, ages, genders):
        user_embedded = self.user_embedding(user_ids)
        movie_embedded = self.movie_embedding(movie_ids)
        age_embedded = self.age_embedding(ages)
        gender_embedded = self.gender_embedding(genders)
        genre_embedded = F.relu(self.genre_layer(genres.float()))
        occupation_embedded = F.relu(self.occupation_layer(occupations.float()))
        x = torch.cat((user_embedded, movie_embedded, genre_embedded, occupation_embedded, age_embedded, gender_embedded), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

# Federated learning functions
def average_weights(client_models):
    average_model_weights = OrderedDict()
    for k in client_models[0].state_dict().keys():
        average_model_weights[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(num_clients)], 0).mean(0)
    return average_model_weights

def train_federated_model(client_loaders, test_loader, global_model, num_clients, epochs, rounds, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001)

    client_losses = {i: [] for i in range(num_clients)}
    global_losses = []
    client_mae = {i: [] for i in range(num_clients)}
    global_maes = []

    for round in range(rounds):
        client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
        for client in range(num_clients):
            optimizer = torch.optim.Adam(client_models[client].parameters(), lr=0.001)
            los = []
            for epoch in range(epochs):
                client_models[client].train()
                running_loss = 0.0
                for batch_data in client_loaders[client]:
                    user_ids, movie_ids, genres, occupations, ages, genders, rating = [item.to(device) for item in batch_data]
                    optimizer.zero_grad()
                    outputs = client_models[client](user_ids, movie_ids, genres, occupations, ages, genders)
                    loss = criterion(outputs, rating.float())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                epoch_loss = running_loss / len(client_loaders[client])
                print(f"Client {client+1} - Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
                client_losses[client].append(epoch_loss)
                los.append(epoch_loss)
                torch.save(client_models[client].state_dict(), os.path.join(model_dir, f'client_{client+1}_round_{round+1}.pth'))
        for client in range(num_clients):
            predictions = []
            true_labels = []
            client_models[client].eval()
            with torch.no_grad():
                for data in client_loaders[client]:
                    user_ids, movie_ids, genres, occupations, ages, genders, rating = [item.to(device) for item in data]
                    outputs = client_models[client](user_ids, movie_ids, genres, occupations, ages, genders)
                    predictions.extend(outputs.cpu().numpy())
                    true_labels.extend(rating.cpu().numpy())

            mae = mean_absolute_error(true_labels, predictions)
            client_mae[client].append(mae)
        global_weights = average_weights(client_models)
        global_model.load_state_dict(global_weights)

        predictions = []
        true_labels = []
        global_model.eval()
        with torch.no_grad():
            for batch_data in test_loader:
               
                user_ids, movie_ids, genres, occupations, ages, genders, rating = [item.to(device) for item in batch_data]
                outputs = global_model(user_ids, movie_ids, genres, occupations, ages, genders)
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(rating.cpu().numpy())
        global_mae = mean_absolute_error(true_labels, predictions)
        global_maes.append(global_mae)
        print(f"After Round {round+1}, Global Model Test MAE: {global_mae:.4f}")




# Export accuracy and loss to CSV
data = [acc, c_loss]
data = list(map(list, zip(*data)))
column_names = ['Accuracy', 'Train Losses']
df1 = pd.DataFrame(data, columns=column_names)
df1.to_csv('results_ff.csv', index=False)

# Plot client training losses
fig, axs = plt.subplots(num_clients, 1, figsize=(10, 5 * num_clients))
for client, losses in client_losses.items():
    if not losses:
        print(f"No losses recorded for client {client + 1}. Skipping...")
        continue
    axs[client].plot(range(1, epochs * rounds + 1), losses, label=f'Client {client + 1}')
    axs[client].set_xlabel('Epoch')
    axs[client].set_ylabel('Loss')
    axs[client].legend()
    axs[client].grid(True)
plt.tight_layout()
plt.show()

# Plot final Mean Absolute Error (MAE) for clients and global model
clients = list(range(1, num_clients + 1))
final_maes = [maes[-1] for maes in client_mae.values()]
plt.figure(figsize=(10, 5))
plt.bar(clients, final_maes, color='blue', alpha=0.7, label='Local Models')
plt.axhline(y=global_maes[-1], color='r', linestyle='--', label='Global Model')
plt.xlabel('Client Number')
plt.ylabel('Mean Absolute Error')
plt.title('Clients and Mean Absolute Error')
plt.legend()
plt.show()

# Plot Mean Absolute Error (MAE) across rounds for global and local models
rounds_list = list(range(1, rounds + 1))
plt.figure(figsize=(10, 5))

plt.plot(rounds_list, global_maes, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=12, label='Global Model')

local_avg_mae = [np.mean([client_mae[client][i] for client in range(num_clients)]) for i in range(rounds)]
plt.plot(rounds_list, local_avg_mae, color='blue', marker='s', linestyle='solid', linewidth=2, markersize=12, label='Local Models')

plt.xlabel('Round')
plt.ylabel('Mean Absolute Error')
plt.title('Global Model vs Local Model Mean Absolute Error')
plt.legend()
plt.show()
