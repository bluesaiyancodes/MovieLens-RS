import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random
import torch.backends.cudnn as cudnn

from src.MovieLensData import MovieLensData
from src.GraphFeatures import GraphFeatures
from src.AutoEncoderModel_pt import AutoEncoder
from src.RecommendationSystem import RecommendationSystem

def setSeeds():
    # Set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.set_grad_enabled(True)

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='none', help='Path to the config file.')
    return parser.parse_args()

def getConfig(path):
    configFile = yaml.load(open(path, 'r'), Loader=yaml.Loader)
    return configFile

if __name__ == '__main__':

    # Set seeds for reproducibility
    setSeeds()

    # Parse arguments
    userConfigPath = arg_init().config
    if userConfigPath == None:
        print("No configuration file specified")
        exit(0)
    config = getConfig(userConfigPath)


    # Load data
    data_path_100k = config['DATASET']['DATA_PATH_100K']
    data_path_1m = config['DATASET']['DATA_PATH_1M']

    # Choose the dataset - '100k' or '1m'
    dataType = config["EXPERIMENT"]["TYPE"]



    if dataType == '100k':
        data_path = data_path_100k
    else:
        data_path = data_path_1m

    plotPath = f'Plots/{dataType}/'
    os.makedirs('Plots', exist_ok=True)
    os.makedirs(plotPath, exist_ok=True)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = MovieLensData(data_path, mode=dataType)
    data.preprocess_user_data()

    # Generate similarity graph and extract graph features
    print("Generating similarity graph and extracting graph features...")
    graph_feat = GraphFeatures(data.df_ratings, data.alpha_coef)
    G = graph_feat.generate_similarity_graph()
    data.df_user = graph_feat.extract_graph_features(G, data.df_user)


    # Split data before training
    train_data, val_data, test_data = data.split_data()

    # Combining user-side information with graph features
    print("Combining user-side information with graph features...")
    X_train = data.df_user.loc[data.df_user['UID'].isin(train_data['UID'])].drop(columns=['UID'])
    X_train.fillna(0, inplace=True)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Converting to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)

    # Autoencoder model
    print("Training the autoencoder...")
    input_dim = X_train_scaled.shape[1]
    autoencoder = AutoEncoder(input_dim=input_dim)

    # Training the autoencoder
    if config["EXPERIMENT"]["LOSS"] == "MSE":
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    optimizer = optim.Adam(autoencoder.parameters(), lr=config["EXPERIMENT"]["LEARNING_RATE"], weight_decay=config["EXPERIMENT"]["WEIGHT_DECAY"])

    num_epochs = config["EXPERIMENT"]["NUM_EPOCHS"]
    batch_size = config["EXPERIMENT"]["BATCH_SIZE"]

    train_loader = torch.utils.data.DataLoader(X_train_tensor, batch_size=batch_size, shuffle=True)

    loss_history = []

    with tqdm(range(num_epochs), desc="Training Epochs") as outer:
        for epoch in outer:
            epoch_loss = 0
            # Create the inner tqdm object
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as inner:
                for data in inner:
                    optimizer.zero_grad()
                    output = autoencoder(data)
                    loss = criterion(output, data) + autoencoder.regularization_loss()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    # Update the inner tqdm object with the current loss
                    inner.set_postfix(loss=loss.item())
            epoch_loss /= len(train_loader)
            loss_history.append(epoch_loss)
            # Update the outer tqdm object with the epoch loss
            outer.set_postfix(loss=epoch_loss)

    # Extracting encoded features
    print("Getting encoded features and plotting training history...")
    encoded_features = autoencoder.encoder(X_train_tensor).detach().numpy()

    # Plot training history (loss over epochs)
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(f'{plotPath}/loss_plot_PT.png')

    # Use the DataFrame objects for the recommendation system
    print("Initializing recommendation system...")
    # load dataframe
    data = MovieLensData(data_path, mode=dataType)

    # Initialize recommendation system
    print("Initializing recommendation system...")
    recommendation_system = RecommendationSystem(data.df_ratings, data.df_user)

    # Cluster users and evaluate the recommendation system
    print("Clustering users and evaluating the recommendation system...")
    if config["CLUSTERING"]['METHOD'] == "k-means":
        recommendation_system.cluster_users_kmeans(encoded_features, n_clusters=config["CLUSTERING"]["NUM_CLUSTERS"])
    else:
        recommendation_system.cluster_users_dbscan(encoded_features, eps=config["CLUSTERING"]["EPS"], min_samples=config["CLUSTERING"]["MIN_SAMPLES"])

    rmse = recommendation_system.evaluate(test_data)
    print(f'Root Mean Squared Error on Test Data: {rmse}')

    #recommendation_system.plot_feature_correlation(path="Plots/100k/", mode="pt")