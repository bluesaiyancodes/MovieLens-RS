import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

from src.MovieLensData import MovieLensData
from src.GraphFeatures import GraphFeatures
from src.AutoEncoderModel_pt import AutoEncoder
from src.RecommendationSystem import RecommendationSystem



if __name__ == '__main__':

    # Load data
    data_path_100k = 'Datasets/100k/ml-100k/'
    data_path_1m = 'Datasets/1m/ml-1m/'

    # Choose the dataset - '100k' or '1m'
    dataType = '100k'



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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    num_epochs = 200
    batch_size = 256

    train_loader = torch.utils.data.DataLoader(X_train_tensor, batch_size=batch_size, shuffle=True)

    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = autoencoder(data)
            loss = criterion(output, data) + autoencoder.regularization_loss()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        loss_history.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

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
    recommendation_system.cluster_users(encoded_features, n_clusters=8)

    rmse = recommendation_system.evaluate(test_data)
    print(f'Root Mean Squared Error on Test Data: {rmse}')

    #recommendation_system.plot_feature_correlation(path="Plots/100k/", mode="pt")