import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import  MinMaxScaler
import argparse
import yaml
import tensorflow as tf
import os


from src.MovieLensData import MovieLensData
from src.GraphFeatures import GraphFeatures
from src.AutoEncoderModel_tf import AutoEncoderModel
from src.RecommendationSystem import RecommendationSystem


def setSeeds(seed = 0):
    # Set seed
    tf.random.set_seed(seed)
    
    # Set environment variable for TensorFlow (useful for certain operations)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Ensure reproducibility for TensorFlow dataset operations
    from tensorflow.python.framework import random_seed
    random_seed.set_random_seed(seed)

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='none', help='Path to the config file.')
    return parser.parse_args()

def getConfig(path):
    configFile = yaml.load(open(path, 'r'), Loader=yaml.Loader)
    return configFile

if __name__ == '__main__':
    # Set seeds for reproducibility
    setSeeds(0)

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

    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the autoencoder
    print("Training the autoencoder...")
    autoencoder = AutoEncoderModel(input_dim=X_train_scaled.shape[1], epochs=200, batch_size=256)
    autoencoder.build_autoencoder()
    history = autoencoder.train_autoencoder(X_train_scaled)

    # Get encoded features and plot training history
    print("Getting encoded features and plotting training history...")
    encoded_features = autoencoder.get_encoded_features(X_train_scaled)
    autoencoder.plot_training_history(history, path=plotPath)

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

    recommendation_system.plot_feature_correlation(path=plotPath, mode='tf')