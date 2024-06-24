import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN



class RecommendationSystem:
    def __init__(self, train_data, df_user):
        """
        Initialize the recommendation system with ratings and user data.
        """
        self.df_ratings = train_data
        self.df_user = df_user

    def cluster_users_kmeans(self, encoded_features, n_clusters=3):
        """
        Cluster users using k-means clustering on the encoded features.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(encoded_features)
        self.df_user['Cluster'] = clusters
        return clusters
    
    def cluster_users_dbscan(self, encoded_features, eps=0.5, min_samples=5):
        """
        Cluster users using DBSCAN clustering on the encoded features.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(encoded_features)
        self.df_user['Cluster'] = clusters
        return clusters

    def predict_rating(self, user_id, item_id):
        """
        Predict the rating for a given user and item.
        """
        user_cluster = self.df_user[self.df_user['UID'] == user_id]['Cluster'].values[0]
        cluster_users = self.df_user[self.df_user['Cluster'] == user_cluster]['UID'].values
        cluster_ratings = self.df_ratings[(self.df_ratings['UID'].isin(cluster_users)) & (self.df_ratings['MID'] == item_id)]
        if len(cluster_ratings) > 0:
            return cluster_ratings['rate'].mean()
        else:
            return self.df_ratings[self.df_ratings['MID'] == item_id]['rate'].mean()

    def evaluate(self, test_data):
        """
        Evaluate the recommendation system using RMSE on the test set.
        """
        y_true = []
        y_pred = []
        for index, row in test_data.iterrows():
            y_true.append(row['rate'])
            y_pred.append(self.predict_rating(row['UID'], row['MID']))
        return mean_squared_error(y_true, y_pred, squared=False)

    def plot_feature_correlation(self, path=None, mode='tf'):
        """
        Plot the correlation matrix of graph-based features.
        """
        graph_features = ['PR', 'CD', 'CC', 'CB', 'LC', 'AND']
        corr_matrix = self.df_user[graph_features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Graph-Based Features')
        if path:
            plt.savefig(path+'graph_feature_correlation_'+mode+'.png')