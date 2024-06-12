import networkx as nx
import itertools
import collections
from src.MovieLensData import MovieLensData

class GraphFeatures:
    def __init__(self, df_ratings, alpha_coef):
        """
        Initialize with ratings data and alpha coefficient for similarity graph.
        """
        self.df_ratings = df_ratings
        self.alpha_coef = alpha_coef

    def generate_similarity_graph(self):
        """
        Generate a similarity graph based on user interactions.
        """
        pairs = []
        grouped = self.df_ratings.groupby(['MID', 'rate'])
        for key, group in grouped:
            pairs.extend(list(itertools.combinations(group['UID'], 2)))
        counter = collections.Counter(pairs)
        alpha = self.alpha_coef * 1682
        edge_list = map(list, collections.Counter(el for el in counter.elements() if counter[el] >= alpha).keys())
        G = nx.Graph()
        for el in edge_list:
            G.add_edge(el[0], el[1], weight=1)
            G.add_edge(el[0], el[0], weight=1)
            G.add_edge(el[1], el[1], weight=1)
        return G

    def extract_graph_features(self, G, df_user):
        """
        Extract graph features like PageRank, Degree Centrality, etc.
        """
        pr = nx.pagerank(G.to_directed())
        df_user['PR'] = df_user['UID'].map(pr)
        df_user['PR'] /= float(df_user['PR'].max())

        dc = nx.degree_centrality(G)
        df_user['CD'] = df_user['UID'].map(dc)
        df_user['CD'] /= float(df_user['CD'].max())

        cc = nx.closeness_centrality(G)
        df_user['CC'] = df_user['UID'].map(cc)
        df_user['CC'] /= float(df_user['CC'].max())

        bc = nx.betweenness_centrality(G)
        df_user['CB'] = df_user['UID'].map(bc)
        df_user['CB'] /= float(df_user['CB'].max())

        lc = nx.load_centrality(G)
        df_user['LC'] = df_user['UID'].map(lc)
        df_user['LC'] /= float(df_user['LC'].max())

        nd = nx.average_neighbor_degree(G, weight='weight')
        df_user['AND'] = df_user['UID'].map(nd)
        df_user['AND'] /= float(df_user['AND'].max())

        return df_user
    

if __name__ == '__main__':
    data_path = 'Datasets/100k/ml-100k/'
    ml_data = MovieLensData(data_path)
    df_user = ml_data.preprocess_user_data()
    df_ratings = ml_data.df_ratings
    alpha_coef = ml_data.alpha_coef
    graph_features = GraphFeatures(df_ratings, alpha_coef)
    G = graph_features.generate_similarity_graph()
    df_user = graph_features.extract_graph_features(G, df_user)
    print(df_user.head())
    print(df_ratings.head())
    print(alpha_coef)