import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

class MovieLensData:
    def __init__(self, data_path, mode, split_ratio=0.2):
        """
        Initialize the class with data paths and load data.
        """
        self.split_ratio = split_ratio
        self.data_path = data_path
        if mode =='100k':
            self.df_ratings = pd.read_csv(self.data_path + 'u.data', sep='\t', header=None, names=['UID', 'MID', 'rate', 'time'])
            self.df_user = pd.read_csv(self.data_path + 'u.user', sep='|', header=None, names=['UID', 'age', 'gender', 'job', 'zip'])
            self.df_item = pd.read_csv(self.data_path + 'u.item', sep='|', header=None, encoding='latin-1', names=[
                'item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 
                'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ])
            self.alpha_coef = 0.03
        elif mode == '1m':
            self.df_ratings = pd.read_csv(self.data_path + 'ratings.dat', sep='::', engine='python', names=['UID', 'MID', 'rate', 'time'])
            self.df_user = pd.read_csv(self.data_path + 'users.dat', sep='::', engine='python', names=['UID', 'gender', 'age', 'job', 'zip'])
            self.df_item = pd.read_csv(self.data_path + 'movies.dat', sep='::', engine='python', header=None, names=['MID', 'title', 'genres'], encoding='latin-1')
            self.alpha_coef = 0.045
        else:
            raise ValueError("Invalid mode. Choose from '100k' or '1m'.")
        

    def convert_categorical(self, df, column):
        """
        Convert categorical features into one-hot encoded format.
        """
        values = np.array(df[column])
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        onehot_encoder = OneHotEncoder(sparse_output=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        df = df.drop(column, axis=1)
        for j in range(integer_encoded.max() + 1):
            df.insert(loc=j + 1, column=str(column) + str(j + 1), value=onehot_encoded[:, j])
        return df

    def preprocess_user_data(self):
        """
        Preprocess user data by converting categorical features and scaling.
        """
        self.df_user = self.convert_categorical(self.df_user, 'job')
        self.df_user = self.convert_categorical(self.df_user, 'gender')
        self.df_user['bin'] = pd.cut(self.df_user['age'], [0, 10, 20, 30, 40, 50, 100], labels=['1', '2', '3', '4', '5', '6'])
        self.df_user['age'] = self.df_user['bin']
        self.df_user = self.df_user.drop('bin', axis=1)
        self.df_user = self.convert_categorical(self.df_user, 'age')
        self.df_user = self.df_user.drop('zip', axis=1)
        return self.df_user

    
    def split_data(self):
        """
        Split the data into training, validation, and test sets.
        """
        train, test = train_test_split(self.df_ratings, test_size=self.split_ratio, random_state=42)
        train, val = train_test_split(train, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
        return train, val, test

    

if __name__ == '__main__':
    data_path = 'Datasets/100k/ml-100k/'
    ml_data = MovieLensData(data_path)
    ml_data.preprocess_user_data()
    print(ml_data.df_user.head())
    print(ml_data.df_ratings.head())
    print(ml_data.alpha_coef)