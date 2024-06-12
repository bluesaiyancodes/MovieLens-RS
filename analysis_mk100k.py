# %% [markdown]
# ## Run Analysis on 100K Movie Lens Data

# %%
# Imports
import numpy as np
import pandas as pd
import os

# Plotting 
import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# %%

# Load data
data_path_100k = 'Datasets/100k/ml-100k/'
data_path_1m = 'Datasets/1m/ml-1m/'

# Choose the dataset - '100k' or '1m'
dataType = '100k'



if dataType == '100k':
    datapath = data_path_100k
else:
    datapath = data_path_1m

plotpath = f'Plots/{dataType}/'
os.makedirs('Plots', exist_ok=True)
os.makedirs(plotpath, exist_ok=True)


# %%
df_user = pd.read_csv(os.path.join(datapath, "u.user"), sep="|", header=None, names=["user_id", "age", "gender", "occupation", "zipcode"])
print(df_user.head())

col_rating = ['user_id','item_id','rating','timestamp']
df_rating = pd.read_csv(os.path.join(datapath, "u.data"), sep="\t", header=None, names=col_rating)
df_rating.drop('timestamp', axis=1, inplace=True)
print(df_rating.head())

col_item = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
df_item = pd.read_csv(os.path.join(datapath, "u.item"), sep="|", header=None, names=col_item, encoding='latin-1')
df_item.drop('video_release_date', axis=1, inplace=True)
print(df_item.head())

# %%
print(f"No. unique users: {len(df_user['user_id'].unique())}")

# %%
sns.histplot(df_user['age'], kde=True)
plt.title("Age Distribution")
plt.savefig(os.path.join(plotpath, "age_distribution.png"))

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

# Plot for 'gender'
sns.countplot(x='gender', data=df_user, palette='viridis', ax=axes[0])
axes[0].set_title('Distribution of Gender')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Count')

# Plot for 'occupation'
sns.countplot(x='occupation', data=df_user, palette='dark', ax=axes[1])
axes[1].set_title('Distribution of Occupations')
axes[1].set_xlabel('Occupation')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

# Adjust layout
plt.tight_layout()
plt.savefig(os.path.join(plotpath, "gender_occupation_distribution.png"))

# %%
occupation_counts = df_user['occupation'].value_counts()

# Plot a pie chart with improved readability
plt.figure(figsize=(12, 8))
plt.pie(occupation_counts, labels=occupation_counts.index, rotatelabels=True, autopct='%1.1f%%', labeldistance=1.0, startangle=30, colors=sns.color_palette('pastel'))

# Add a legend and adjust layout
plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.tight_layout()
plt.title('Occupation Distribution')
plt.savefig(os.path.join(plotpath, "occupation_distribution.png"))

# %% [markdown]
# ### Additional Analyses and Visualizations

# %%
# Rating distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_rating['rating'], bins=5, kde=False)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig(os.path.join(plotpath, "rating_distribution.png"))

# %%
# Ratings per movie
ratings_per_movie = df_rating['item_id'].value_counts()
plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_movie, bins=50, kde=False)
plt.title('Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Count of Movies')
plt.yscale('log')
plt.savefig(os.path.join(plotpath, "ratings_per_movie.png"))

# %%
# Ratings per user
ratings_per_user = df_rating['user_id'].value_counts()
plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_user, bins=50, kde=False)
plt.title('Number of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Count of Users')
plt.yscale('log')
plt.savefig(os.path.join(plotpath, "ratings_per_user.png"))

# %%
# Top-N movies by average rating
N = 10
top_n_movies = df_rating.groupby('item_id').agg({'rating': ['mean', 'count']})
top_n_movies.columns = ['average_rating', 'rating_count']
top_n_movies = top_n_movies[top_n_movies['rating_count'] >= 50]  # Filter out movies with fewer ratings
top_n_movies = top_n_movies.sort_values('average_rating', ascending=False).head(N)
top_n_movies = top_n_movies.merge(df_item[['item_id', 'movie_title']], on='item_id')

plt.figure(figsize=(10, 6))
sns.barplot(x='average_rating', y='movie_title', data=top_n_movies, palette='viridis')
plt.title(f'Top {N} Movies by Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Movie Title')
plt.savefig(os.path.join(plotpath, "top_n_movies.png"))

# %%
# Heatmap of user-item interactions
user_item_matrix = df_rating.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

plt.figure(figsize=(15, 10))
sns.heatmap(user_item_matrix, cmap='viridis', cbar=False)
plt.title('Heatmap of User-Item Interactions')
plt.xlabel('Movies')
plt.ylabel('Users')
plt.savefig(os.path.join(plotpath, "user_item_heatmap.png"))

# %%
# Correlation between different genres
genre_columns = df_item.columns[5:]
df_genres = df_item[genre_columns].astype(int)
genre_corr = df_genres.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(genre_corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Movie Genres')
plt.savefig(os.path.join(plotpath, "genre_correlation_matrix.png"))
