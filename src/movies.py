import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Dataset path
ratings_path = os.path.join('..', 'resources', 'dataset', 'ratings.csv')
movies_path = os.path.join('..', 'resources', 'dataset', 'movies_metadata.csv')

# Model path
model_path = os.path.join('..', 'resources', 'model', 'trained_model.pkl')


def load_datasets():
    print("Start loading datasets for training.")
    movie_dataset = pd.read_csv(movies_path, low_memory=False)
    ratings = pd.read_csv(ratings_path)
    ratings = ratings.dropna()
    movie_dataset = movie_dataset.dropna()
    ratings['movieId'] = ratings['movieId'].astype(int)
    print("Finished loading datasets for training.")
    return movie_dataset, ratings


def load_model():
    if os.path.exists(model_path):
        print(f"Movies model exists loading model from file: {model_path}")
        with open(model_path, 'rb') as model_file:
            movie_model = pickle.load(model_file)

        movie_data = pd.read_csv(movies_path, low_memory=False)
        movie_data = movie_data.dropna()
        print(f"AI movie_data model loaded from file: {model_path}")
    else:
        print(f"Loading datasets: {model_path}")
        movie_data, ratings = load_datasets()
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

        movie_model = SVD()
        cross_validate(movie_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

        train_set = data.build_full_trainset()
        movie_model.fit(train_set)

        with open('trained_model.pkl', 'wb') as model_file:
            pickle.dump(movie_model, model_file)

    return movie_data, movie_model


def load_metadata(movie_id, movies):
    movie = movies[movies['id'] == str(movie_id)].iloc[0]
    return f"Title: {movie['title']}, Genres: {movie['genres']}, Rating: {movie['vote_average']}"


def recommendations(user_id, model, movies, n_recommendations=5):
    predictions = [model.predict(user_id, int(movie_id)) for movie_id in movies['id'].astype(int)]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n = predictions[:n_recommendations]
    movie_recommendations = [load_metadata(pred.iid, movies) for pred in top_n]

    return movie_recommendations

