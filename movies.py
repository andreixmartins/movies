import os
import json
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Dataset path
ratings_path = os.path.join('..', 'resources', 'dataset', 'ratings.csv')
movies_path = os.path.join('..', 'resources', 'dataset', 'movies_metadata.csv')

# Model path
model_path = os.path.join('..', 'resources', 'model', 'trained_model.pkl')


def load_datasets(ratings_dataset=ratings_path, movie_dataset=movies_path):
    print("Start loading datasets for training.")
    movie_data = pd.read_csv(movie_dataset, low_memory=False)
    movie_data = movie_data.dropna()

    ratings_data = pd.read_csv(ratings_dataset)
    ratings_data = ratings_data.dropna()

    ratings_data['movieId'] = ratings_data['movieId'].astype(int)
    print("Finishing loading datasets for training.")
    return movie_data, ratings_data


def load_model(ratings_dataset, movie_dataset, trained_model_path):
    # Load datasets
    movie_data, ratings = load_datasets(ratings_dataset, movie_dataset)

    # Check if the model file exists (already trained):
    if os.path.exists(trained_model_path):
        with open(trained_model_path, 'rb') as model_file:
            movie_model = pickle.load(model_file)
        return movie_data, movie_model
    else:
        # Train the model
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
    # return movie
    # collection = json.loads(movie['belongs_to_collection'])
    data = {
        'title': movie['title'],
        'collection': movie['belongs_to_collection'],
        'rating': movie['vote_average'],
    }
    return json.dumps(data)


def recommendations(user_id, model, movies, n_recommendations=5):
    predictions = [model.predict(user_id, int(movie_id)) for movie_id in movies['id'].astype(int)]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n = predictions[:n_recommendations]
    movie_recommendations = [load_metadata(pred.iid, movies) for pred in top_n]
    return movie_recommendations


if __name__ == '__main__':
    user_id = 20
    movies, model = load_model(ratings_path, movies_path, model_path)
    movie_recommendations = recommendations(user_id, model, movies)
    for rec in movie_recommendations:
        print(rec)