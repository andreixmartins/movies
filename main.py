import os

from movies import load_model, recommendations
from typing import Union

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html=True), name="static")


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Movies recommendation": "1.0"}


@app.get("/user/{user_id}")
def user_movies(user_id: int):
    movies = recommendation_movies(user_id)
    return movies


def recommendation_movies(user_id):
    ratings_path = os.path.join('resources', 'dataset', 'ratings.csv')
    movies_path = os.path.join('resources', 'dataset', 'movies_metadata.csv')
    model_path = os.path.join('resources', 'model', 'trained_model.pkl')

    movies, model = load_model(ratings_path, movies_path, model_path)
    return recommendations(user_id, model, movies)
