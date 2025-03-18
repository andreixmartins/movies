import os

from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from movies import load_model, recommendations
from typing import Union

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import Request

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Movies Recommender"})


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
