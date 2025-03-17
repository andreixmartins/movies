from movies import load_model, recommendations

if __name__ == '__main__':
    user_id = 20
    movies, model = load_model()
    movie_recommendations = recommendations(user_id, model, movies)
    for rec in movie_recommendations:
        print(rec)
