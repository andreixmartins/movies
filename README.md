

## Movies Recommendation System

## Requirements

- Python 3.8+

- Remove numpy 2.* and install numpy 1.26.0
```shell
pip install numpy
```

- Install numpy 1.26.0
```shell
pip install numpy==1.26.0
```

- Install requirements
```shell
pip install -r requirements.txt
```

- Install FastAPI
```shell
pip install "fastapi[standard]"
```


### Running FastAPI server

```shell
fastapi dev main.py
```

### Recomenndation API
```shell
curl -L http://127.0.0.1:8000/user/1
```

### Open API Docs
- http://127.0.0.1:8000/docs


### References
- https://fastapi.tiangolo.com/
- https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
- https://grouplens.org/datasets/movielens/
