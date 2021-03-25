from fastapi import FastAPI
from .predictor import Predictor

app = FastAPI()
predictor = Predictor("./models/best.pth")


@app.get("/")
def home():
    return "Try to use /predict?url=url_to_image " \
           "with url_to_image you want to classify"


@app.get("/predict")
def predict(url: str):
    return {"predict": predictor.predict(url)}
