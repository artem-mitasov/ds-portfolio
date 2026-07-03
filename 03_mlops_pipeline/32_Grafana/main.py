from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import dill


app = FastAPI()

with open('model/cars_pipe.pkl', 'rb') as f:
    model_data = dill.load(f)

model = model_data['model']
metadata = model_data['metadata']


class CarData(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: float
    manufacturer: str
    image_url: str
    description: str
    posting_date: str
    lat: float
    long: float
    fuel: str
    transmission: str
    model: str
    year: int
    odometer: int
    state: str
    title_status: str


class Prediction(BaseModel):
    id: int
    price: float
    pred: str


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return metadata


@app.post('/prediction', response_model=Prediction)
def predict(car: CarData):
    df = pd.DataFrame([car.dict()])
    pred = model.predict(df)[0]

    return Prediction(
        id=car.id,
        price=car.price,
        pred=pred
    )
