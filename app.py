from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schema.prediction_request import PredictionRequest
from schema.prediction_response import PredictionResponse
from model import prediction as pred

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pred.get_model()
    except Exception as e:
        pred.model_load_error = str(e)
    yield

app = FastAPI(lifespan=lifespan)

@app.get('/')
def home():
    return {'message':'Customer Support Ticket Dispatcher'}

@app.get('/health')
def health_check():
    return {
        'status': 'OK',
        'version': pred.MODEL_VERSION,
        'model_loaded': pred.model is not None,
        'model_error': pred.model_load_error
    }

@app.post('/predict', response_model=PredictionResponse, status_code=200)
def predict_premium(request: PredictionRequest):
    try:
        return pred.predict_output([request.email])
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")