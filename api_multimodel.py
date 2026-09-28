import datetime
import logging
import time
import uvicorn
from fastapi import FastAPI
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto

from predict_multimodel import predict
from utils import validate_response

HOST = '0.0.0.0'
PORT = 8000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
start_time = time.time()

@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    response = predict(request)
    validate_response(response)
    rv = response.requested_view
    logger.info('[V3 MULTIMODEL] frame %s L%s -> det: %s | req L%s',
                request.frame_index, request.view.resolution_level,
                len(response.annotations), rv.resolution_level if rv else '?')
    return response

@app.get('/')
def index(): return "API Multi-Model running!"

if __name__ == '__main__':
    uvicorn.run('api_multimodel:app', host=HOST, port=PORT)
