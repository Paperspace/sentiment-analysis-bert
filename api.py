from typing import Dict
import uvicorn
from fastapi import Depends, FastAPI
from pydantic import BaseModel
import os
from model import Model, get_model

#app = FastAPI()
app = FastAPI(openapi_prefix='/model-serving/'+os.getenv("HOSTNAME").split('-')[0])

class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str
    confidence: float

class HeartbeatResult(BaseModel):
    is_alive: bool

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: Model = Depends(get_model)):
    sentiment, confidence, probabilities = model.predict(request.text)
    return SentimentResponse(
        sentiment=sentiment, confidence=confidence, probabilities=probabilities
    )

@app.get("/", response_model=HeartbeatResult)
def get_heartbeat()-> HeartbeatResult:
    heartbeat = HeartbeatResult(is_alive=True)
    return heartbeat

if __name__ == "__main__":
    uvicorn.run(app, log_level="info")