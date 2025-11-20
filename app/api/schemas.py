from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    data: List[float]

class PredictResponse(BaseModel):
    prediction: List[float]