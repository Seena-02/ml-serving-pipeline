from pydantic import BaseModel


class PredictRequest(BaseModel):
    data: list[float]


class PredictResponse(BaseModel):
    prediction: list[float]
