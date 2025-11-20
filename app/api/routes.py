from fastapi import APIRouter
from app.api.schemas import PredictRequest, PredictResponse
from app.models.loader import get_model
import torch

router = APIRouter()
model = get_model()  # load once at module level

@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    # Convert input list to tensor (adjust shape as needed)
    input_tensor = torch.tensor([request.data], dtype=torch.float32)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    return PredictResponse(prediction=output_tensor.tolist()[0])
