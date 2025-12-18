from fastapi import APIRouter, HTTPException
from app.api.schemas import PredictRequest, PredictResponse
from app.models.loader import get_model
import torch

router = APIRouter()
model = get_model()

EXPECTED_INPUT_SIZE = 784  # 28 * 28


@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    # Validate input size
    if len(request.data) != EXPECTED_INPUT_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input size: expected {EXPECTED_INPUT_SIZE}, got {len(request.data)}"
        )

    # Convert input list to tensor and reshape for MNIST (1, 1, 28, 28)
    input_tensor = torch.tensor(request.data, dtype=torch.float32).view(1, 1, 28, 28)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    return PredictResponse(prediction=output_tensor.tolist()[0])