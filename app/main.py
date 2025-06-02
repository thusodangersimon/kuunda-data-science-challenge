from fastapi import FastAPI, HTTPException, status

from .models import TrainRequest, TrainResponse, PredictRequest, PredictResponse, ExplainResponse,CustomerLoanRecord
from .preprocessing import get_preprocessor, FeatureSelectorByNames
from .train_model_main import main as train_model
from .utils import save_artifacts, load_artifacts, get_shap_explainer, calculate_shap_values, MODEL_DIR, \
    PREPROCESSOR_FILE, MODEL_FILE, FEATURE_NAMES_FILE, SHAP_BACKGROUND_FILE

app = FastAPI(
    title="ML Model API with SHAP Explanations",
    description="A simple API for training a Credid scoring model, making predictions, and generating SHAP explanations.",
    version="1.0.0"
)

#model = load_artifacts()


@app.post("/train", response_model=TrainResponse, summary="Train the ML model")
async def train_model(request: TrainRequest):
    train_model(request)


@app.post("/predict", response_model=PredictResponse, summary="Get predictions from the trained model")
async def predict(request: CustomerLoanRecord):
    return model.predict(request)


@app.post("/explain", response_model=ExplainResponse, summary="Get predictions and SHAP explanations")
async def explain_prediction(request: PredictRequest):
    raise NotImplementedError
