import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import load_image, load_img_classification_model, predict_img_labels
from fastapi.responses import JSONResponse

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# basic logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Install logger
logger = logging.getLogger(__name__)

# Set the model path
model_path = "model/best_model_OCTMNIST.pth"


@app.get("/")
def root():
    return {"message": "Welcome to the ML Model API"}


@app.post("/predict/image_classes")
async def predict_img_caption(file: UploadFile = File(...)):
    try:
        inputs = load_image(await file.read())

        model = load_img_classification_model()

        predicted_class_label = predict_img_labels(inputs, model)

    except Exception as exception:
        raise HTTPException(status_code=400, detail=str(exception))

    return JSONResponse(content={"class_label": predicted_class_label})
