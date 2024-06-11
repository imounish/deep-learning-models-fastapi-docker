import os
import logging
from fastapi import FastAPI, File, UploadFile
from utils import load_image, predict_octmnist
from fastapi.responses import JSONResponse

app = FastAPI()

# basic logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# install logger
logger = logging.getLogger(__name__)

# set the model path
model_path = "model/best_model_OCTMNIST.pth"


@app.get("/")
def root():
    return {"message": "Welcome to the ML Model API"}


@app.post("/predict")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    image_array = load_image(file.filename)

    prediction = predict_octmnist(image_array, model_path)

    os.remove(file.filename)

    return JSONResponse(content=prediction)
