import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from utils import load_image, predict_octmnist
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline

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

# Initialize the summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


class TextRequest(BaseModel):
    text: str


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


# def preprocess_text(text: str) -> str:
#     """
#     Preprocess the input text by removing unnecessary whitespace and normalizing.
#     """
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text)
#     # Remove leading and trailing whitespace
#     text = text.strip()
#     return text


@app.post("/summarize")
async def summarize_text(request: TextRequest):
    text = request.text

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Generate summary using the pipeline
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        summary_text = summary[0]["summary_text"]
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail="Error generating summary")

    return {"summary": summary_text}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
