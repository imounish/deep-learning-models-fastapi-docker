import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import load_image, load_img_classification_model, predict_img_labels
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline

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

# Initialize the summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


class TextRequest(BaseModel):
    text: str


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
