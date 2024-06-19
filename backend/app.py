import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import (
    load_image,
    load_img_classification_model,
    predict_img_labels,
    load_tokenizer,
    load_t5_model,
    load_bart_pipeline,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# Allow all origins for CORS
origins = ["*"]

# Set up CORS middleware to allow all origins, credentials, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up basic logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Install logger
logger = logging.getLogger(__name__)


# Request Class for the text input
class TextRequest(BaseModel):
    text: str


# Load the models when the server first starts
img_classification_model = load_img_classification_model()
t5_model = load_t5_model()
bart_summarizer = (
    load_bart_pipeline()
)  # loading bart model would take up time as the model is large


@app.get("/")
def root() -> dict:
    """
    Root endpoint returning a welcome message.

    Returns:
        dict: A dictionary containing a welcome message.
    """
    return {"message": "Welcome to the ML Model API"}


@app.post("/predict/image_classes")
async def predict_img_caption(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint to predict classes for each image among on the 1000 ImageNet classes.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: A JSON response containing the predicted class label.

    Raises:
        HTTPException: If any error occurs during processing.
    """
    try:
        # Load image from the uploaded file
        inputs = load_image(await file.read())

        # Load the model here or during the first start of the server
        # img_classification_model = load_img_classification_model()

        # Predict the class label for the input image
        predicted_class_label = predict_img_labels(inputs, img_classification_model)

    except Exception as exception:
        logger.error(exception)
        raise HTTPException(status_code=400, detail=str(exception))

    return JSONResponse(content={"class_label": predicted_class_label})


@app.post("/predict/text_summarize")
async def t5_prediction(request: TextRequest, model: str = "t5") -> JSONResponse:
    """
    Endpoint to summarize text using the specified model.

    Args:
        request (TextRequest): The request body containing the text to summarize.
        model (str): The model to use for summarization (default is "t5").

    Returns:
        JSONResponse: A JSON response containing the summary.

    Raises:
        HTTPException: If any error occurs during processing.
    """
    try:
        # Raise an exception if the posted string is empty
        if not request.text.strip():
            raise HTTPException(status_code=400, detail=str("Text cannot be empty"))

        if model == "t5":
            # preprocess the input text
            sequence = request.text
            tokenizer = load_tokenizer()
            inputs = tokenizer.encode(
                "summarize: " + sequence,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
            # Load the model here or during the first start of the server
            # t5_model = load_t5_model()

            # Generate summary using the T5 model
            output = t5_model.generate(inputs, min_length=80, max_length=100)
            summary = tokenizer.decode(output[0])

        elif model == "bart":
            # raise HTTPException(
            #     status_code=400, detail=str("We do not support BART model yet.")
            # )

            # Load the model here or during the first start of the server
            bart_summarizer = (
                load_bart_pipeline()
            )  # this would time out as the model is too large

            output = bart_summarizer(
                request.text, max_length=130, min_length=30, do_sample=False
            )
            summary = output[0]["summary_text"]

        else:
            raise HTTPException(status_code=400, detail=str("Invalid model specified."))

    except Exception as exception:
        raise HTTPException(status_code=400, detail=str(exception))

    return JSONResponse(content={"summary": summary})
