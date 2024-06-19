from PIL import Image
from io import BytesIO
from typing import Dict, Any
from transformers import (
    SegformerImageProcessor,
    SegformerForImageClassification,
    AutoTokenizer,
    AutoModelWithLMHead,
    pipeline,
)


def load_image(data: bytes) -> Dict[str, Any]:
    """
    Load an image from bytes and preprocess it for model input.

    Args:
        data (bytes): The image data in bytes.

    Returns:
        Dict[str, Any]: A dictionary containing the preprocessed image tensor.
    """
    # Load the Segformer image processor
    image_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    # Open the image and convert it to RGB
    raw_image = Image.open(BytesIO(data)).convert("RGB")
    # Preprocess the image and return the tensor
    inputs = image_processor(images=raw_image, return_tensors="pt")

    return inputs


def load_img_classification_model() -> SegformerForImageClassification:
    """
    Load the Segformer image classification model.

    Returns:
        SegformerForImageClassification: The loaded Segformer image classification model.
    """
    return SegformerForImageClassification.from_pretrained("nvidia/mit-b0")


def predict_img_labels(
    inputs: Dict[str, Any], model: SegformerForImageClassification
) -> str:
    """
    Predict the class label for the given input image using the specified model.

    Args:
        inputs (Dict[str, Any]): The input image tensor.
        model (SegformerForImageClassification): The image classification model.

    Returns:
        str: The predicted class label.
    """
    # Get the model outputs and Extract the logits
    outputs = model(**inputs)
    logits = outputs.logits

    # Get the index of the predicted class
    predicted_class_index = logits.argmax(-1).item()

    # Return the class label corresponding to the predicted class index
    return model.config.id2label[predicted_class_index].split(", ")[0]


def load_tokenizer() -> AutoTokenizer:
    """
    Load the tokenizer for T5 model.

    Returns:
        AutoTokenizer: The loaded T5 tokenizer.
    """
    return AutoTokenizer.from_pretrained("T5-small")


def load_t5_model() -> AutoModelWithLMHead:
    """
    Load the T5 model for text generation.

    Returns:
        AutoModelWithLMHead: The loaded T5 model.
    """
    return AutoModelWithLMHead.from_pretrained("T5-small", return_dict=True)


def load_bart_pipeline() -> pipeline:
    """
    Load the BART pipeline for text summarization.

    Returns:
        pipeline: The loaded BART summarization pipeline.
    """
    return pipeline("summarization", model="facebook/bart-large-cnn")
