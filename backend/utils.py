from PIL import Image
from io import BytesIO
from typing import Tuple
from transformers import SegformerImageProcessor, SegformerForImageClassification


def load_image(data):
    image_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    raw_image = Image.open(BytesIO(data)).convert("RGB")
    inputs = image_processor(images=raw_image, return_tensors="pt")

    return inputs


def load_img_classification_model():
    return SegformerForImageClassification.from_pretrained("nvidia/mit-b0")


def predict_img_labels(inputs, model):

    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_index = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_index].split(", ")[0]
