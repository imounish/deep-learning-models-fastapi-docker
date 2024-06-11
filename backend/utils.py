from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from model.CNN import CNN

octmnist_classes = {
    "0": "choroidal neovascularization",
    "1": "diabetic macular edema",
    "2": "drusen",
    "3": "normal",
}


def load_image(infilename: str) -> np.ndarray:
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def load_model(model_path: str) -> CNN:
    model = CNN(1, 4)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))["model"]
    )
    return model


def predict_octmnist(image_array: np.ndarray, model_path: str):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    input = transform(image_array)
    input = input.unsqueeze(0)
    input = input.type(torch.float32)

    model = load_model(model_path)

    model.eval()
    with torch.no_grad():
        output = model(input)
        pred = torch.argmax(output, dim=1).item()
    return {"class": pred, "label": octmnist_classes[str(pred)]}
