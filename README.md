# OCTMNIST Dataset Prediction API

This repository contains a Dockerized FastAPI web server for predicting OCTMNIST images. The server takes an OCTMNIST image as input (only `28 x 28`) and returns the predicted class.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Running the Docker Container](#running-the-docker-container)
  - [Making Predictions](#making-predictions)
- [Creating an Image from the OCTMNIST Dataset](#creating-an-image-from-the-octmnist-dataset)

## Installation

### Prerequisites

- Docker installed on your system
- A trained model file (`model.pth`) (this is already included in the repository)

### Steps

1. Clone the repository or open the code in a new directory:

2. Build the Docker image by changing your current directory into `backend` directory:

   ```sh
   docker build -t octmnist-prediction-api .
   ```

3. Run the Docker container:

   ```sh
   docker run -d -p 8000:8000 --name octmnist-api octmnist-prediction-api
   ```

You can also build and run the container in one step using `docker-compose`. You need to change your current directory to outside `backend`.

```sh
docker-compose up --build
```

## Usage

### Running the Docker Container

After following the installation steps, the Docker container should be running and the FastAPI server should be accessible at `http://localhost:8000`.

### Making Predictions

To make a prediction, send a POST request to the `/predict` endpoint with the image file. You can use `curl` or any API client like Postman.

#### Example using `curl`:

```sh
curl -X POST "http://localhost:8000/predict" -F "file=@path_to_your_image.png"
```

#### Example Response

The API will return a JSON response with the predicted class and label.

```json
{
  "class": 2,
  "label": "drusen"
}
```

## Creating an Image from the OCTMNIST Dataset

#### Prerequisites

- Python installed on your system
- Required Python libraries: PIL and numpy

#### Step-by-Step Guide

1. Install the required libraries:

If you don't have the necessary libraries installed, you can install them using pip:

```sh
pip install pillow numpy
```

2. Download the OCTMNIST dataset:

You can download the OCTMNIST dataset from the official MedMNIST [website](https://medmnist.com/).

3. Load the dataset:

Load the dataset using the appropriate method.

```python
from numpy import load
data = load('octmnist.npz')
```

4. Define the function to save the numpy array as an image

```python
from PIL import Image
import numpy as np

def save_image(npdata, filename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(filename)
```

5. You can save the numpy array as an image:

```python
save_image(data["test_images"][0], "sample_octmnist_image.png")
```

Here, I used the first test image from the dataset.
