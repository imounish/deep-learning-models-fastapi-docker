# Deploying DL Models as a Web Server

## Table of Contents

- [What is FastAPI](#what-is-fastapi)
  - [Features](#features)
  - [Installation](#installation)
  - [Tutorial](#tutorial)
- [How we used FastAPI to deploy DL models](#how-we-used-fastapi-to-deploy-dl-models)
  - [Setting up the server](#setting-up-the-server)
  - [Image Classification Model](#image-classification-model)
  - [Document Summarization Models](#document-summarization-models)
  - [Deployment](#deployment)

# How we used FastAPI to deploy DL models

We used FastAPI as a REST API server that takes takes in the input to a DL model and sends back the prediction after inferencing.

## Setting up the server

## Image Classification Model

- Docker installed on your system
- A trained model file (`model.pth`) (this is already included in the repository)

## Document Summarization Models

## Deployment

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
