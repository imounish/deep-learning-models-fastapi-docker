FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn joblib numpy scikit-learn torch torchvision medmnist python-multipart pillow transformers

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]