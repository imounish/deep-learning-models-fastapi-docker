FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN mkdir -p ~/.cache/huggingface

RUN cp -R model/models--Salesforce--blip-image-captioning-base ~/.cache/huggingface/

RUN pip install fastapi uvicorn joblib numpy scikit-learn torch torchvision python-multipart pillow transformers timm einops

# RUN pip install flash-attn --no-build-isolation

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]