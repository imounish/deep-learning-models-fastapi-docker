FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn pillow torch transformers sentencepiece

# RUN pip install flash-attn --no-build-isolation

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]