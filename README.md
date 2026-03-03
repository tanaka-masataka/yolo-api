# YOLOv8 FastAPI Docker API

A lightweight object detection API using:
- FastAPI
- Ultralytics YOLOv8
- PyTorch (CPU)
- Docker

## Run with Docker

```bash
docker build -t yolo-api .
docker run -p 8000:8000 yolo-api

