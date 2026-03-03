from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

model = YOLO("yolov8n.pt")
@app.get("/")
def read_root():
    return {"message": "YOLO API is running!"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # 画像データを読み込む
    contents = await file.read()

    # バイト列 → numpy 配列に変換
    nparr = np.frombuffer(contents, np.uint8)

    # numpy 配列 → OpenCV 画像に変換
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                "class": model.names[cls],
                "confidence": conf
            })
    return {"detections" : detections}