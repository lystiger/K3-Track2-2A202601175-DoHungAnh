from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tải file model.pkl từ GCS về máy khi server khởi động.

    Hàm này được gọi một lần khi module được import. Sử dụng
    GOOGLE_APPLICATION_CREDENTIALS để xác thực (được đặt trong systemd service).
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if GCS_BUCKET:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print(f"Model đã được tải xuống từ GCS: gs://{GCS_BUCKET}/{GCS_MODEL_KEY} -> {MODEL_PATH}")
    else:
        # Hỗ trợ chạy local / test khi chưa cấu hình GCS_BUCKET
        local_model = "models/model.pkl"
        if os.path.exists(local_model) and not os.path.exists(MODEL_PATH):
            import shutil
            shutil.copy(local_model, MODEL_PATH)
            print(f"Sử dụng mô hình cục bộ từ {local_model}")


try:
    download_model()
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
except Exception as e:
    print(f"Khởi tạo mô hình: {e}")
    model = None


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiểm tra sức khỏe server.
    GitHub Actions gọi endpoint này sau khi deploy để xác nhận server đang chạy.

    Trả về: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận chính.

    Đầu vào : JSON {"features": [f1, f2, ..., f12]}
    Đầu ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thứ tự 12 đặc trưng (khớp với thứ tự trong FEATURE_NAMES của test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected 12 features (wine quality)"
        )

    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=500, detail="Model chưa được tải hoặc không tồn tại.")

    pred = int(model.predict([req.features])[0])
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = label_map.get(pred, "unknown")

    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
