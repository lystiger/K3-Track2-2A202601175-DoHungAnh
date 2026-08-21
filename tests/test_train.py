import os
import json
import numpy as np
import pandas as pd
from src.train import train


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tạo dataset nhỏ với cùng schema Wine Quality để sử dụng trong test.

    pytest cung cấp `tmp_path` là một thư mục tạm thời, tự động xóa sau khi test kết thúc.
    Hàm này dùng dữ liệu ngẫu nhiên nên không cần kết nối GCS hay tải file CSV thực.
    """
    rng = np.random.default_rng(0)
    n = 200

    # Tạo mảng X có kích thước (n, len(FEATURE_NAMES)) với giá trị [0, 1)
    X = rng.random((n, len(FEATURE_NAMES)))

    # Tạo mảng y gồm n phần tử nguyên ngẫu nhiên trong [0, 3)
    y = rng.integers(0, 3, size=n)

    # Xây dựng DataFrame, thêm cột "target"
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    # Lưu 160 dòng đầu làm tập huấn luyện, 40 dòng cuối làm tập đánh giá
    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiểm tra hàm train() trả về một số thực nằm trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)

    acc = train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(acc, (float, np.floating))
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """Kiểm tra file outputs/metrics.json được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics


def test_model_file_created(tmp_path):
    """Kiểm tra file models/model.pkl được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.pkl")


def test_multiple_algorithms(tmp_path):
    """Kiểm tra hỗ trợ nhiều thuật toán khác nhau (Bonus 2)."""
    train_path, eval_path = _make_temp_data(tmp_path)

    # Gradient Boosting
    acc_gb = train(
        {"model_type": "gradient_boosting", "n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(acc_gb, (float, np.floating))
    assert 0.0 <= acc_gb <= 1.0

    # Logistic Regression
    acc_lr = train(
        {"model_type": "logistic_regression", "C": 1.0},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(acc_lr, (float, np.floating))
    assert 0.0 <= acc_lr <= 1.0


def test_serve_endpoints():
    """Kiểm tra các endpoints của FastAPI serving (GET /health và POST /predict)."""
    from fastapi.testclient import TestClient
    from src.serve import app

    client = TestClient(app)

    # Test /health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}

    # Test /predict valid
    sample_features = [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0.0]
    res_pred = client.post("/predict", json={"features": sample_features})
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1, 2]
    assert data["label"] in ["thap", "trung_binh", "cao"]

    # Test /predict invalid feature length
    res_bad = client.post("/predict", json={"features": [1.0, 2.0]})
    assert res_bad.status_code == 400


