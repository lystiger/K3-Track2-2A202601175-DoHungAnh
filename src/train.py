import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params     : dict chứa các siêu tham số cho mô hình (RandomForestClassifier, GradientBoostingClassifier, LogisticRegression).
        data_path  : đường dẫn đến file dữ liệu huấn luyện.
        eval_path  : đường dẫn đến file dữ liệu đánh giá.

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá.
    """

    # Đọc dữ liệu huấn luyện và đánh giá
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Bonus 5: Kiểm tra phân phối dữ liệu huấn luyện và cảnh báo lệch lạc dữ liệu
    class_counts = df_train["target"].value_counts(normalize=True).to_dict()
    class_distribution = {int(k): float(v) for k, v in sorted(class_counts.items())}
    for cls, ratio in class_distribution.items():
        if ratio < 0.10:
            print(f"[CẢNH BÁO LỆCH DỮ LIỆU - Bonus 5]: Lớp {cls} chỉ chiếm {ratio:.2%}, nhỏ hơn 10% tổng số mẫu!")

    # Tách đặc trưng (X) và nhãn (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Cấu hình MLflow backend URI mặc định nếu chưa được set từ môi trường (hỗ trợ DagsHub - Bonus 1)
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run():
        # Ghi nhận các siêu tham số vào MLflow
        mlflow.log_params(params)

        # Bonus 2: Hỗ trợ đa thuật toán (RandomForest, GradientBoosting, LogisticRegression)
        model_type = params.get("model_type", "random_forest")
        clf_params = {k: v for k, v in params.items() if k != "model_type"}

        if model_type == "random_forest":
            model = RandomForestClassifier(**clf_params, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**clf_params, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**clf_params, random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Không hỗ trợ model_type: {model_type}")

        # Huấn luyện mô hình
        model.fit(X_train, y_train)

        # Dự đoán trên tập đánh giá
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # Ghi nhận các chỉ số vào MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        for cls, ratio in class_distribution.items():
            mlflow.log_metric(f"class_{cls}_ratio", ratio)

        # Log mô hình vào MLflow artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"Model: {model_type} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Bonus 3: Tự động tạo báo cáo hiệu suất chi tiết (Confusion Matrix & Classification Report)
        os.makedirs("outputs", exist_ok=True)
        cm = confusion_matrix(y_eval, preds)
        unique_labels = sorted(list(set(y_eval) | set(preds)))
        target_names = [f"Class {i}" for i in unique_labels]
        cr = classification_report(y_eval, preds, labels=unique_labels, target_names=target_names, zero_division=0)

        report_content = f"""==================================================
BÁO CÁO ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH (MLOps Report)
==================================================
Thuật toán: {model_type}
Siêu tham số: {json.dumps(params, indent=2)}

Chỉ số tổng thể:
----------------
Accuracy : {acc:.4f}
Weighted F1-Score: {f1:.4f}

Phân phối nhãn tập Train (Bonus 5):
----------------------------------
{json.dumps(class_distribution, indent=2)}

Ma trận nhầm lẫn (Confusion Matrix - Bonus 3):
---------------------------------------------
{cm}

Báo cáo phân loại chi tiết theo từng lớp (Bonus 3):
--------------------------------------------------
{cr}
==================================================
"""
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write(report_content)

        try:
            mlflow.log_artifact("outputs/report.txt")
        except Exception as e:
            print(f"Không thể log artifact report.txt vào MLflow: {e}")

        # Lưu metrics ra file outputs/metrics.json để GitHub Actions đọc
        metrics_data = {
            "accuracy": acc,
            "f1_score": f1,
            "model_type": model_type,
            "class_distribution": class_distribution,
        }
        with open("outputs/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        # Lưu mô hình ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
