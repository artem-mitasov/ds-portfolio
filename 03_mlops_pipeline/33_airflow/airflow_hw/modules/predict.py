import dill
import json
import pandas as pd
from pathlib import Path

# Пути к папкам
MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
TEST_DIR = Path(__file__).parent.parent / "data" / "test"
PREDICTIONS_DIR = Path(__file__).parent.parent / "data" / "predictions"


def load_model() -> object:
    """Загружает последний обученный pipeline через dill"""
    model_files = sorted(MODEL_DIR.glob("*.pkl"))
    if not model_files:
        raise FileNotFoundError("Нет файлов модели в папке data/models")
    latest_model = model_files[-1]
    with open(latest_model, "rb") as f:
        model = dill.load(f)
    return model


def predict():
    """Основная функция предсказания"""
    model = load_model()
    all_predictions = []

    for file_path in sorted(TEST_DIR.glob("*.json")):
        # Читаем JSON
        with open(file_path, "r") as f:
            data = json.load(f)
        # Конвертируем в DataFrame с одной строкой
        df = pd.DataFrame([data])
        # Делаем предсказание
        preds = model.predict(df)
        # Сохраняем вместе с исходными данными
        df_with_pred = df.copy()
        df_with_pred["prediction"] = preds
        all_predictions.append(df_with_pred)

    if all_predictions:
        final_df = pd.concat(all_predictions, ignore_index=True)
        PREDICTIONS_DIR.mkdir(exist_ok=True)
        output_file = PREDICTIONS_DIR / "predictions.csv"
        final_df.to_csv(output_file, index=False)
        print(f"Предсказания сохранены: {output_file}")
    else:
        print("Нет предсказаний для сохранения.")


if __name__ == "__main__":
    predict()