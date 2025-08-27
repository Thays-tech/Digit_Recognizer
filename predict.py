# predict.py
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import sys

# Garante que a pasta do arquivo esteja no sys.path para achar preprocess/load_data
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import preprocess
import load_data


def predict_and_save_submission(model_path: str | None = None):
    # Define o caminho do modelo relativo à pasta deste arquivo
    if model_path is None:
        # Preferir o formato novo .keras; se não existir, cair para .h5
        candidates = [BASE_DIR / "digit_recognize_model.keras",
                      BASE_DIR / "digit_recognize_model.h5"]
        found = next((p for p in candidates if p.exists()), None)
        if not found:
            raise FileNotFoundError(
                "Modelo não encontrado. Procurei em:\n"
                + "\n".join(str(p) for p in candidates)
                + "\nDica: rode o script de treino para gerar o arquivo, ou passe o caminho do modelo em model_path."
            )
        model_path = found

    print(f"[INFO] Carregando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Carregar e pré-processar dados
    train, test = load_data.load_data()
    _, _, X_test = preprocess.preprocess(train, test)

    # Prever
    print("[INFO] Fazendo predições...")
    predictions = model.predict(X_test, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    # Salvar submission ao lado do script
    out_csv = BASE_DIR / "submission.csv"
    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(predicted_classes) + 1),
        "Label": predicted_classes
    })
    submission.to_csv(out_csv, index=False)
    print(f"✅ Arquivo criado: {out_csv}")


if __name__ == "__main__":
    # Se quiser forçar um caminho específico, passe aqui, ex.:
    # predict_and_save_submission(r"C:\caminho\para\digit_recognize_model.keras")
    predict_and_save_submission()
