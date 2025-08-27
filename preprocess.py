import numpy as np
import load_data  # nosso módulo que já carrega os CSVs com Path

def preprocess(train, test):
    X_train = train.drop("label", axis=1).values
    y_train = train["label"].values
    X_test = test.values

    # Normalizar pixels para [0,1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Redimensionar para CNN: (n amostras, 28, 28, 1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    return X_train, y_train, X_test


if __name__ == "__main__":
    # Chama o load_data que já está preparado para endereçar certo
    train, test = load_data.load_data()
    X_train, y_train, X_test = preprocess(train, test)

    # Conferência
    print("X_train shape:", X_train.shape)
    print("y_train sample:", y_train[:5])
