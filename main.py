import load_data
import preprocess
import train_model
import predict

def main():
    print("Carregando dados...")
    train, test = load_data.load_data()

    print("Pré-processando dados...")
    X_train, y_train, X_test = preprocess.preprocess(train, test)

    print("Treinando modelo...")
    model = train_model.train_model(X_train, y_train)
    model.save("digit_recognize_model.h5")

    print("Fazendo predição e salvando submissão...")
    predict.predict_and_save_submission("digit_recognize_model.h5")

if __name__ == "__main__":
    main()