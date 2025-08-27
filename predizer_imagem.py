import tensorflow as tf
import carregar_imagem  # nosso arquivo de carregar imagem

def predizer_digito(caminho_imagem, caminho_modelo="digit_recognize_model.h5"):
    # Carrega a imagem e prepara
    img_array = carregar_imagem.carregar_imagem_externa(caminho_imagem)
    
    # Carrega o modelo treinado
    model = tf.keras.models.load_model(caminho_modelo)
    
    # Faz predição
    resultado = model.predict(img_array)
    
    # Escolhe o dígito com maior probabilidade
    digito_predito = resultado.argmax()
    
    return digito_predito

if __name__ == "__main__":
    caminho_img = r"C:\Users\tb89857\Downloads\terceira_imagem.png"  # informe aqui sua imagem
    digito = predizer_digito(caminho_img)
    print(f"Dígito previsto para a imagem '{caminho_img}': {digito}")