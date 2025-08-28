# 📖 Digit Recognizer (MNIST / Kaggle)

Este projeto implementa um pipeline simples e modular para **treinar um classificador de dígitos manuscritos** utilizando o dataset **MNIST** (disponível no Kaggle).  

O fluxo contempla:  
- 📥 Carregamento dos dados  
- ⚙️ Pré-processamento  
- 🤖 Treinamento do modelo  
- 📊 Geração do arquivo de submissão  
- 🖼️ Predição de imagens externas  

---

## 📂 Estrutura do Projeto
```bash
digit-recognizer/
├── load_data.py
├── preprocess.py
├── train_model.py
├── predict.py
├── predizer_imagem.py
├── carregar_imagem.py
├── main.py
├── train.csv
├── test.csv
└── (gerados: digit_recognize_model.h5/.keras, submission.csv)
```

---

## 🔧 Requisitos
- Python **3.10+**  
- Bibliotecas:  
  ```bash
  pip install pandas numpy tensorflow pillow
  ```
- Compatível com Windows / Linux / macOS  

---

## ▶️ Execução

### Treinar modelo completo
```bash
python main.py
```

### Gerar arquivo de submissão (modelo já treinado)
```bash
python predict.py
```

### Prever dígito a partir de imagem externa
```bash
python predizer_imagem.py
```

---

## ⚠️ Erros Comuns

- `FileNotFoundError`: `train.csv` ou `test.csv` não encontrados.  
- Modelo não encontrado: execute `main.py` antes de rodar `predict.py`.  
- `OSError` em imagens: verifique caminho e formato.  
- Shapes incompatíveis: ajuste `input_shape` no modelo.  

---

## ❓ FAQ

**Posso usar GPU?**  
✔️ Sim, basta instalar o TensorFlow compatível.  

**Posso salvar em `.keras`?**  
✔️ Sim, é o formato preferido atualmente.  

**Como aumentar a acurácia?**  
📈 Ajuste número de épocas, use *callbacks* e *data augmentation*.  

---

## 📜 Licença
Este projeto é de uso educacional e segue a licença padrão do Kaggle para datasets.
