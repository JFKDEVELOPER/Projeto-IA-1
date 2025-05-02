# services/classificacao_cnn.py

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json
import os

MODELOS_FOLDER = os.path.join('models')

def classificar_imagem_cnn(imagem_file):
    try:
        modelo_path = os.path.join(MODELOS_FOLDER, 'modelo_cnn.h5')
        labels_path = os.path.join(MODELOS_FOLDER, 'labels_map.json')

        if not os.path.exists(modelo_path) or not os.path.exists(labels_path):
            return "Modelo ou mapeamento de classes não encontrado."

        model = load_model(modelo_path)
        with open(labels_path, 'r') as f:
            labels_map = json.load(f)

        imagem = Image.open(imagem_file).convert('RGB').resize((64, 64))
        imagem_array = np.array(imagem) / 255.0
        imagem_array = np.expand_dims(imagem_array, axis=0)

        predicao = model.predict(imagem_array)
        classe_idx = int(np.argmax(predicao))
        nome_classe = labels_map.get(str(classe_idx), "Classe desconhecida")

        return f"Classe prevista: {nome_classe}"
    except Exception as e:
        return f"Erro ao classificar a imagem: {str(e)}"
