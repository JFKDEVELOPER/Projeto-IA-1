# services/classificacao_cnn.py

import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODELO_PATH = os.path.join('models', 'modelo_cnn.h5')
LABELS_PATH = os.path.join('models', 'labels_map.json')

def classificar_imagem_cnn(imagem_file):
    try:
        # Carregar o modelo treinado
        modelo = load_model(MODELO_PATH)

        # Carregar o mapeamento de rótulos
        with open(LABELS_PATH, 'r') as f:
            labels_map = json.load(f)

        # Inverter o dicionário: índice → nome da classe
        labels_map_inv = {int(v): k for k, v in labels_map.items()}

        # Processar imagem
        img = Image.open(imagem_file).convert('RGB').resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Fazer predição
        predicao = modelo.predict(img_array)
        classe_idx = int(np.argmax(predicao))

        # DEBUG opcional (remova se não quiser no terminal):
        print("Probabilidades:", predicao)
        print("Classe índice prevista:", classe_idx)
        print("Labels invertido:", labels_map_inv)

        classe_nome = labels_map_inv.get(classe_idx, "Classe desconhecida")

        return f"Classe prevista: {classe_nome}"

    except Exception as e:
        return f"Erro ao classificar imagem: {str(e)}"
