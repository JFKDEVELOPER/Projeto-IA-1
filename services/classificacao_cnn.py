# services/classificacao_cnn.py

import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import os

def classificar_imagem_cnn(imagem_file):
    # Carrega o modelo treinado
    modelo = load_model('models/modelo_cnn.h5')

    # Processa a imagem recebida
    imagem = Image.open(imagem_file).convert('RGB').resize((64, 64))
    imagem_array = np.array(imagem) / 255.0
    imagem_array = np.expand_dims(imagem_array, axis=0)

    # Faz a previsão
    predicao = modelo.predict(imagem_array)
    classe = np.argmax(predicao)

    return f'Classe prevista: {classe}'
