import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'uploads'

#services/classificacao.py – apenas classifica a imagem

def classificar_imagem(imagem):
    try:
        caminho = os.path.join(UPLOAD_FOLDER, imagem.filename)
        imagem.save(caminho)

        modelo = load_model(os.path.join(UPLOAD_FOLDER, 'modelo_pixel.h5'))

        # processamento e predição
        return f"Classe prevista: Personagem X"
    except Exception as e:
        return f"Erro: {str(e)}"
