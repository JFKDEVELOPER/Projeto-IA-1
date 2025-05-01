import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
MODEL_PATH = os.path.join(MODEL_FOLDER, 'mlp_model.h5')

def treinar_mlp():
    try:
        dados, rotulos = carregar_dados()
        if len(dados) == 0:
            return False, "Nenhuma imagem encontrada para treinamento."

        dados = np.array(dados)
        rotulos = np.array(rotulos)
        rotulos = to_categorical(rotulos)

        modelo = Sequential([
            Flatten(input_shape=(32, 32, 3)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(rotulos.shape[1], activation='softmax')
        ])

        modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        modelo.fit(dados, rotulos, epochs=10, batch_size=8, verbose=0)

        os.makedirs(MODEL_FOLDER, exist_ok=True)
        modelo.save(MODEL_PATH)

        return True, "Treinamento da MLP concluído com sucesso!"

    except Exception as e:
        return False, f"Erro no treinamento: {str(e)}"

def carregar_dados():
    dados = []
    rotulos = []
    cores = {
        "vermelho": 0,
        "verde": 1,
        "azul": 2,
        "amarelo": 3,
        "roxo": 4,
        "laranja": 5
    }

    for cor_nome, rotulo in cores.items():
        caminho = os.path.join(UPLOAD_FOLDER, f"{cor_nome}.png")
        if os.path.exists(caminho):
            imagem = Image.open(caminho).resize((32, 32))
            imagem_array = np.array(imagem) / 255.0
            dados.append(imagem_array)
            rotulos.append(rotulo)

    return dados, rotulos
