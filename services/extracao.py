import os
from PIL import Image
import numpy as np
import cv2

UPLOAD_FOLDER = 'uploads'
ICONS_FOLDER = 'static/icons'

def processar_extracao(files):
    try:
        arquivos1 = files.getlist('personagem1')
        arquivos2 = files.getlist('personagem2')

        if not arquivos1 or not arquivos2:
            return False, "As duas pastas precisam ser enviadas!"

        salvar_arquivos(arquivos1, 'personagem_1')
        salvar_arquivos(arquivos2, 'personagem_2')

        gerar_imagens_com_icones()
        return True, "Imagens extraídas e salvas com sucesso!"

    except Exception as e:
        return False, f"Erro na extração: {str(e)}"

def salvar_arquivos(arquivos, nome_pasta):
    caminho = os.path.join(UPLOAD_FOLDER, nome_pasta)
    os.makedirs(caminho, exist_ok=True)
    
    for arquivo in arquivos:
        if arquivo and arquivo.filename:
            arquivo.save(os.path.join(caminho, arquivo.filename))

def gerar_imagens_com_icones():
    cores = {
        "vermelho": (255, 0, 0),
        "verde": (0, 255, 0),
        "azul": (0, 0, 255),
        "amarelo": (255, 255, 0),
        "roxo": (128, 0, 128),
        "laranja": (255, 165, 0),
    }

    for cor_nome, cor_rgb in cores.items():
        imagem = np.zeros((32, 32, 3), dtype=np.uint8)
        imagem[:] = cor_rgb

        caminho_icone = os.path.join(ICONS_FOLDER, f"{cor_nome}.png")
        if not os.path.exists(caminho_icone):
            continue

        icone = cv2.imread(caminho_icone, cv2.IMREAD_UNCHANGED)
        icone = cv2.resize(icone, (32, 32))

        for c in range(3):  # mescla RGB
            imagem[:, :, c] = np.where(icone[:, :, 3] > 0, icone[:, :, c], imagem[:, :, c])

        imagem_pil = Image.fromarray(imagem)
        imagem_pil.save(os.path.join(UPLOAD_FOLDER, f"{cor_nome}.png"))
