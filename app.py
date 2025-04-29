import os
import json
import csv
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
from PIL import Image
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
import tensorflow as tf

app = Flask(__name__)

# Configurações de pastas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ICONS_FOLDER = os.path.join(BASE_DIR, 'static', 'icons')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ICONS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

RESULTADO_CSV = os.path.join(UPLOAD_FOLDER, 'resultado_extracao.csv')

# --- ROTAS --- #

@app.route('/')
def home():
    return render_template('index.html')

# --- PIXEL: Upload das pastas --- #
@app.route('/extracao_pixel', methods=['GET', 'POST'])
def extracao_pixel():
    if request.method == 'POST':
        arquivos = request.files.getlist('folders')
        if len(arquivos) < 2:
            return "Erro: Ambas as pastas precisam ser enviadas."

        pasta_personagem1 = os.path.join(UPLOAD_FOLDER, 'personagem_1')
        pasta_personagem2 = os.path.join(UPLOAD_FOLDER, 'personagem_2')
        os.makedirs(pasta_personagem1, exist_ok=True)
        os.makedirs(pasta_personagem2, exist_ok=True)

        arquivos[0].save(os.path.join(pasta_personagem1, arquivos[0].filename))
        arquivos[1].save(os.path.join(pasta_personagem2, arquivos[1].filename))

        renomear_icons()

        return redirect(url_for('definir_atributos'))

    return render_template('extracao_pixel.html')

def renomear_icons():
    pasta_personagem1 = os.path.join(UPLOAD_FOLDER, 'personagem_1')
    pasta_personagem2 = os.path.join(UPLOAD_FOLDER, 'personagem_2')

    if os.path.exists(pasta_personagem1):
        imagens_p1 = os.listdir(pasta_personagem1)
        if imagens_p1:
            imagem_p1 = Image.open(os.path.join(pasta_personagem1, imagens_p1[0]))
            imagem_p1.save(os.path.join(ICONS_FOLDER, 'icon_personagem_1.png'))

    if os.path.exists(pasta_personagem2):
        imagens_p2 = os.listdir(pasta_personagem2)
        if imagens_p2:
            imagem_p2 = Image.open(os.path.join(pasta_personagem2, imagens_p2[0]))
            imagem_p2.save(os.path.join(ICONS_FOLDER, 'icon_personagem_2.png'))

# --- PIXEL: Definir atributos e treinar MLP --- #
@app.route('/definir_atributos', methods=['GET', 'POST'])
def definir_atributos():
    if request.method == 'POST':
        try:
            atributos = {
                'personagem1': [
                    request.form['personagem1_cor1'],
                    request.form['personagem1_cor2'],
                    request.form['personagem1_cor3']
                ],
                'personagem2': [
                    request.form['personagem2_cor1'],
                    request.form['personagem2_cor2'],
                    request.form['personagem2_cor3']
                ]
            }

            atributos_com_intervalo = processar_cores(atributos)
            X_train, X_test, y_train, y_test = gerar_csv(atributos_com_intervalo)

            modelo = criar_modelo_mlp(X_train.shape[1])
            modelo.fit(X_train, y_train, epochs=10, batch_size=32)

            modelo.save(os.path.join(UPLOAD_FOLDER, 'modelo_pixel.h5'))

            acuracia = modelo.evaluate(X_test, y_test)
            return f"Rede por Pixel treinada com sucesso! Acurácia: {acuracia[1]*100:.2f}%"

        except Exception as e:
            return f"Erro ao processar atributos: {str(e)}"

    return render_template('definir_atributos.html')

def processar_cores(atributos):
    tolerancia = 20
    atributos_com_intervalo = {}
    for personagem, cores in atributos.items():
        atributos_com_intervalo[personagem] = []
        for cor in cores:
            r = int(cor[1:3], 16)
            g = int(cor[3:5], 16)
            b = int(cor[5:7], 16)
            intervalo = {
                'R_min': max(0, r - tolerancia),
                'R_max': min(255, r + tolerancia),
                'G_min': max(0, g - tolerancia),
                'G_max': min(255, g + tolerancia),
                'B_min': max(0, b - tolerancia),
                'B_max': min(255, b + tolerancia)
            }
            atributos_com_intervalo[personagem].append(intervalo)
    return atributos_com_intervalo

def gerar_csv(atributos_com_intervalo):
    dados_csv = []
    pastas = ['personagem_1', 'personagem_2']

    X, y = [], []

    for personagem in pastas:
        pasta_completa = os.path.join(UPLOAD_FOLDER, personagem)
        if not os.path.exists(pasta_completa):
            continue

        for nome_arquivo in os.listdir(pasta_completa):
            caminho_arquivo = os.path.join(pasta_completa, nome_arquivo)
            imagem = Image.open(caminho_arquivo).convert('RGB')
            largura, altura = imagem.size
            pixels = imagem.load()

            contagem_p1 = 0
            contagem_p2 = 0

            for x in range(largura):
                for y_ in range(altura):
                    pixel = pixels[x, y_]
                    if any(verificar_pixel(pixel, intervalo) for intervalo in atributos_com_intervalo['personagem1']):
                        contagem_p1 += 1
                    elif any(verificar_pixel(pixel, intervalo) for intervalo in atributos_com_intervalo['personagem2']):
                        contagem_p2 += 1

            dados_csv.append([nome_arquivo, personagem, contagem_p1, contagem_p2])

            X.append([contagem_p1, contagem_p2])
            y.append(0 if personagem == 'personagem_1' else 1)

    with open(RESULTADO_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Arquivo', 'Classe', 'Total_Personagem1', 'Total_Personagem2'])
        writer.writerows(dados_csv)

    return train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)

def verificar_pixel(pixel, intervalo):
    r, g, b = pixel
    return (intervalo['R_min'] <= r <= intervalo['R_max'] and
            intervalo['G_min'] <= g <= intervalo['G_max'] and
            intervalo['B_min'] <= b <= intervalo['B_max'])

def criar_modelo_mlp(input_dim):
    modelo = Sequential()
    modelo.add(Dense(64, input_dim=input_dim, activation='relu'))
    modelo.add(Dense(32, activation='relu'))
    modelo.add(Dense(1, activation='sigmoid'))
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

# --- PIXEL: Classificar nova imagem --- #
@app.route('/classificar_imagem_pixel', methods=['GET', 'POST'])
def classificar_imagem_pixel():
    if request.method == 'POST':
        try:
            imagem = request.files['imagem']
            if imagem.filename == '':
                return "Nenhuma imagem enviada."

            caminho_imagem = os.path.join(UPLOAD_FOLDER, imagem.filename)
            imagem.save(caminho_imagem)

            modelo = load_model(os.path.join(UPLOAD_FOLDER, 'modelo_pixel.h5'))

            img = Image.open(caminho_imagem).convert('RGB')
            largura, altura = img.size
            pixels = img.load()

            contagem_p1 = 0
            contagem_p2 = 0

            for x in range(largura):
                for y in range(altura):
                    r, g, b = pixels[x, y]
                    if r > g and r > b:
                        contagem_p1 += 1
                    else:
                        contagem_p2 += 1

            entrada = np.array([[contagem_p1, contagem_p2]])
            predicao = modelo.predict(entrada)
            classe_predita = 'Personagem 1' if predicao >= 0.5 else 'Personagem 2'

            return f"Classe prevista para a imagem: {classe_predita}"

        except Exception as e:
            return f"Erro ao classificar imagem: {str(e)}"

    return render_template('classificar_imagem_pixel.html')

# --- CNN: Treinar CNN --- #
@app.route('/cnn', methods=['GET', 'POST'])
def cnn():
    if request.method == 'POST':
        try:
            folders = request.files.getlist('folders')
            camadas = int(request.form['camadas'])
            neuronios = int(request.form['neuronios'])
            epocas = int(request.form['epocas'])

            pasta_cnn = os.path.join(UPLOAD_FOLDER, 'cnn')
            os.makedirs(pasta_cnn, exist_ok=True)

            imagens = []
            labels = []

            for arquivo in folders:
                if arquivo.filename == '':
                    continue
                caminho_completo = os.path.join(pasta_cnn, arquivo.filename)
                arquivo.save(caminho_completo)

                nome_classe = arquivo.filename.split('/')[0] if '/' in arquivo.filename else arquivo.filename.split('\\')[0]
                labels.append(nome_classe)

                img = Image.open(caminho_completo).convert('RGB').resize((64, 64))
                img_array = np.array(img)
                imagens.append(img_array)

            imagens = np.array(imagens) / 255.0
            labels_unicos = list(set(labels))
            labels_map = {nome: idx for idx, nome in enumerate(labels_unicos)}
            labels_num = np.array([labels_map[label] for label in labels])

            X_train, X_test, y_train, y_test = train_test_split(imagens, labels_num, test_size=0.2, random_state=42)

            model = Sequential()
            for i in range(camadas):
                if i == 0:
                    model.add(Conv2D(neuronios, (3, 3), activation='relu', input_shape=(64, 64, 3)))
                else:
                    model.add(Conv2D(neuronios, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(neuronios, activation='relu'))
            model.add(Dense(len(labels_unicos), activation='softmax'))

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(X_train, y_train, epochs=epocas, validation_data=(X_test, y_test))

            model.save(os.path.join(UPLOAD_FOLDER, 'modelo_cnn.h5'))
            with open(os.path.join(UPLOAD_FOLDER, 'labels_map.json'), 'w') as f:
                json.dump(labels_map, f)

            acuracia = history.history['val_accuracy'][-1] * 100
            return f"Modelo CNN treinado com sucesso! Acurácia: {acuracia:.2f}%"

        except Exception as e:
            return f"Erro durante o treinamento: {str(e)}"

    return render_template('cnn.html')

# --- CNN: Classificar nova imagem --- #
@app.route('/classificar_imagem', methods=['GET', 'POST'])
def classificar_imagem():
    if request.method == 'POST':
        try:
            imagem = request.files['imagem']
            if imagem.filename == '':
                return "Nenhuma imagem enviada."

            caminho_imagem = os.path.join(UPLOAD_FOLDER, imagem.filename)
            imagem.save(caminho_imagem)

            modelo_path = os.path.join(UPLOAD_FOLDER, 'modelo_cnn.h5')
            labels_path = os.path.join(UPLOAD_FOLDER, 'labels_map.json')

            if not os.path.exists(modelo_path) or not os.path.exists(labels_path):
                return "Modelo treinado não encontrado."

            model = load_model(modelo_path)
            with open(labels_path, 'r') as f:
                labels_map = json.load(f)
            labels_map_invertido = {v: k for k, v in labels_map.items()}

            img = Image.open(caminho_imagem).convert('RGB').resize((64, 64))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predicao = model.predict(img_array)
            classe_predita_idx = np.argmax(predicao)
            nome_classe_predita = labels_map_invertido.get(str(classe_predita_idx), "Classe desconhecida")

            return f"Classe prevista para a imagem: {nome_classe_predita}"

        except Exception as e:
            return f"Erro ao classificar a imagem: {str(e)}"

    return render_template('classificar_imagem.html')

# --- Rodar o servidor --- #
if __name__ == '__main__':
    app.run(debug=True, port=5001)
