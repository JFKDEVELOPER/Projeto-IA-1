import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
import csv
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)

# Caminho absoluto para garantir que a pasta de uploads fique dentro do diretório principal do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Caminho para o diretório onde o app.py está
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')  # Pasta 'uploads' dentro do diretório do projeto

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pasta para salvar as imagens renomeadas como ícones
ICONS_FOLDER = os.path.join(BASE_DIR, 'static', 'icons')
os.makedirs(ICONS_FOLDER, exist_ok=True)

# Caminho onde o CSV será salvo dentro da pasta de uploads
RESULTADO_CSV = os.path.join(UPLOAD_FOLDER, 'resultado_extracao.csv')

@app.route('/cnn')
def cnn():
    # A lógica do seu endpoint CNN vai aqui
    return "Aqui está o CNN!"

@app.route('/classificar_imagem', methods=['GET', 'POST'])
def classificar_imagem():
    if request.method == 'POST':
        try:
            imagem = request.files['imagem']

            if imagem.filename == '':
                return "Nenhuma imagem enviada."

            caminho_imagem = os.path.join(app.config['UPLOAD_FOLDER'], imagem.filename)
            imagem.save(caminho_imagem)

            modelo_path = os.path.join(app.config['UPLOAD_FOLDER'], 'modelo_cnn.h5')
            if not os.path.exists(modelo_path):
                return "Modelo treinado não encontrado. Treine primeiro."

            model = load_model(modelo_path)

            img = Image.open(caminho_imagem).convert('RGB').resize((64, 64))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predicao = model.predict(img_array)
            classe_predita = np.argmax(predicao)

            return f"Classe prevista para a imagem: {classe_predita}"

        except Exception as e:
            return f"Erro ao classificar a imagem: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extracao_pixel', methods=['GET', 'POST'])
def extracao_pixel():
    if request.method == 'POST':
        # Recebe as pastas enviadas como arquivos (não pastas de fato)
        arquivos = request.files.getlist('folders')

        if len(arquivos) < 2:
            return "Erro: Ambas as pastas precisam ser enviadas."

        # Diretórios onde as imagens serão salvas
        pasta_personagem1 = os.path.join(app.config['UPLOAD_FOLDER'], 'personagem_1')
        pasta_personagem2 = os.path.join(app.config['UPLOAD_FOLDER'], 'personagem_2')

        os.makedirs(pasta_personagem1, exist_ok=True)
        os.makedirs(pasta_personagem2, exist_ok=True)

        # A primeira pasta (arquivo) vai para personagem_1
        pasta_1 = arquivos[0]
        if pasta_1 and hasattr(pasta_1, 'filename') and pasta_1.filename != '':
            # Salva todos os arquivos dessa pasta em 'personagem_1'
            for file in pasta_1:
                filename = os.path.basename(file.filename)
                try:
                    file.save(os.path.join(pasta_personagem1, filename))
                except Exception as e:
                    return f"Erro ao salvar o arquivo {filename}: {str(e)}"

        # A segunda pasta (arquivo) vai para personagem_2
        pasta_2 = arquivos[1]
        if pasta_2 and hasattr(pasta_2, 'filename') and pasta_2.filename != '':
            # Salva todos os arquivos dessa pasta em 'personagem_2'
            for file in pasta_2:
                filename = os.path.basename(file.filename)
                try:
                    file.save(os.path.join(pasta_personagem2, filename))
                except Exception as e:
                    return f"Erro ao salvar o arquivo {filename}: {str(e)}"

        # Após salvar as imagens, renomeia as imagens de ícones
        renomear_icons()

        # Se tudo der certo, redireciona para a página de atributos
        return redirect(url_for('definir_atributos'))

    return render_template('extracao_pixel.html')


def renomear_icons():
    pasta_personagem1 = os.path.join(app.config['UPLOAD_FOLDER'], 'personagem_1')
    pasta_personagem2 = os.path.join(app.config['UPLOAD_FOLDER'], 'personagem_2')

    if os.path.exists(pasta_personagem1):
        imagens_p1 = os.listdir(pasta_personagem1)
        if imagens_p1:
            imagem_personagem1 = imagens_p1[0]  # Pega a primeira imagem
            caminho_imagem_p1 = os.path.join(pasta_personagem1, imagem_personagem1)
            imagem_p1 = Image.open(caminho_imagem_p1)
            imagem_p1.save(os.path.join(ICONS_FOLDER, 'icon_personagem_1.png'))  # Salva na pasta static/icons

    if os.path.exists(pasta_personagem2):
        imagens_p2 = os.listdir(pasta_personagem2)
        if imagens_p2:
            imagem_personagem2 = imagens_p2[0]  # Pega a primeira imagem
            caminho_imagem_p2 = os.path.join(pasta_personagem2, imagem_personagem2)
            imagem_p2 = Image.open(caminho_imagem_p2)
            imagem_p2.save(os.path.join(ICONS_FOLDER, 'icon_personagem_2.png'))  # Salva na pasta static/icons

@app.route('/definir_atributos', methods=['GET', 'POST'])
def definir_atributos():
    if request.method == 'POST':
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

        # Após definir os atributos, processa as imagens
        X_train, X_test, y_train, y_test = gerar_csv(atributos_com_intervalo)

        # Após gerar o CSV e a divisão dos dados, cria e treina o modelo
        modelo = criar_modelo(X_train.shape[1])
        modelo.fit(X_train, y_train, epochs=10, batch_size=32)

        # Avaliar o modelo
        acuracia = modelo.evaluate(X_test, y_test)
        print(f"Acurácia do modelo: {acuracia[1]*100}%")

        # Após gerar o CSV, disponibiliza para download
        if not os.path.exists(RESULTADO_CSV):
            return "Erro: O arquivo CSV não foi gerado corretamente."
        
        return send_file(RESULTADO_CSV, as_attachment=True)

    return render_template('definir_atributos.html')

def verificar_pixel(pixel, intervalo):
    r, g, b = pixel
    return (intervalo['R_min'] <= r <= intervalo['R_max'] and
            intervalo['G_min'] <= g <= intervalo['G_max'] and
            intervalo['B_min'] <= b <= intervalo['B_max'])

def gerar_csv(atributos_com_intervalo):
    dados_csv = []
    pastas = ['personagem_1', 'personagem_2']

    X = []  # Atributos
    y = []  # Classes

    for personagem in pastas:
        pasta_completa = os.path.join(UPLOAD_FOLDER, personagem)
        if not os.path.exists(pasta_completa):
            continue

        for nome_arquivo in os.listdir(pasta_completa):
            caminho_arquivo = os.path.join(pasta_completa, nome_arquivo)

            # Verifica se o arquivo é uma imagem
            try:
                imagem = Image.open(caminho_arquivo)
                imagem.verify()  # Verifica se é uma imagem válida
                imagem = Image.open(caminho_arquivo).convert('RGB')  # Reabre para processamento
            except (IOError, SyntaxError):
                print(f"Arquivo ignorado (não é imagem válida): {nome_arquivo}")
                continue

            largura, altura = imagem.size
            pixels = imagem.load()

            contagem_p1 = 0
            contagem_p2 = 0

            for x in range(largura):
                for y in range(altura):
                    pixel = pixels[x, y]

                    if any(verificar_pixel(pixel, intervalo) for intervalo in atributos_com_intervalo['personagem1']):
                        contagem_p1 += 1
                    elif any(verificar_pixel(pixel, intervalo) for intervalo in atributos_com_intervalo['personagem2']):
                        contagem_p2 += 1

            dados_csv.append([nome_arquivo, 'personagem_1', contagem_p1])
            dados_csv.append([nome_arquivo, 'personagem_2', contagem_p2])

            # Adicionar os dados para o modelo
            X.append([contagem_p1, contagem_p2])  # Aqui você pode adicionar outras características, se necessário
            y.append(0 if personagem == 'personagem_1' else 1)

    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Salvar o CSV
    try:
        with open(RESULTADO_CSV, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Arquivo', 'Classe', 'Total de Pixels'])
            writer.writerows(dados_csv)
    except Exception as e:
        print(f"Erro ao salvar o CSV: {e}")

    return X_train, X_test, y_train, y_test

def criar_modelo(input_dim):
    modelo = Sequential()
    modelo.add(Dense(64, input_dim=input_dim, activation='relu'))
    modelo.add(Dense(32, activation='relu'))
    modelo.add(Dense(1, activation='sigmoid'))  # Saída binária (0 ou 1)

    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

if __name__ == '__main__':
    app.run(debug=True)
