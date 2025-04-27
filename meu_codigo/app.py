import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import csv
from PIL import Image

app = Flask(__name__)

# Caminho absoluto para garantir que a pasta de uploads fique dentro do diretório principal do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Caminho para o diretório onde o app.py está
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')  # Pasta 'uploads' dentro do diretório do projeto

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Caminho onde o CSV será salvo dentro da pasta de uploads
RESULTADO_CSV = os.path.join(UPLOAD_FOLDER, 'resultado_extracao.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extracao_pixel', methods=['GET', 'POST'])
def extracao_pixel():
    if request.method == 'POST':
        folders = request.files.getlist('folders')

        if not folders:
            return "Erro: Nenhum arquivo foi enviado."

        pasta_personagem1 = os.path.join(app.config['UPLOAD_FOLDER'], 'personagem_1')
        pasta_personagem2 = os.path.join(app.config['UPLOAD_FOLDER'], 'personagem_2')

        os.makedirs(pasta_personagem1, exist_ok=True)
        os.makedirs(pasta_personagem2, exist_ok=True)

        for i, folder in enumerate(folders):
            if folder.filename == '':
                continue

            filename = os.path.basename(folder.filename)

            # Arquivos enviados para a primeira pasta (personagem_1)
            if i < len(folders) // 2:
                destino = pasta_personagem1
            # Arquivos enviados para a segunda pasta (personagem_2)
            else:
                destino = pasta_personagem2

            try:
                folder.save(os.path.join(destino, filename))
            except Exception as e:
                return f"Erro ao salvar o arquivo {filename}: {str(e)}"

        # Se tudo der certo, redireciona para a página de atributos
        return redirect(url_for('definir_atributos'))

    return render_template('extracao_pixel.html')

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
        gerar_csv(atributos_com_intervalo)

        # Após gerar o CSV, disponibiliza para download
        if not os.path.exists(RESULTADO_CSV):
            return "Erro: O arquivo CSV não foi gerado corretamente."
        
        return send_file(RESULTADO_CSV, as_attachment=True)

    return render_template('definir_atributos.html')

@app.route('/cnn')
def cnn():
    return render_template('cnn.html')

def verificar_pixel(pixel, intervalo):
    r, g, b = pixel
    return (intervalo['R_min'] <= r <= intervalo['R_max'] and
            intervalo['G_min'] <= g <= intervalo['G_max'] and
            intervalo['B_min'] <= b <= intervalo['B_max'])

def gerar_csv(atributos_com_intervalo):
    dados_csv = []
    pastas = ['personagem_1', 'personagem_2']

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

    # Salvar o CSV
    try:
        with open(RESULTADO_CSV, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Arquivo', 'Classe', 'Total de Pixels'])
            writer.writerows(dados_csv)
    except Exception as e:
        print(f"Erro ao salvar o CSV: {e}")

if __name__ == '__main__':
    app.run(debug=True)
