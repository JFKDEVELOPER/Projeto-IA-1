from flask import Flask, render_template, request, redirect, url_for, flash
from services.extracao import processar_extracao
from services.treinamento import treinar_mlp
from services.classificacao import classificar_imagem
from services.classificacao_cnn import classificar_imagem_cnn
from services.cnn_treinamento import treinar_cnn  # corrigido: nome do arquivo é cnn_treinamento.py

# app.py – apenas rotas e chamadas de funções

app = Flask(__name__)
app.secret_key = "segredo"
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extracao_pixel', methods=['GET', 'POST'])
def extracao_pixel():
    if request.method == 'POST':
        sucesso, mensagem = processar_extracao(request.files)
        flash(mensagem, 'success' if sucesso else 'danger')
        return redirect(url_for('definir_atributos'))
    return render_template('extracao_pixel.html')
    
@app.route('/definir_atributos', methods=['GET', 'POST'])
def definir_atributos():
    if request.method == 'POST':
        sucesso, mensagem = treinar_mlp(request.form)
        return mensagem if sucesso else f"Erro: {mensagem}"
    return render_template('definir_atributos.html')

@app.route('/classificar_imagem_pixel', methods=['GET', 'POST'])
def classificar_imagem_pixel():
    if request.method == 'POST':
        return classificar_imagem(request.files['imagem'])
    return render_template('classificar_imagem_pixel.html')

@app.route('/treinamento_mlp', methods=['GET', 'POST'])
def treinamento_mlp():
    if request.method == 'POST':
        sucesso, mensagem = treinar_mlp()
        flash(mensagem, 'success' if sucesso else 'danger')
        return redirect(url_for('treinamento_mlp'))
    return render_template('treinamento_mlp.html')

@app.route('/treinamento_cnn', methods=['GET', 'POST'])
def treinamento_cnn():
    if request.method == 'POST':
        sucesso, mensagem = treinar_cnn()
        flash(mensagem, 'success' if sucesso else 'danger')
        return redirect(url_for('treinamento_cnn'))
    return render_template('treinamento_cnn.html')

@app.route('/classificar_imagem_cnn', methods=['GET', 'POST'])
def classificar_imagem():
    if request.method == 'POST':
        return classificar_imagem_cnn(request.files['imagem'])
    return render_template('classificar_imagem_cnn.html')

@app.route('/treinamento_cnn', methods=['GET', 'POST'])
def cnn_treinamento():
    if request.method == 'POST':
        sucesso, mensagem = treinar_cnn(request)
        flash(mensagem, 'success' if sucesso else 'danger')
        return redirect(url_for('cnn_treinamento'))
    return render_template('treinamento_cnn.html')

if __name__ == '__main__':
    app.run(debug=True)
