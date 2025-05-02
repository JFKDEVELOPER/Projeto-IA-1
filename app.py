from flask import Flask, render_template, request, redirect, url_for, flash, session
from services.extracao import processar_extracao
from services.treinamento import treinar_mlp
from services.classificacao import classificar_imagem
from services.classificacao_cnn import classificar_imagem_cnn
from services.cnn_treinamento import treinar_cnn
import os

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
        sucesso, mensagem, acuracia = treinar_cnn(request)  # <- retorna 3 valores agora
        if sucesso:
            session['acuracia'] = acuracia
            session['matriz_path'] = 'matriz_cnn.png'
            return redirect(url_for('resultado_cnn'))
        flash(mensagem, 'danger')
        return redirect(url_for('treinamento_cnn'))

    matriz_existe = os.path.exists('static/matriz_cnn.png')
    return render_template('treinamento_cnn.html', matriz_existe=matriz_existe)

@app.route('/classificar_imagem_cnn', methods=['GET', 'POST'])
def classificar_imagem_cnn_route():
    if request.method == 'POST':
        return classificar_imagem_cnn(request.files['imagem'])
    return render_template('classificar_imagem_cnn.html')

@app.route('/resultado_cnn')
def resultado_cnn():
    acuracia = session.pop('acuracia', None)
    matriz_path = session.pop('matriz_path', None)

    if acuracia is not None and matriz_path:
        return render_template('resultado_cnn.html', acuracia=acuracia, matriz_path=matriz_path)

    flash("Nenhum resultado encontrado.", "warning")
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
