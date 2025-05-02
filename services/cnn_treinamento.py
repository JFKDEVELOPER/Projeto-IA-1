# services/cnn_treinamento.py

import os
import numpy as np
import json
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
MODELO_PATH = os.path.join('models', 'modelo_cnn.h5')
LABELS_PATH = os.path.join('models', 'labels_map.json')


def treinar_cnn(request):
    try:
        folders = request.files.getlist('folders')
        camadas = int(request.form['camadas'])
        neuronios = int(request.form['neuronios'])
        epocas = int(request.form['epocas'])

        pasta_cnn = os.path.join(UPLOAD_FOLDER, 'cnn')
        os.makedirs(pasta_cnn, exist_ok=True)

        imagens, labels = [], []

        for arquivo in folders:
            if arquivo.filename == '':
                continue

            classe = os.path.basename(os.path.dirname(arquivo.filename))
            caminho = os.path.join(pasta_cnn, os.path.basename(arquivo.filename))
            arquivo.save(caminho)

            try:
                img = Image.open(caminho).convert('RGB').resize((64, 64))
                imagens.append(np.array(img))
                labels.append(classe)
            except:
                continue

        if len(imagens) < 2:
            return False, "Erro: quantidade de imagens insuficiente.", None

        imagens = np.array(imagens) / 255.0
        classes = sorted(list(set(labels)))
        labels_map = {c: i for i, c in enumerate(classes)}
        y = np.array([labels_map[l] for l in labels])

        X_train, X_test, y_train, y_test = train_test_split(imagens, y, test_size=0.2, random_state=42)

        model = Sequential()
        for i in range(camadas):
            if i == 0:
                model.add(Conv2D(neuronios, (3, 3), activation='relu', input_shape=(64, 64, 3)))
            else:
                model.add(Conv2D(neuronios, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(neuronios, activation='relu'))
        model.add(Dense(len(classes), activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=epocas, validation_data=(X_test, y_test), verbose=1)

        model.save(MODELO_PATH)
        with open(LABELS_PATH, 'w') as f:
            json.dump(labels_map, f)

        # Matriz de confusão
        y_pred = np.argmax(model.predict(X_test), axis=1)
        matriz = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão - CNN')
        matriz_path = os.path.join(STATIC_FOLDER, 'matriz_cnn.png')
        plt.savefig(matriz_path)
        plt.close()

        acuracia = round((y_pred == y_test).mean() * 100, 2)
        return True, "Modelo treinado com sucesso!", acuracia

    except Exception as e:
        return False, f"Erro no treinamento: {str(e)}", None
