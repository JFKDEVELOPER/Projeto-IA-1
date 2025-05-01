# services/cnn_treinamento.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def treinar_cnn(imagens, rotulos, input_shape, num_classes):
    # Pré-processamento dos dados
    X = np.array(imagens).astype('float32') / 255.0
    y = to_categorical(rotulos, num_classes)

    # Construção do modelo
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinamento
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    return model
