

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Carregar o conjunto de dados Fashion MNIST
(imagens_treino, labels_treino), (imagens_teste, labels_teste) = datasets.fashion_mnist.load_data()

# Mapear os rótulos para os nomes das classes correspondentes
nomes_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalizar as imagens
imagens_treino = imagens_treino / 255.0
imagens_teste = imagens_teste / 255.0

# Função para visualizar imagens
def visualiza_imagens(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(nomes_classes[labels[i]])
    plt.show()

visualiza_imagens(imagens_treino, labels_treino)

# Criar o modelo
modelo_lia = models.Sequential()
modelo_lia.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
modelo_lia.add(layers.MaxPooling2D((2, 2)))
modelo_lia.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo_lia.add(layers.MaxPooling2D((2, 2)))
modelo_lia.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo_lia.add(layers.Flatten())
modelo_lia.add(layers.Dense(64, activation='relu'))
modelo_lia.add(layers.Dense(10, activation='softmax'))

modelo_lia.summary()

# Compilar o modelo
modelo_lia.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Treinar o modelo
history = modelo_lia.fit(np.expand_dims(imagens_treino, axis=-1),
                         labels_treino,
                         epochs=10,
                         validation_data=(np.expand_dims(imagens_teste, axis=-1), labels_teste))

# Avaliar o modelo
erro_teste, acc_teste = modelo_lia.evaluate(np.expand_dims(imagens_teste, axis=-1), labels_teste, verbose=2)
print('\nAcurácia com dados de Teste:', acc_teste)

# Carregar uma nova imagem
nova_imagem = Image.open("dados/nova_imagem.jpg")
nova_imagem = nova_imagem.convert('L')  # Converter para escala de cinza

# Redimensionar a imagem para o tamanho esperado pelo modelo (28x28)
nova_imagem = nova_imagem.resize((28, 28))

plt.figure(figsize=(1, 1))
plt.imshow(nova_imagem, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

# Converter a imagem para um array normalizado
nova_imagem_array = np.array(nova_imagem) / 255.0
nova_imagem_array = np.expand_dims(nova_imagem_array, axis=0)

# Fazer previsões
previsoes = modelo_lia.predict(nova_imagem_array)
classe_prevista = np.argmax(previsoes)
nome_classe_prevista = nomes_classes[classe_prevista]
print("A nova imagem foi classificada como:", nome_classe_prevista)
