import os
import tensorflow as tf
import pandas as pd
import numpy as np
from google.colab import drive
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

# Montagem no Drive do Google
drive.mount('/content/drive')

# Parâmetros de repetição, Fold e modelo
REPETICAO = 4
FOLD = 2
SD_10 = True
SEED_TREINO = REPETICAO * 10 + FOLD

# Carregamento das divisões(Split) previamente separadas
Divisao = f'/content/drive/MyDrive/TCC_Splits_25/split_R{REPETICAO}_F{FOLD}.csv'
dados = pd.read_csv(Divisao)
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED_TREINO)

treino = []
validacao = []
teste = []
imagens_teste = []
classes_teste = []
resolucao = [112, 74, 56, 37, 28, 20]
blur = [0.5, 1.0, 1.5, 2.0]
resultados = []
matrizes = {}

# Dimensionamento e carregamento das imagens
def carregar_imagens(caminho, classe):
    imagem = tf.io.read_file(caminho)
    imagem = tf.io.decode_image(imagem, channels=3, expand_animations=False)
    imagem.set_shape([None, None, 3])
    imagem = tf.image.resize(imagem, (224, 224), method='bilinear', antialias=True)
    return imagem, classe

# Criação do conjunto de dados em lote de 32 e embaralhamento
def criar_dataset(dados, treinamento=False):
    caminhos = dados['filepath'].to_numpy()
    classes = dados['label'].to_numpy(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((caminhos, classes))

    if treinamento:
        dataset = dataset.shuffle( len(caminhos), seed=SEED_TREINO, reshuffle_each_iteration=True)

    dataset = dataset.map(carregar_imagens, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Separação dos conjuntos de dados para treino, validação e teste
for index, coluna in dados.iterrows():
  if coluna['conjunto'] == 'treino':
    treino.append(coluna)
  elif coluna['conjunto'] == 'validacao':
    validacao.append(coluna)
  elif coluna['conjunto'] == 'teste':
    teste.append(coluna)

treino = pd.DataFrame(treino)
validacao = pd.DataFrame(validacao)
teste = pd.DataFrame(teste)

dataset_treino = criar_dataset(treino, treinamento=True)
dataset_validacao = criar_dataset(validacao)

for index, coluna in teste.iterrows():
  caminho = coluna['filepath']
  classe = coluna['label']
  imagem, classe = carregar_imagens(caminho, classe)
  imagens_teste.append(imagem)
  classes_teste.append(classe)

imagens_teste = np.array(imagens_teste)
classes_teste = np.array(classes_teste)


# Carregamento da arquitetura pré-treinada VGG16
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for camada in vgg16.layers:
    camada.trainable = False

entrada = layers.Input(shape=(224, 224, 3))

# Data Augmentation
aumento = tf.keras.Sequential()
aumento.add(layers.RandomFlip('horizontal_and_vertical', seed=SEED_TREINO + 1))
aumento.add(layers.RandomRotation(0.05, seed=SEED_TREINO + 2))
aumento.add(layers.RandomZoom(0.10, seed=SEED_TREINO + 3))
aumento.add(layers.RandomTranslation(0.10, 0.10, seed=SEED_TREINO + 4))

x = aumento(entrada)
x = layers.Lambda(tf.keras.applications.vgg16.preprocess_input)(x)

#Construção da arquitetura com ou sem Dropout Espacial
contador_sd = 0
for camada in vgg16.layers:
    if not isinstance(camada, layers.InputLayer):
        x = camada(x)
        if SD_10 and isinstance(camada, layers.MaxPooling2D):
            contador_sd += 1
            x = layers.SpatialDropout2D( 0.10, seed=SEED_TREINO + 100*contador_sd)(x)

#Camadas responsáveis pela classificação
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense( 128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform( seed=SEED_TREINO + 1000))(x)
x = layers.Dropout(0.5, seed=SEED_TREINO + 2000)(x)
x = layers.Dense( 5, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform( seed=SEED_TREINO + 3000))(x)

# Transfer Learning
modelo = tf.keras.Model(inputs=entrada, outputs=x)
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
parada = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, min_delta=0.0001, restore_best_weights=True)
historico = modelo.fit(dataset_treino, validation_data=dataset_validacao, epochs=60, callbacks=[parada])

# Fine-Tuning com o último bloco convolucional(quinto)
for camada in vgg16.layers:
  if camada.name == 'block5_conv1' or camada.name == 'block5_conv2' or camada.name == 'block5_conv3':
    camada.trainable = True

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
parada_FT = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, min_delta=0.0001, restore_best_weights=True)
historico_FT = modelo.fit( dataset_treino, validation_data=dataset_validacao, epochs=40, callbacks=[parada_FT])

# Resultados e pesos salvos na pasta do Google Drive
Caminho_Resultados = '/content/drive/MyDrive/TCC_Resultados'
os.makedirs(Caminho_Resultados, exist_ok=True)
NOME = f'{"SD_10" if SD_10 else "BASE"}_R{REPETICAO}_F{FOLD}'

resultados_treino = modelo.evaluate(criar_dataset(treino))
resultados_validacao = modelo.evaluate(dataset_validacao)
resultados_teste = modelo.evaluate(imagens_teste, classes_teste)

# Avaliação nas imagens originais
probabilidades = modelo.predict(imagens_teste, batch_size=32)
previsoes = np.argmax(probabilidades, axis=1)

acuracia = accuracy_score(classes_teste, previsoes)
recall = recall_score(classes_teste, previsoes, average='macro', labels=[0,1,2,3,4], zero_division=0)
f1 = fbeta_score(classes_teste, previsoes, beta=1, average='macro', labels=[0,1,2,3,4], zero_division=0)
f2 = fbeta_score(classes_teste, previsoes, beta=2, average='macro', labels=[0,1,2,3,4], zero_division=0)
f2_doencas = fbeta_score(classes_teste, previsoes, beta=2, average='macro', labels=[1,2,3,4], zero_division=0)
perda = log_loss(classes_teste, probabilidades, labels=[0,1,2,3,4])
matriz = confusion_matrix(classes_teste, previsoes, labels=[0,1,2,3,4])

resultados.append({'Condicao': 'Clean', 'Nivel': 224, 'Acuracia': acuracia, 'Recall': recall, 'F1': f1, 'F2': f2, 'F2_Doencas': f2_doencas, 'Loss': perda})
matrizes['Clean'] = matriz

# Avaliação de Redução de Resolução
for tamanho in resolucao:
    imagens_reduzidas = []

    for imagem in imagens_teste:
        imagem_reduzida = tf.image.resize(imagem, (tamanho, tamanho), method='bilinear', antialias=True)
        imagem_restaurada = tf.image.resize(imagem_reduzida, (224, 224), method='bilinear', antialias=False)
        imagens_reduzidas.append(imagem_restaurada.numpy())

    imagens_reduzidas = np.array(imagens_reduzidas)
    probabilidades = modelo.predict(imagens_reduzidas, batch_size=32)
    previsoes = np.argmax(probabilidades, axis=1)

    acuracia = accuracy_score(classes_teste, previsoes)
    recall = recall_score(classes_teste, previsoes, average='macro', labels=[0,1,2,3,4], zero_division=0)
    f1 = fbeta_score(classes_teste, previsoes, beta=1, average='macro', labels=[0,1,2,3,4], zero_division=0)
    f2 = fbeta_score(classes_teste, previsoes, beta=2, average='macro', labels=[0,1,2,3,4], zero_division=0)
    f2_doencas = fbeta_score(classes_teste, previsoes, beta=2, average='macro', labels=[1,2,3,4], zero_division=0)
    perda = log_loss(classes_teste, probabilidades, labels=[0,1,2,3,4])
    matriz = confusion_matrix(classes_teste, previsoes, labels=[0,1,2,3,4])

    resultados.append({'Condicao': 'Resolucao', 'Nivel': tamanho, 'Acuracia': acuracia, 'Recall': recall, 'F1': f1, 'F2': f2, 'F2_Doencas': f2_doencas, 'Loss': perda})
    matrizes['Resolucao_'+str(tamanho)] = matriz


# Função de aplicação do Blur Gaussiano
def aplicar_blur(imagem, sigma):
    eixo = tf.range(-5, 6, dtype=tf.float32)
    gaussiana = tf.exp(-(eixo ** 2) / (2 * sigma ** 2))
    gaussiana = gaussiana / tf.reduce_sum(gaussiana)

    kernel = gaussiana[:, None] * gaussiana[None, :]
    kernel = tf.reshape(kernel, (11, 11, 1, 1))
    kernel = tf.tile(kernel, (1, 1, 3, 1))

    imagem = tf.expand_dims(imagem, axis=0)
    imagem = tf.nn.depthwise_conv2d(imagem, kernel, strides=[1, 1, 1, 1], padding='SAME')

    return imagem[0].numpy()

# Avaliação com Blur Gaussiano
for sigma in blur:
    imagens_blur = []

    for imagem in imagens_teste:
        imagem_blur = aplicar_blur(imagem, sigma)
        imagens_blur.append(imagem_blur)

    imagens_blur = np.array(imagens_blur)
    probabilidades = modelo.predict(imagens_blur, batch_size=32)
    previsoes = np.argmax(probabilidades, axis=1)

    acuracia = accuracy_score(classes_teste, previsoes)
    recall = recall_score(classes_teste, previsoes, average='macro', labels=[0,1,2,3,4], zero_division=0)
    f1 = fbeta_score(classes_teste, previsoes, beta=1, average='macro', labels=[0,1,2,3,4], zero_division=0)
    f2 = fbeta_score(classes_teste, previsoes, beta=2, average='macro', labels=[0,1,2,3,4], zero_division=0)
    f2_doencas = fbeta_score(classes_teste, previsoes, beta=2, average='macro', labels=[1,2,3,4], zero_division=0)
    perda = log_loss(classes_teste, probabilidades, labels=[0,1,2,3,4])
    matriz = confusion_matrix(classes_teste, previsoes, labels=[0,1,2,3,4])

    resultados.append({'Condicao': 'Blur', 'Nivel': sigma, 'Acuracia': acuracia, 'Recall': recall, 'F1': f1, 'F2': f2, 'F2_Doencas': f2_doencas, 'Loss': perda})
    matrizes['Blur_'+str(sigma)] = matriz

# Cálculo do Macro-F2 Combinado
tabela = pd.DataFrame(resultados)

f2_resolucao = tabela.loc[tabela['Condicao'] == 'Resolucao', 'F2'].mean()
f2_blur = tabela.loc[tabela['Condicao'] == 'Blur', 'F2'].mean()
f2_combinado = (f2_resolucao + f2_blur) / 2

# Salva as métricas Individuais e Matrizes de confusão
tabela.to_csv(Caminho_Resultados + '/' + NOME + '_resultados.csv', index=False)
print(tabela)

for nome, matriz in matrizes.items():
    tabela_matriz = pd.DataFrame(matriz)
    tabela_matriz.to_csv(Caminho_Resultados + '/' + NOME + '_' + nome + '.csv', index=False)

# Métricas individuais por classe
nomes_classes = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
resultados_classes = []

for nome, matriz in matrizes.items():
  for i in range(5):
    VP = matriz[i, i]
    FP = np.sum(matriz[:, i]) - VP
    FN = np.sum(matriz[i, :]) - VP
    precisao = VP / (VP + FP) if VP + FP > 0 else 0
    recall = VP / (VP + FN) if VP + FN > 0 else 0
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall > 0 else 0
    f2 = 5 * precisao * recall / (4 * precisao + recall) if 4 * precisao + recall > 0 else 0
    resultados_classes.append({'Condicao': nome,'Classe': nomes_classes[i],'Precisao': precisao,'Recall': recall,'F1': f1,'F2': f2})

tabela_classes = pd.DataFrame(resultados_classes)
tabela_classes.to_csv(Caminho_Resultados + '/' + NOME + '_classes.csv', index=False)

# Salva o resumo do experimento
resumo = pd.DataFrame([{'Modelo': 'SD_10' if SD_10 else 'BASE','Repeticao': REPETICAO,'Fold': FOLD,'Seed': SEED_TREINO,'Macro_F2_Clean': tabela.loc[tabela['Condicao'] == 'Clean', 'F2'].iloc[0],'Macro_F2_Resolucao': f2_resolucao,'Macro_F2_Blur': f2_blur,'Macro_F2_Combinado': f2_combinado}])
resumo.to_csv(Caminho_Resultados + '/' + NOME + '_resumo.csv', index=False)
