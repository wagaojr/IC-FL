import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score
import cv2
import os

tf.config.set_visible_devices([], 'GPU')

print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))

train_data_dir = '/net/tijuca/lab/users/LuisH/wagner/ic_fl/TrainData/TrainData'
train_data_negative = '/net/tijuca/lab/users/LuisH/wagner/ic_fl/TrainData/TrainData/negative'
train_data_positive = '/net/tijuca/lab/users/LuisH/wagner/ic_fl/TrainData/TrainData/positive'

imagens = []
labels = []

image_size = (224, 224)
image_shape = image_size + (3,)
batch_size = 16

for nome_arquivo in os.listdir(train_data_negative):
    if nome_arquivo.endswith('.JPG'):
        caminho_completo = os.path.join(train_data_negative, nome_arquivo)
        try:
            imagem = Image.open(caminho_completo)
            array_imagem = np.array(imagem)
            imagens.append(array_imagem)
            labels.append(0)
        except Exception as e:
            print(f"Erro ao processar {nome_arquivo}: {str(e)}")

for nome_arquivo in os.listdir(train_data_positive):
    if nome_arquivo.endswith('.JPG'):
        caminho_completo = os.path.join(train_data_positive, nome_arquivo)
        try:
            imagem = Image.open(caminho_completo)
            array_imagem = np.array(imagem)
            imagens.append(array_imagem)
            labels.append(1)
        except Exception as e:
            print(f"Erro ao processar {nome_arquivo}: {str(e)}")

data = {'Imagem': imagens, 'Label': labels}
df = pd.DataFrame(data)

print(df.head())

kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(kf.get_n_splits(df['Imagem']))

for i, (train_index, test_index) in enumerate(kf.split(df['Imagem'])):
    print(f"Fold {i}:")
    print(f"   Train: index={train_index}")
    print(f"   Test: index={test_index}")

base_model = VGG16(input_shape=image_shape,
                   include_top=False,
                   weights='imagenet')

preprocess_input = tf.keras.applications.vgg16.preprocess_input

base_model.trainable = False

global_average_layer = GlobalAveragePooling2D()

prediction_layer = Dense(1)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.compile(loss='binary_crossentropy', optimizer='adam')

image_train = np.array([np.array(val) for val in df['Imagem']])
label_train = np.array([np.array(val2) for val2 in df['Label']])

print(image_train.shape)
print(label_train.shape)

# Redimensionando o image_train para (224,224,3)
largura, altura = 224, 224

imagens_redimensionadas = []

for image in image_train:
    imagem_redimensionada = cv2.resize(image, (altura, largura))
    imagens_redimensionadas.append(imagem_redimensionada)

array_redimensionado_train = np.array(imagens_redimensionadas)

model.fit(x=array_redimensionado_train,
          y=label_train,
          epochs=10,
          verbose=1,
          callbacks=[early_stop])

array_2d = array_redimensionado_train

n_samples = array_2d.shape[0]

array_2d = np.reshape(array_2d, (n_samples, 224, 224, 3))

print(array_2d.shape)

predictions_list = []

accuracy_list = []
:
for i, (train_index, test_index) in enumerate(kf.split(array_2d)):
    print(f"Fold {i}:")
    print(f"   Train: index={train_index}")
    print(f"   Test: index={test_index}")
    X_train, X_test = array_2d[train_index], array_2d[test_index]
    y_train, y_test = label_train[train_index], label_train[test_index]

    keras_clf = KerasClassifier(model = model, epochs=100, optimizer='adam', verbose=0)

    keras_clf.fit(X_train, y_train, callbacks=[early_stop])

    predictions_fold = keras_clf.predict(X_test)
    accuracy_fold = accuracy_score(y_test, predictions_fold)
    accuracy_list.append(accuracy_fold)
    predictions_list.append(predictions_fold)

    print(f"Acurácia do Fold {i}: {accuracy_fold}")

predictions = np.concatenate(predictions_list)

average_accuracy = np.mean(accuracy_list)
print(f"Acurácia média: {average_accuracy}")

print(predictions)
