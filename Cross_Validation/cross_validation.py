from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
import numpy as np
import pandas as pd
from PIL import Image

import os

train_data_dir = '/net/tijuca/lab/users/LuisH/wagner/ic_fl/TrainData/TrainData'
train_data_negative = '/net/tijuca/lab/users/LuisH/wagner/ic_fl/TrainData/TrainData/Negative data'
train_data_positive = '/net/tijuca/lab/users/LuisH/wagner/ic_fl/TrainData/TrainData/Positive data'

imagens = []
labels = []

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

'''# Exibir as primeiras linhas do DataFrame para verificar se os dados foram carregados corretamente
print(df.head())
'''

image_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def create_model():
    model = Sequential([         # ---------
        Flatten(input_shape=(image_size[0], image_size[1], 3)),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_and_preprocess_data(train_indices, val_indices):
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        seed=47
    )
    
    val_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        seed=47
    )

    return train_generator, val_generator

# Aplicar a validação cruzada
results = []
for train_indices, val_indices in kfold.split(np.zeros((len(df['Imagem']),)), len(df['Label'])):    # --------------
    train_data, val_data = load_and_preprocess_data(train_indices, val_indices)
    model = create_model()
    
    model.fit(train_data, epochs=5)
    
    score = model.evaluate(val_data)
    results.append(score[1])

print("Acurácia média: %.2f%%" % (np.mean(results)*100))
print("Desvio padrão: %.2f" % (np.std(results)))