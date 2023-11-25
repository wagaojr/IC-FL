from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
import numpy as np

import os

train_data_dir = '/net/tijuca/lab/users/LuisH/wagner/ic_fl/TrainData/TrainData'

def count_images(directory):
    total = 0
    for root, dirs, files in os.walk(directory):
        total += len(files)
    return total

total_samples = count_images(train_data_dir)
print("Total de amostras:", total_samples)

labels = os.listdir(train_data_dir)
num_classes = len(labels)

print("Labels:", labels)
print("Número de classes:", num_classes)

image_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def create_model():
    model = Sequential([
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
for train_indices, val_indices in kfold.split(np.zeros((total_samples,)), labels):
    train_data, val_data = load_and_preprocess_data(train_indices, val_indices)
    model = create_model()
    
    model.fit(train_data, epochs=5)
    
    score = model.evaluate(val_data)
    results.append(score[1])

print("Acurácia média: %.2f%%" % (np.mean(results)*100))
print("Desvio padrão: %.2f" % (np.std(results)))