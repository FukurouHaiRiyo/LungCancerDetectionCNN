# imports
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_logarithmic_error, categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# keras imports
import keras
import itertools

# Set the path to the dataset folder, set batch_size, epochs and image height and width
PATH = '/home/andrei/Desktop/ProiectLicenta/lung_image_sets'
batch_size = 32
epochs = 8
IMG_H = 180
IMG_W = 180
INIT_LR = 1e-3  # Initial learning rate

# Creating a list of file paths and labels
for i, d in enumerate([PATH]):
    paths = []  # Path to each image from the dataset
    labels = []  # Label for each image from the dataset (lung_n, lung_aca, and lung_scc)
    classes = os.listdir(d)

    for Class in classes:
        classPath = os.path.join(d, Class)
        if os.path.isdir(classPath):
            fList = os.listdir(classPath)
            for f in fList:
                fPath = os.path.join(classPath, f)
                paths.append(fPath)
                labels.append(Class)

    fSeries = pd.Series(paths, name='filepaths')
    lSeries = pd.Series(labels, name='labels')
    lung_df = pd.concat([fSeries, lSeries], axis=1)

df = pd.concat([lung_df], axis=0).reset_index(drop=True)

print(df['labels'].value_counts())

# Creating a sample list
sampleSize = 3000
sampleList = []

gr = df.groupby('labels')

for label in df['labels'].unique():
    labelGroup = gr.get_group(label).sample(sampleSize, replace=False, random_state=123, axis=0)
    sampleList.append(labelGroup)

df = pd.concat(sampleList, axis=0).reset_index(drop=True)
print(len(df))

# Creating train, test, and validation datasets from the original dataset
trainSplit = 0.8
testSplit = 0.1
validSplit = testSplit / (1 - trainSplit)  # Splits the test data and validation data in half

train_df, dummy_df = train_test_split(df, train_size=trainSplit, shuffle=True, random_state=123)
test_df, valid_df = train_test_split(dummy_df, train_size=validSplit, shuffle=True, random_state=123)

print(f'Train length: {len(train_df)}\nTest length: {len(test_df)}\nValidation length: {len(valid_df)}')

# GAN-C data generator
def gan_c_data_generator(df, image_size, batch_size):
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filepaths',
        y_col='labels',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filepaths',
        y_col='labels',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator

img_size = (IMG_H, IMG_W)
train_gen, valid_gen = gan_c_data_generator(train_df, img_size, batch_size)

classes = list(train_gen.class_indices.keys())
class_count = len(classes)

# GAN-C model architecture
def gan_c_model(image_size, num_classes):
    inputs = Input(shape=(image_size[0], image_size[1], 3))
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# GAN-C model instantiation
model = gan_c_model(img_size, class_count)

# Compile the model
print('Compiling model...')
earlystop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    verbose=1,
    restore_best_weights=True
)

opt = Adam(learning_rate=INIT_LR)
model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

# Train the model
history = model.fit(
    x=train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=valid_gen,
    validation_steps=len(valid_gen),
    epochs=epochs,
    callbacks=[earlystop_callback]
)

# Evaluate the network
test_gen, _ = gan_c_data_generator(test_df, img_size, batch_size)
predictions = model.predict_generator(test_gen, steps=len(test_gen))
print(classification_report(test_gen.classes, predictions.argmax(axis=1), target_names=test_gen.class_indices.keys()))

# Save the model to disk
print('Saving model...')
model.save('modelGAN-C.h5')

# Plot graph
def plot_graph(history):
    plt.style.use('ggplot')
    plt.figure()
    N = len(history.history['loss'])
    plt.plot(np.arange(0, N), history.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), history.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), history.history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, N), history.history['val_accuracy'], label='val_acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')
    plt.savefig('plotGAN-C.png')

plot_graph(history)
