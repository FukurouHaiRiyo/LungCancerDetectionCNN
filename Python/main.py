#imports
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_logarithmic_error
from tensorflow.keras.callbacks import EarlyStopping

# sklearn imports
from sklearn.metrics import confusion_matrix, classification_report

# keras imports
import keras
import itertools

# Set the path to the dataset folder, set batch_size, epochs and image height and width
PATH = '/home/andrei/Desktop/ProiectLicenta/lung_image_sets'
batch_size = 8
epochs = 10
IMG_H = 180
IMG_W = 180
INIT_LR = 1e-3    # Initial learning rate

# Creating a list of file paths and labels
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')
test_dir = os.path.join(PATH, 'test')

print(f'Train directory: {train_dir}\nTest Directory: {test_dir}\nValidation directory: {validation_dir}')

#print all the images for training, validation and testing
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = sum([len(files) for r, d, files in os.walk(test_dir)])

print(f'Total train: {total_train}\nTotal validation: {total_val}\nTotal test: {total_test}')

def scalar(img):
    return img / 127.5 - 1  # Scale pixels between -1 and +1

gen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)


train_gen = gen.flow_from_directory(
    train_dir,
    target_size=(IMG_H, IMG_W),
    class_mode='categorical',
    shuffle=True,
    batch_size=batch_size
)

test_gen = gen.flow_from_directory(
    test_dir,
    target_size=(IMG_H, IMG_W),
    class_mode='categorical',
    shuffle=True,
    batch_size=batch_size
)

valid_gen = gen.flow_from_directory(
    validation_dir,
    target_size=(IMG_H, IMG_W),
    class_mode='categorical',
    shuffle=True,
    batch_size=batch_size
)

classes = list(train_gen.class_indices.keys())
class_count = len(classes)

def display_all_images(images):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img, label = images.next()
        plt.imshow(img[0])
        plt.title(classes[np.argmax(label[0])])
        plt.axis('off')
    plt.show()
    return

display_all_images(train_gen)

# create the model
model = Sequential()
model.add(Conv2D(64,(3,3), padding='same', activation='relu', input_shape=(IMG_H, IMG_W, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(8,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(3, activation='sigmoid'))

#compile the model
print('Compiling model...')

opt = Adam(learning_rate=INIT_LR)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#train the model
history = model.fit(
    x = train_gen,
    steps_per_epoch = total_train // batch_size,
    epochs = epochs,
    validation_data = valid_gen,
    validation_steps = total_val // batch_size
)

# # evaluate the network
# print('Evaluating network...')
# predictions = model.predict_generator(test_gen, steps=test_steps)
# print(classification_report(test_gen.classes, predictions.argmax(axis=1), target_names=test_gen.class_indices.keys()))

# save the model
print('Saving model...')
model.save('modelCNN.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print(f'Accuracy: {acc}, Validation accuracy: {val_acc}\nLoss: {loss}, Validation Loss: {val_loss}')


#plot graph
def graph():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training accuracy')
    plt.plot(epochs_range, val_acc, label='Validation accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and validation accuracy')

    plt.plot(epochs_range, loss, label='Training loss')
    plt.plot(epochs_range, val_loss, label='Validation loss')
    plt.legend(loc='upper right')
    plt.title('Training and validation loss')
    plt.show()

graph()


# show the confusion matrix
def show_confusion_matrix():
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.labels
    confusion_mtx = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(class_count)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_mtx.max() / 2.0
    for i, j in itertools.product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
        plt.text(j, i, format(confusion_mtx[i, j], 'd'),
                horizontalalignment="center",
                color="white" if confusion_mtx[i, j] > thresh else "black") 

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

show_confusion_matrix()
