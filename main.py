#imports
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# keras imports
import keras
import itertools

#set the path to the dataset folder, set batch_size, epochs and image height and width
PATH = '/home/andrei/Desktop/LungCancerDetection/lung_image_sets'
batch_size = 64
epochs = 30
IMG_HEIGHT = 180
IMG_WIDTH = 180
INIT_LR = 1e-3    # initial learning rate

'''
Abreviations: 
      Lung adenocarcinoma -> lung_aca (Adenocarcinoma of the lung is the most common type of lung cancer, and like other forms of lung cancer, it is characterized by distinct cellular and molecular features)
      Lung squamous cell carcinoma -> lung_scc (Lung squamous cell carcinoma (SCC) is a type of non-small cell (NSCLC) cancer that occurs when abnormal lung cells multiply out of control and form a tumor)
      Lung benign tissue -> lung_n (Lung benign tissue is an abnormal growth of tissue that serves no purpose and is found not to be cancerous. Hamartomas are the most common type of benign lung tumor and the third most common cause of solitary pulmonary nodules)
'''

'''First, I'm creating a list of file paths and a list of labels. Next, I'm creating 
a Pandas Series object from the file paths and labels. Then, we're creating a Pandas 
DataFrame by concatenating the Series objects. Finally, we're resetting the index of 
the DataFrame.
'''

for i, d in enumerate([PATH]):
      paths = [] # path to each image from the dataset
      labels = [] # label for each image from the dataset (lung_n, lung_aca and lung_scc)
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
      lung_df = pd.concat([fSeries, lSeries], axis = 1)

df = pd.concat([lung_df], axis = 0).reset_index(drop = True)

print(df['labels'].value_counts())


'''
First, I'm creating a list called sample_list. Then, I iterate through the 
unique values of the labels column.
For each label, I'm creating a sample of 3000 observations. I'm appending the sample 
to the sample_list. Finally, I'm concatenating the samples and resetting the index.
'''
sampleSize = 3000
sampleList = []

gr = df.groupby('labels')

for label in df['labels'].unique():
      labelGroup = gr.get_group(label).sample(sampleSize, replace=False, random_state=123, axis = 0)
      sampleList.append(labelGroup)

df = pd.concat(sampleList, axis=0).reset_index(drop=True)
print(len(df))


'''Here, I create the train, test and validation datasets from the original dataset.
The original dataset came with all the data in one folder so the train, test and validation
data had to be separated
'''
trainSplit = .8
testSplit = .1
validSplit = testSplit/(1-trainSplit) # splits the test data and validation data in half

train_df, dummy_df = train_test_split(df, train_size = trainSplit, shuffle=True, random_state=123)
test_df, valid_df = train_test_split(dummy_df, train_size=validSplit, shuffle=True, random_state=123)

print(f'Train length: {len(train_df)}\nTest length: {len(test_df)}\nValidation length: {len(valid_df)}')

height=224
width=224
channels=3
batch_size=32
img_shape=(height, width, channels)
img_size=(height, width)
length=len(test_df)
test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n == 0 and length/n<=80],reverse=True)[0]  
test_steps=int(length/test_batch_size)
print ('test batch size: ', test_batch_size, 'test steps: ', test_steps)

def scalar(img):
    return img/127.5-1  # scale pixels between -1 and +1

gen=ImageDataGenerator(preprocessing_function=scalar)
train_gen=gen.flow_from_dataframe( 
      train_df, 
      x_col='filepaths', 
      y_col='labels', target_size=img_size, class_mode='categorical',
      color_mode='rgb', shuffle=True, batch_size=batch_size
)

test_gen=gen.flow_from_dataframe(
      test_df,
      x_col='filepaths', 
      y_col='labels', 
      target_size=img_size, 
      class_mode='categorical',
      color_mode='rgb', shuffle=False, batch_size=test_batch_size
)

valid_gen=gen.flow_from_dataframe( 
      valid_df, 
      x_col='filepaths', 
      y_col='labels', 
      target_size=img_size, 
      class_mode='categorical',
      color_mode='rgb', shuffle=True, batch_size=batch_size
)

classes=list(train_gen.class_indices.keys())
class_count=len(classes)


def display_all_images(umages):
      plt.figure(figsize=(10,10))
      for i in range(9):
            plt.subplot(3,3,i+1)
            img,label=images.next()
            plt.imshow(img[0])
            plt.title(classes[np.argmax(label[0])])
            plt.axis('off')
      plt.show()
      return

display_all_images(train_gen)

# train data
train_steps = int(len(train_df) / batch_size)
print(f'Train steps: {train_steps}')

# test data
test_steps = int(len(test_df) / test_batch_size)
print(f'{test_steps}')

# validation data
valid_steps = int(len(valid_df) / batch_size)
print(f'{valid_steps}')

# create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=img_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(3, activation='relu'))
model.add(Dropout(0.25))

#compile the model
print('Compiling model...')
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / epochs)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

#train the model
history = model.fit(
      x = train_gen,
      steps_per_epoch = train_steps,
      validation_data = valid_gen,
      validation_steps = valid_steps,
      epochs = epochs
)


# evaluate the network
print_color('Evaluating network...', (0, 0, 0), (255, 255, 255))
predictions = model.predict_generator(test_gen, steps=test_steps)
print(classification_report(test_gen.classes, predictions.argmax(axis=1), target_names=test_gen.class_indices.keys()))

# save the model to disk
print_color('Saving model...', (0, 0, 0), (255, 255, 255))
model.save('model.h5')


#plot graph
def graph():
      plt.style.use('ggplot')
      plt.figure()
      N = 20
      plt.plot(np.arange(0, N), history.history['loss'], label='train_loss')
      plt.plot(np.arange(0, N), history.history['val_loss'], label='val_loss')
      plt.plot(np.arange(0, N), history.history['accuracy'], label='train_acc')
      plt.plot(np.arange(0, N), history.history['val_accuracy'], label='val_acc')
      plt.title('Training Loss and Accuracy')
      plt.xlabel('Epoch #')
      plt.ylabel('Loss/Accuracy')
      plt.legend(loc='lower left')
      plt.savefig('plot.png')

# draw confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
      '''
      This function prints and plots the confusion matrix.
      Normalization can be applied by setting `normalize=True`.
      '''
      if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print('Normalized confusion matrix')
      else:
            print('Confusion matrix, without normalization')
            print(cm)
      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      plt.title(title)
      plt.colorbar()
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation=45)
      plt.yticks(tick_marks, classes)
      fmt = '.2f' if normalize else 'd'
      thresh = cm.max() / 2.
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
      plt.show()
      return

cm = confusion_matrix(test_gen.classes, valid_gen.classes)
plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
graph()