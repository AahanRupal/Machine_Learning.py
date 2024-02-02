import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import mnist

def load_images(data_dir):
    data = []
    labels = []
    for label in os.listdir(data_dir):
        for img in os.listdir(os.path.join(data_dir, label)):
            img_path = os.path.join(data_dir, label, img)
            img = load_img(img_path, target_size=(32, 32)) # adjust the size to match your model input size
            img_data = img_to_array(img)
            data.append(img_data)
            labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    
    return data,labels

X_train,y_train=load_images('train')

print(X_train.shape)

plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print("Size of new train set:", X_train_new.shape[0])
print("Size of validation set:", X_val.shape[0])

import numpy as np

# Flatten images in the train set
X_train_flattened = np.empty((X_train_new.shape[0], 3072))
for i in range(X_train_new.shape[0]):
    X_train_flattened[i] = X_train_new[i].flatten()


# Flatten images in the validation set
X_val_flattened = np.empty((X_val.shape[0], 3072))
for i in range(X_val.shape[0]):
    X_val_flattened[i] = X_val[i].flatten()

num_classes = 3
num_pixels = X_train.shape[1] * X_train.shape[2]


# print(X_train.shape)
# print(y_train.shape)
encoder = LabelEncoder()
numeric_labels_train = encoder.fit_transform(y_train_new)
numeric_labels_val = encoder.transform(y_val)

# One-hot encode labels
num_classes = len(encoder.classes_)
y_train_categorical = to_categorical(numeric_labels_train, num_classes=num_classes)
y_val_categorical = to_categorical(numeric_labels_val, num_classes=num_classes)

X_train_normalized = X_train_flattened / 255.0
X_val_normalized = X_val_flattened / 255.0

model = Sequential()

model.add(Dense(256,input_dim=X_train_flattened.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation = 'relu'))  # relu tanh   prelu  lrelu
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(X_train_normalized.shape)
# print(y_train_categorical.shape)
# print(X_val_normalized.shape)
# print(y_val_categorical.shape)

# Training model
history = model.fit(X_train_normalized, y_train_categorical, validation_data=(X_val_normalized, y_val_categorical), epochs=500, batch_size=200)
training_acc = history.history['accuracy']
validation_acc = history.history['val_accuracy']

# Plotting accuracies
# plt.plot(training_acc, label='Training Accuracy')
# plt.plot(validation_acc, label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracies')
# plt.legend()
# plt.show()


def load_images(data_dir):
    data = []
    labels = []
    for label in os.listdir(data_dir):
        for img in os.listdir(os.path.join(data_dir, label)):
            img_path = os.path.join(data_dir, label, img)
            img = load_img(img_path, target_size=(32, 32)) # adjust the size to match your model input size
            img_data = img_to_array(img)
            data.append(img_data)
            labels.append(label)
    data = np.array(data)
    labels = np.array(labels)
    
    return data,labels

#testing
X_test,y_test=load_images('test')

X_test_flattened = np.empty((X_test.shape[0], 3072))
for i in range(X_test.shape[0]):
    X_test_flattened[i] = X_test[i].flatten()

numeric_labels_test = encoder.fit_transform(y_test)
num_classes = len(encoder.classes_)
y_test_categorical = to_categorical(numeric_labels_test, num_classes=num_classes)

X_test_normalized = X_test_flattened / 255.0

scores = model.evaluate(X_test_normalized, y_test_categorical)
scores = model.evaluate(X_test_normalized, y_test_categorical)

print(f'Test loss: {scores[0]}')
print(f'Test accuracy: {scores[1]}')