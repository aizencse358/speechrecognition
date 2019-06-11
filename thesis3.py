import librosa
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from keras.callbacks import TensorBoard
import preprocess
from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import tensorboard
import matplotlib
import itertools
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from keras import optimizers
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import scipy.ndimage


DATA_PATH = "audio/"
feature_dim_2 = 11
max_len=11
path = DATA_PATH


def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# Predicts one sample
def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]





labels, _, _ = get_labels(path)
print(labels)
for label in labels:
    # Init mfcc vectors
    mfcc_vectors = []

    wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
    for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            if (max_len > mfcc.shape[1]):
                pad_width = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :max_len]    
            mfcc_vectors.append(mfcc)
    np.save(label + '.npy', mfcc_vectors)
    
split_ratio=0.6
random_state=42

# Get available labels
labels, indices, _ = get_labels(DATA_PATH)
print(indices)

# Getting first arrays
X = np.load(labels[0] + '.npy')
y = np.zeros(X.shape[0])

# Append all of the dataset into one single array, same goes for y
for i, label in enumerate(labels[1:]):
    x = np.load(label + '.npy')
    X = np.vstack((X, x))
    y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

assert X.shape[0] == len(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


# # Feature dimension
feature_dim_1 = 20
channel = 1
epochs = 50
batch_size = 100
verbose = 1
num_classes = 20

X_train = X_train.reshape(X_train.shape[0],feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

print(X_train.shape)


model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, 
          verbose=verbose, validation_data=(X_test, y_test_hot),callbacks=[tbCallBack])

model.save('thesis1.h5')
model.summary()
from keras.models import load_model

classifier1 = load_model()




def wav2mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

# Getting the MFCC
sample = wav2mfcc('pred/00f0204f_nohash_0.wav')
# We need to reshape it remember?
sample_reshaped = sample.reshape(1, 20, 11, 1)
# Perform forward pass
print(get_labels()[0][
    np.argmax(model.predict(sample_reshaped))
])    
        
        
        
        
        
        
        
        
        
        