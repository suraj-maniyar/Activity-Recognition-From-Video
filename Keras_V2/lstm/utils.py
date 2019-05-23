import os
import cv2
import numpy as np
import re
import keras
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.layers import Dense, Flatten
from keras import losses
from keras.layers import Input, Lambda, Reshape, Concatenate, Activation, Dropout
from keras.layers import Conv2DTranspose, ZeroPadding2D, BatchNormalization, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras import metrics
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.models import model_from_json
from keras.callbacks import Callback
from keras import regularizers
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import TensorBoard
import tensorflow as tf

input_shape = []
if K.image_data_format() == 'channels_first':
    print('channels first')
    input_shape = (3, 224, 224)
else:
    print('channels last')
    input_shape = (224, 224, 3)




def model_lstm_without_top(input_shape):
    model = Sequential()
    model.add(TimeDistributed( ZeroPadding2D( (1,1)) , input_shape=input_shape ))
    model.add(TimeDistributed( Conv2D(64, (3, 3), activation='relu') ))
    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(64, (3, 3), activation='relu') ))
    model.add(TimeDistributed( MaxPooling2D((2,2), strides=(2,2)) ))
    
    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(128, (3, 3), activation='relu') ))
    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(128, (3, 3), activation='relu') ))
    model.add(TimeDistributed( MaxPooling2D((2,2), strides=(2,2)) ))

    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(256, (3, 3), activation='relu') ))
    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(256, (3, 3), activation='relu') ))
    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(256, (3, 3), activation='relu') ))
    model.add(TimeDistributed( MaxPooling2D((2,2), strides=(2,2)) ))

    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(512, (3, 3), activation='relu') ))
    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(512, (3, 3), activation='relu') ))
    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(512, (3, 3), activation='relu') ))
    model.add(TimeDistributed( MaxPooling2D((2,2), strides=(2,2)) ))

    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(512, (3, 3), activation='relu') ))
    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(512, (3, 3), activation='relu') ))
    model.add(TimeDistributed( ZeroPadding2D((1,1)) ))
    model.add(TimeDistributed( Conv2D(512, (3, 3), activation='relu') ))
    model.add(TimeDistributed( MaxPooling2D((2,2), strides=(2,2)) ))
       
    return model


def model_without_top(input_shape):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    return model



class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='tensorboard/', **kwargs):
        if not os.path.isdir(log_dir):
            os.mkdir( log_dir )
            os.mkdir( log_dir+'training')
            os.mkdir( log_dir+'validation')
            print('Created tensorboard directories') 
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
