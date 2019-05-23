import warnings
warnings.filterwarnings("ignore")

from data_loader import TestGenerator
from utils import model_lstm_without_top

from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.layers import Dense, Flatten
from keras.layers import Activation, Dropout
from keras.layers import ZeroPadding2D, BatchNormalization, GlobalAveragePooling1D
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from utils import TrainValTensorBoard

import argparse
import os
import numpy as np
from sklearn.metrics import confusion_matrix


config = {'data_path'     :  '../data/',
          'batch_size'    :  8,
          'seq_len'       :  16, 
          'epochs'        :  10,
          'learning_rate' :  0.001,
          'decay_rate'    :  0.1, 
          'train_frames'  :  8000,
          'cv_frames'     :  800,
          'initial_epoch'   :  0}


def get_args():
    global config
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--load_checkpoint', default='', help='Path to hdf5 file to restore model from checkpoint')
    argparser.add_argument('--initial_epoch', default=0, help='start_epoch parameter in model.fit()')
    args = argparser.parse_args()
    return args


pred_val, true_val= [], []

def main():
    args = get_args()
    config['load_checkpoint'] = args.__dict__['load_checkpoint']

    input_shape = (config['seq_len'], 224, 224, 3)

    test_gen = TestGenerator(config)

    len_test = len(test_gen.test_paths)
    print('len_train : ', len_test)

    model = model_lstm_without_top(input_shape)

    if os.path.isfile('weights/vgg16_weights_without_top.h5'):
        model.load_weights('weights/vgg16_weights_without_top.h5')
        print('Loaded VGG16 weights')
        
    model.add(TimeDistributed(Flatten()))

    for layer in model.layers:
        layer.trainable = False

    model.add( TimeDistributed( Dense(128) ) )
    model.add(Activation('relu') )
    model.add( Dropout(0.3) )
    model.add(LSTM(16, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(16, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(7))
    model.add(GlobalAveragePooling1D())
    model.add(Activation('softmax'))

    optim = optimizers.Adam(lr = config['learning_rate'], decay = config['decay_rate'])
    model.compile(optimizer=optim,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if config['load_checkpoint'] != '':
        model.load_weights(config['load_checkpoint'])
        print('Successfully loaded weights from %s' % config['load_checkpoint'])
    else:
        print('No checkpoint found')
    model.summary()
    
    y_true = []
    y_prediction = []
    y_prediction_prob = []   
    for i in range(int(len_test/config['batch_size'])):
        print( i, ' / ', int(len_test/config['batch_size']) ) 
        X_test, Y_test = test_gen.__getitem__(i)
        Y_pred = model.predict(X_test)
        
        for j in range(config['batch_size']):
            y_true.append(np.argmax(Y_test[j]))
            y_prediction.append(np.argmax(Y_pred[j])) 


    print('y_true : ', y_true)
    print('y_prediction : ', y_prediction)
    
    cm = confusion_matrix(y_true, y_prediction)
    print('\n\n')
    print(cm)

if __name__ == '__main__':
    main()
