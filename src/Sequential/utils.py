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


input_shape = []
if K.image_data_format() == 'channels_first':
    print('channels first')
    input_shape = (3, 224, 224)
else:
    print('channels last')
    input_shape = (224, 224, 3)

os.chdir('../..')
root_dir = os.getcwd()
frames_path =  root_dir + '/data/frames'
actions_list_path = root_dir+'/data/classInd.txt'
dirs = os.listdir(frames_path)


f = open(actions_list_path, 'r')
a = f.readlines()
f.close()
a = [l.replace("\n", "") for l in a]
a = [l.replace(" ", "") for l in a]
a = [re.sub('[0-9]+', '', i) for i in a]
a = ['_'+l for l in a]
my_dict = {}
for i in range(0, 101):
    my_dict[i] = a[i]


################################################################################################################################################




def get_folders(action_num):
    action_str = my_dict[action_num]
    folder_names = []
    for folder in dirs:
        if action_str in folder:
            folder_names.append(folder)
    return folder_names 


def get_total_videos(action_num):
    assert  0 <= action_num <= 100 , ('wrong action index %d. Actions vary from 0-100' % action_num)
    folder_names= get_folders(action_num)
    return len(folder_names)


def get_vid(action_num, vid_num, num_frames = -1, flip=False, include_both=False, img_size=(224,224)):
    assert  0 <= action_num <= 100 , ('wrong action index %d. Actions vary from 1-101' % action_num)
    action_str = my_dict[action_num]
    assert  0 <= vid_num < get_total_videos(action_num), ("action-"+str(action_num) + " has total %d videos." % get_total_videos(action_num)) 
    
    folder_names= get_folders(action_num)
    imgs_path = os.path.join(frames_path, folder_names[vid_num])
    im_paths = os.listdir(imgs_path)
    im_paths.sort()
    X = []
    count = 0
    for i in range(len(im_paths)):
        img = cv2.imread(os.path.join(imgs_path, im_paths[i]))
        img = cv2.resize(img, img_size)
        img_flip = cv2.flip(img, 1) 
       
        img = img - 127
        img_flip = img_flip - 127

        if(not include_both):         
            if(flip):
                X.append(img_flip)
            else:
                X.append(img)

        if(include_both):
            X.append(img)
            X.append(img_flip)  

        count += 1
        if(count == num_frames):
            break
    X = np.array(X)
    print(X.shape)
    
    return X


def get_frames_count(action_num, vid_num):
    assert  0 <= action_num <= 100 , ('wrong action index %d. Actions vary from 0-100' % action_num)
    action_str = my_dict[action_num]
    assert  0 <= vid_num < get_total_videos(action_num), ("action-"+str(action_num) + " has total %d videos." % get_total_videos(action_num)) 
    
    folder_names= get_folders(action_num)
    imgs_path = os.path.join(frames_path, folder_names[vid_num])
    arr = os.listdir(imgs_path) 
    
    return len(arr)


def get_action_info(action_num):
    assert  0 <= action_num <= 100 , ('wrong action index %d. Actions vary from 0-100' % action_num)
    action_str = my_dict[action_num]
    print(action_str)
    
    total_vids = get_total_videos(action_num)
    arr = []
    for i in range(total_vids):
        fc = get_frames_count(action_num, i)
        #print('Vid %d - %d' % (i, fc))
        arr.append([i, fc])
    arr = np.array(arr)
    arr = np.array( sorted(arr,key=lambda x: x[1]) )
    arr = arr[::-1]
    for i in range(total_vids):
        print('Vid %d - %d' % (arr[i,0], arr[i,1]))
        
    return arr


def get_total_frames(action_num):
    assert  0 <= action_num <= 100 , ('wrong action index %d. Actions vary from 0-100' % action_num)
    action_str = my_dict[action_num]
    
    total_vids = get_total_videos(action_num)
    count = 0
    for i in range(total_vids):
        count += get_frames_count(action_num, i)
        
    return count    


def display_vid(X, name):
    for i in range(X.shape[0]):
        cv2.imshow(name, X[i])
        if(cv2.waitKey(30) == ord('q')):
            break
    cv2.destroyAllWindows()


def getX(vid_list, action_num, samples_per_video=-1, flip=False, include_both=False):
    X_train = []
    for i in range(len(vid_list)):
        X = get_vid(action_num, vid_list[i], samples_per_video, flip, include_both)
        if(X_train == []):
            X_train = X
        else:
            X_train = np.concatenate((X_train, X), axis=0)
    return X_train  



def get_model_classification(actions_len, input_shape, drop_prob):

    model = Sequential()
    model.add( Conv2D(96, kernel_size=(7, 7), strides=(2, 2), padding='valid', input_shape=input_shape) )
    model.add( Activation('relu') )
    model.add( MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same') )
    model.add( BatchNormalization() )

    model.add( Conv2D(384, kernel_size=(5, 5), strides=(2, 2), padding='valid') )
    model.add( Activation('relu'))
    model.add( MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same') )
    model.add( BatchNormalization() )

    model.add( Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='valid') )
    model.add( Activation('relu') )

    model.add( Conv2D(512, kernel_size=(3, 3),strides=(1, 1), padding='valid') )
    model.add( Activation('relu') )

    
    model.add( Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid') )
    model.add( Activation('relu') )
    

    model.add( Flatten() )
    model.add( Activation('relu') )

    model.add( Dense(1024) )
    model.add( Activation('relu') )
    model.add(Dropout(drop_prob))


    model.add( Dense(512) )
    model.add( Activation('relu') )
    model.add(Dropout(drop_prob))
  
    return model


def get_model_lstm(actions_len, input_shape, drop_prob):

    model = Sequential()
    model.add(TimeDistributed(  Conv2D(96, kernel_size=(7, 7), strides=(2, 2), padding='valid') , input_shape=input_shape  ))
    model.add(  Activation('relu')  )
    model.add(TimeDistributed(  MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same')  ))
    model.add(TimeDistributed(  BatchNormalization()  ))

    model.add(TimeDistributed(  Conv2D(384, kernel_size=(5, 5), strides=(2, 2), padding='valid')  ))
    model.add( Activation('relu') )
    model.add(TimeDistributed(  MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')  ))
    model.add(TimeDistributed(  BatchNormalization()  ))

    model.add(TimeDistributed(  Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='valid')  ))
    model.add( Activation('relu') )

    model.add(TimeDistributed(  Conv2D(512, kernel_size=(3, 3),strides=(1, 1), padding='valid')  ))
    model.add( Activation('relu') )
    
    model.add(TimeDistributed(  Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='valid') ) )
    model.add( Activation('relu') )

    model.add( TimeDistributed( Flatten() ) )
    model.add( Activation('relu') )

    model.add( TimeDistributed( Dense(1024) ) )
    model.add( Activation('relu') )
    model.add( Dropout(drop_prob) )


    model.add( TimeDistributed( Dense(512) ) )
    model.add( Activation('relu') )
    model.add( Dropout(drop_prob) )

    
    return model




