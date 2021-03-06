import numpy as np
import skimage.io
import skimage.transform
import os, sys, time
import re
import matplotlib.pyplot as plt
import glob
from keras.utils import to_categorical, Sequence
from keras.applications.vgg16 import preprocess_input
from sklearn.utils import class_weight
import os
import random


class BaseGenerator(Sequence):
    def __init__(self, config):

        self.config = config
        
 
        self.path_action0, _, self.files_action0 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action0') ))
        self.path_action1, _, self.files_action1 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action1') ))
        self.path_action2, _, self.files_action2 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action2') ))
        self.path_action3, _, self.files_action3 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action3') ))
        self.path_action4, _, self.files_action4 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action4') ))
        self.path_action5, _, self.files_action5 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action5') ))
        self.path_action6, _, self.files_action6 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action6') ))

        self.cv_path_action0, _, self.cv_files_action0 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action0') ))
        self.cv_path_action1, _, self.cv_files_action1 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action1') ))
        self.cv_path_action2, _, self.cv_files_action2 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action2') ))
        self.cv_path_action3, _, self.cv_files_action3 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action3') ))
        self.cv_path_action4, _, self.cv_files_action4 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action4') ))
        self.cv_path_action5, _, self.cv_files_action5 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action5') ))
        self.cv_path_action6, _, self.cv_files_action6 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action6') ))


        self.batch_size = self.config['batch_size']


        print('reading files')
        self.paths_train0 = glob.glob( os.path.join(self.path_action0, '*') )
        self.paths_train1 = glob.glob( os.path.join(self.path_action1, '*') )
        self.paths_train2 = glob.glob( os.path.join(self.path_action2, '*') )
        self.paths_train3 = glob.glob( os.path.join(self.path_action3, '*') )
        self.paths_train4 = glob.glob( os.path.join(self.path_action4, '*') )
        self.paths_train5 = glob.glob( os.path.join(self.path_action5, '*') )
        self.paths_train6 = glob.glob( os.path.join(self.path_action6, '*') )

        self.paths_cv0 = glob.glob( os.path.join(self.cv_path_action0, '*') )
        self.paths_cv1 = glob.glob( os.path.join(self.cv_path_action1, '*') )
        self.paths_cv2 = glob.glob( os.path.join(self.cv_path_action2, '*') )
        self.paths_cv3 = glob.glob( os.path.join(self.cv_path_action3, '*') )
        self.paths_cv4 = glob.glob( os.path.join(self.cv_path_action4, '*') )
        self.paths_cv5 = glob.glob( os.path.join(self.cv_path_action5, '*') )
        self.paths_cv6 = glob.glob( os.path.join(self.cv_path_action6, '*') )

        
        self.paths_train0 = self.paths_train0[0 : self.config['train_frames']]
        self.paths_train1 = self.paths_train1[0 : self.config['train_frames']]
        self.paths_train2 = self.paths_train2[0 : self.config['train_frames']]
        self.paths_train3 = self.paths_train3[0 : self.config['train_frames']]
        self.paths_train4 = self.paths_train4[0 : self.config['train_frames']]
        self.paths_train5 = self.paths_train5[0 : self.config['train_frames']]
        self.paths_train6 = self.paths_train6[0 : self.config['train_frames']]  
 
        self.paths_cv0 = self.paths_cv0[0 : self.config['cv_frames']]
        self.paths_cv1 = self.paths_cv1[0 : self.config['cv_frames']]
        self.paths_cv2 = self.paths_cv2[0 : self.config['cv_frames']]
        self.paths_cv3 = self.paths_cv3[0 : self.config['cv_frames']]
        self.paths_cv4 = self.paths_cv4[0 : self.config['cv_frames']]
        self.paths_cv5 = self.paths_cv5[0 : self.config['cv_frames']]
        self.paths_cv6 = self.paths_cv6[0 : self.config['cv_frames']]


        self.cv_output0 = [0]*len(self.paths_cv0)
        self.cv_output1 = [1]*len(self.paths_cv1)
        self.cv_output2 = [2]*len(self.paths_cv2)
        self.cv_output3 = [3]*len(self.paths_cv3)
        self.cv_output4 = [4]*len(self.paths_cv4)
        self.cv_output5 = [5]*len(self.paths_cv5)
        self.cv_output6 = [6]*len(self.paths_cv6)

        self.train_output0 = [0]*len(self.paths_train0)
        self.train_output1 = [1]*len(self.paths_train1)
        self.train_output2 = [2]*len(self.paths_train2)
        self.train_output3 = [3]*len(self.paths_train3)
        self.train_output4 = [4]*len(self.paths_train4)
        self.train_output5 = [5]*len(self.paths_train5)
        self.train_output6 = [6]*len(self.paths_train6)


        self.train_paths = self.paths_train0 + self.paths_train1 + self.paths_train2 + self.paths_train3 + \
                           self.paths_train4 + self.paths_train5 + self.paths_train6


        print('train_paths : ', len(self.train_paths))
 
        
        self.train_outputs = self.train_output0 + self.train_output1 + self.train_output2 + self.train_output3 + \
                             self.train_output4 + self.train_output5 + self.train_output6

 
        self.cv_paths = self.paths_cv0 + self.paths_cv1 + self.paths_cv2 + self.paths_cv3 + \
                           self.paths_cv4 + self.paths_cv5 + self.paths_cv6

        self.cv_outputs = self.cv_output0 + self.cv_output1 + self.cv_output2 + self.cv_output3 + \
                             self.cv_output4 + self.cv_output5 + self.cv_output6


        print('Shuffling') 
        z = list(zip(self.train_paths, self.train_outputs))
        random.shuffle(z)
        self.train_paths, self.train_outputs = zip(*z)

        z = list(zip(self.cv_paths, self.cv_outputs))
        random.shuffle(z)
        self.cv_paths, self.cv_outputs = zip(*z)


    def __len__(self):
        raise NotImplementedError


    def __getitem__(self, idx):
        raise NotImplementedError

    def process(self, img):
        img = skimage.transform.resize(img, (224, 224))
        img = ((img - np.min(img)) * (1/(np.max(img) - np.min(img)) * 255)).astype('uint8')
        img = preprocess_input(img)
        return img


class TrainGenerator(BaseGenerator):

     def __init__(self, config):
        super(TrainGenerator, self).__init__(config)
        print('train action0 : ', len(self.paths_train0))
        print('train action1 : ', len(self.paths_train1))
        print('train action2 : ', len(self.paths_train2))
        print('train action3 : ', len(self.paths_train3))
        print('train action4 : ', len(self.paths_train4))
        print('train action5 : ', len(self.paths_train5))
        print('train action6 : ', len(self.paths_train6))

     def __len__(self):
        return int(len(self.train_paths) / (self.batch_size))


     def __getitem__(self, idx):

        X = np.zeros((self.batch_size, 224, 224, 3))
        Y = np.zeros((self.batch_size, 7))
        batchx = self.train_paths[ int(idx*self.batch_size) : int((idx+1)*self.batch_size) ]
        batchy = self.train_outputs[ int(idx*self.batch_size) : int((idx+1)*self.batch_size) ]

        for i in range(self.batch_size):
            path = batchx[i]
            img = skimage.io.imread( batchx[i] )
            img = self.process(img)
            X[i] = img
            Y[i] = to_categorical( batchy[i], num_classes=7 )
        return (X, Y)


class CVGenerator(BaseGenerator):

     def __init__(self, config):
        super(CVGenerator, self).__init__(config)
        print('cv action0 : ', len(self.paths_cv0))
        print('cv action1 : ', len(self.paths_cv1))
        print('cv action2 : ', len(self.paths_cv2))
        print('cv action3 : ', len(self.paths_cv3))                                                                                      
        print('cv action4 : ', len(self.paths_cv4))                                                                                      
        print('cv action5 : ', len(self.paths_cv5))                                                                                      
        print('cv action6 : ', len(self.paths_cv6)) 

     def __len__(self): 
        return int(len(self.cv_paths) / (self.batch_size))


     def __getitem__(self, idx):

        X = np.zeros((self.batch_size, 224, 224, 3))
        Y = np.zeros((self.batch_size, 7))

        batchx = self.cv_paths[ int(idx*self.batch_size) : int((idx+1)*self.batch_size) ]
        batchy = self.cv_outputs[ int(idx*self.batch_size) : int((idx+1)*self.batch_size) ]

        for i in range(self.batch_size):
            path = batchx[i]
            img = skimage.io.imread( batchx[i] )
            img = self.process(img)
            X[i] = img
            Y[i] = to_categorical( batchy[i], num_classes=7 )

        return (X, Y)


class TestGenerator(BaseGenerator):

    def __init__(self, config):
        super(TestGenerator, self).__init__(config)         
       
        self.config = config
         
        self.path_action0, _, self.files_action0 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action0') ))
        self.path_action1, _, self.files_action1 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action1') ))
        self.path_action2, _, self.files_action2 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action2') ))
        self.path_action3, _, self.files_action3 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action3') ))
        self.path_action4, _, self.files_action4 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action4') ))
        self.path_action5, _, self.files_action5 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action5') ))
        self.path_action6, _, self.files_action6 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action6') ))

        print('reading files')
        self.paths_test0 = glob.glob( os.path.join(self.path_action0, '*') )
        self.paths_test1 = glob.glob( os.path.join(self.path_action1, '*') )
        self.paths_test2 = glob.glob( os.path.join(self.path_action2, '*') )
        self.paths_test3 = glob.glob( os.path.join(self.path_action3, '*') )
        self.paths_test4 = glob.glob( os.path.join(self.path_action4, '*') )
        self.paths_test5 = glob.glob( os.path.join(self.path_action5, '*') )
        self.paths_test6 = glob.glob( os.path.join(self.path_action6, '*') )

        
        self.test_paths = self.paths_test0 + self.paths_test1 + self.paths_test2 + self.paths_test3 + \
                           self.paths_test4 + self.paths_test5 + self.paths_test6


        print('test_paths : ', len(self.test_paths))

        print('sorting')
        self.test_paths = sorted(self.test_paths)
        self.test_outputs = [-1] * len(self.test_paths)

        print('Generating output')
        for i in range(len(self.test_paths)):
            if "action0" in self.test_paths[i]:
                self.test_outputs[i] = 0
            elif "action1" in self.test_paths[i]:
                self.test_outputs[i] = 1
            elif "action2" in self.test_paths[i]:
                self.test_outputs[i] = 2
            elif "action3" in self.test_paths[i]:
                self.test_outputs[i] = 3
            elif "action4" in self.test_paths[i]:
                self.test_outputs[i] = 4
            elif "action5" in self.test_paths[i]:
                self.test_outputs[i] = 5
            elif "action6" in self.test_paths[i]:
                self.test_outputs[i] = 6
       

        print('Shuffling')
        z = list(zip(self.test_paths, self.test_outputs))
        random.shuffle(z)
        self.test_paths, self.test_outputs = zip(*z)


    def __len__(self):
         return int(len(self.test_paths) / (self.batch_size))


    def __getitem__(self, idx):

        X = np.zeros((self.batch_size, 224, 224, 3))
        Y = np.zeros((self.batch_size, 7))

        batchx = self.test_paths[ int(idx*self.batch_size) : int((idx+1)*self.batch_size) ]
        batchy = self.test_outputs[ int(idx*self.batch_size) : int((idx+1)*self.batch_size) ]

        for i in range(self.batch_size):
            img = skimage.io.imread(batchx[i])
            img = self.process(img)
            X[i] = img
            Y[i] = to_categorical(batchy[i], num_classes=7)
    
        return (X, Y)
