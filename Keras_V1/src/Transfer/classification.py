from utils import *

################################################################################################################################################

seq = 1
samples_per_video = -1
dump = False
load_from_dump = False 
shuffle = True 

learning_rate = 1e-4
batch_size = 128
n_epochs = 16
drop_prob = 0.5
random_state = 1

decay_rate = 0.01

actions = [0, 1, 7, 19, 25, 55, 70]
opt = np.eye(len(actions))

filepath = 'weights-classification-' + str(len(actions)) + '.hdf5'


optim = optimizers.Adam(lr = learning_rate, decay = decay_rate)

include_both = False
flip = False

################################################################################################################################################

print('seq: ', seq)

vid_list = []
X_train, Y_train, X_CV, Y_CV, X_test, Y_test = [], [], [], [], [], []

if(load_from_dump):
    print('Loading X_train')
    X_train = pickle.load(open('dumps/X_train.p', 'rb'))
    print('Loading Y_train')
    Y_train = pickle.load(open('dumps/Y_train.p', 'rb'))
    print('Loading X_CV')
    X_CV = pickle.load(open('dumps/X_CV.p', 'rb'))
    print('Loading Y_CV')
    Y_CV = pickle.load(open('dumps/Y_CV.p', 'rb'))
    print('Loading X_test')
    X_test = pickle.load(open('dumps/X_test.p', 'rb'))
    print('Loading Y_test')
    Y_test = pickle.load(open('dumps/Y_test.p', 'rb'))


else:

    print('Train Data')

    if(seq == 1):
        vid_list = [37, 92, 10, 121, 13, 5, 57, 45]
    elif(seq == 2):
        vid_list = [86, 28, 66, 63, 112, 117, 61, 87]

    vid_list = [37, 92, 10, 121, 13, 5, 57, 45, 86, 28, 66, 63, 112, 117, 61, 87]
    X0_train = getX(vid_list, actions[0], samples_per_video, flip=flip, include_both=include_both)
    print('X0_train.shape', X0_train.shape)


    if(seq == 1):
        vid_list = [84, 80, 62, 24, 3, 93, 12, 111]
    elif(seq == 2):
        vid_list = [63, 5, 1, 78, 107, 76, 106, 85, 59]

    vid_list = [84, 80, 62, 24, 3, 93, 12, 111, 63, 5, 1, 78, 107, 76, 106, 85, 59]
    X1_train = getX(vid_list, actions[1], samples_per_video, flip=flip, include_both=include_both)
    print('X1_train.shape', X1_train.shape)

    
    if(seq == 1):
        vid_list = [11, 193, 168, 40, 160, 112, 150, 106]
    elif(seq == 2):
        vid_list = [78, 230, 166, 254, 199, 221, 251, 234, 233, 48, 63]

    vid_list = [11, 193, 168, 40, 160, 112, 150, 106, 78, 230, 166, 254, 199, 221, 251, 234, 233, 48, 63]
    X2_train = getX(vid_list, actions[2], samples_per_video, flip=flip, include_both=include_both)
    print('X2_train.shape', X2_train.shape)

    
    if(seq == 1):
        vid_list = [112, 87, 14, 62, 12, 71, 84]
    elif(seq == 2):
        vid_list = [82, 65, 98, 94, 91, 35, 75]
    
    vid_list = [112, 87, 14, 62, 12, 71, 84, 82, 65, 98, 94, 91, 35, 75]
    X3_train = getX(vid_list, actions[3], samples_per_video, flip=flip, include_both=include_both)
    print('X3_train.shape', X3_train.shape)

    
    if(seq == 1):
        vid_list = [74, 3, 127, 53, 121, 88, 107]
    elif(seq == 2):
        vid_list = [99, 70, 123, 32, 28, 106, 9, 24]

    vid_list = [74, 3, 127, 53, 121, 88, 107, 99, 70, 123, 32, 28, 106, 9, 24]
    X4_train = getX(vid_list, actions[4], samples_per_video, flip=flip, include_both=include_both)
    print('X4_train.shape', X4_train.shape)


    if(seq == 1):
        vid_list = [59, 53, 125, 94, 19, 9, 14]
    elif(seq == 2):
        vid_list = [51, 127, 91, 43, 30, 72, 113]

    vid_list = [59, 53, 125, 94, 19, 9, 14, 51, 127, 91, 43, 30, 72, 113]
    X5_train = getX(vid_list, actions[5], samples_per_video, flip=flip, include_both=include_both)
    print('X5_train.shape', X5_train.shape)


    if(seq == 1):
        vid_list = [158, 157, 156, 155, 154, 152, 149]
    elif(seq == 2):
        vid_list = [147, 146, 145, 144, 142, 141, 140]

    vid_list = [158, 157, 156, 155, 154, 152, 149, 147, 146, 145, 144, 142, 141, 140]
    X6_train = getX(vid_list, actions[6], samples_per_video, flip=flip, include_both=include_both)
    print('X6_train.shape', X6_train.shape)
    
    ##############################################################################

    print('CV Data')

    vid_list = [62, 31, 55, 76, 82, 130, 71, 99, 48, 118]
    X0_CV = getX(vid_list, actions[0], samples_per_video, flip=flip, include_both=False)
    print('X0_CV.shape', X0_CV.shape)


    vid_list = [8, 46, 52, 50, 83, 79, 99, 95, 36, 47]
    X1_CV = getX(vid_list, actions[1], samples_per_video, flip=flip, include_both=False)
    print('X1_CV.shape', X1_CV.shape)        


    vid_list = [190, 180, 92, 87, 54, 44, 260, 131, 116, 236]
    X2_CV = getX(vid_list, actions[2], samples_per_video, flip=flip, include_both=False)
    print('X2_CV.shape', X2_CV.shape)        


    vid_list = [72, 108, 1, 99, 73, 13, 60, 57, 27, 19]
    X3_CV = getX(vid_list, actions[3], samples_per_video, flip=flip, include_both=False)
    print('X3_CV.shape', X3_CV.shape)        
        

    vid_list = [61, 146, 125, 69, 93, 5, 79, 103, 52, 80]
    X4_CV = getX(vid_list, actions[4], samples_per_video, flip=flip, include_both=False)
    print('X4_CV.shape', X4_CV.shape)


    vid_list = [130, 35, 36, 111, 80, 56, 89]
    X5_CV = getX(vid_list, actions[5], samples_per_video, flip=flip, include_both=False)
    print('X5_CV.shape', X5_CV.shape)


    vid_list = [114, 67, 26, 118, 117, 40, 52, 28, 14]
    X6_CV = getX(vid_list, actions[6], samples_per_video, flip=flip, include_both=False)
    print('X6_CV.shape', X6_CV.shape)

    ##############################################################################
 
    print('Test Data')

    vid_list = [60, 32, 11, 98, 47, 90, 142, 115, 116]
    X0_test = getX(vid_list, actions[0], samples_per_video, flip=flip, include_both=False)
    print('X0_test.shape', X0_test.shape) 


    vid_list = [55, 92, 58, 94, 72, 75, 43, 39, 68]
    X1_test = getX(vid_list, actions[1], samples_per_video, flip=flip, include_both=False)
    print('X1_test.shape', X1_test.shape) 


    vid_list = [88, 9, 235, 198, 249, 161, 80, 73, 243, 222]
    X2_test = getX(vid_list, actions[2], samples_per_video, flip=flip, include_both=False)
    print('X2_test.shape', X2_test.shape) 


    vid_list = [78, 5, 3, 59, 120, 43, 111]
    X3_test = getX(vid_list, actions[3], samples_per_video, flip=flip, include_both=False)
    print('X3_test.shape', X3_test.shape) 


    vid_list = [60, 21, 8, 141, 76, 115, 148, 7, 4]
    X4_test = getX(vid_list, actions[4], samples_per_video, flip=flip, include_both=False)
    print('X4_test.shape', X4_test.shape)


    vid_list = [28, 60, 52, 82, 63, 95, 55]
    X5_test = getX(vid_list, actions[5], samples_per_video, flip=flip, include_both=False)
    print('X5_test.shape', X5_test.shape)


    vid_list = [108, 98, 136, 150, 105, 119]
    X6_test = getX(vid_list, actions[6], samples_per_video, flip=flip, include_both=False)
    print('X6_test.shape', X6_test.shape)

    ##############################################################################
    
    len_train = X0_train.shape[0]+X1_train.shape[0]+X2_train.shape[0]+X3_train.shape[0]+X4_train.shape[0]+X5_train.shape[0]+X6_train.shape[0]
    len_CV = X0_CV.shape[0] + X1_CV.shape[0] + X2_CV.shape[0] + X3_CV.shape[0] + X4_CV.shape[0] + X5_CV.shape[0] + X6_CV.shape[0]
    len_test = X0_test.shape[0]+X1_test.shape[0]+X2_test.shape[0]+X3_test.shape[0]+X4_test.shape[0]+X5_test.shape[0]+X6_test.shape[0]
    

    X_train, Y_train = [], []
    X_CV, Y_CV = [], []
    X_test, Y_test = [], []    
    
    print('Generating XY-Train ......')
    for i in range(X0_train.shape[0]):
        X_train.append(X0_train[i])
        Y_train.append(opt[0])
    for i in range(X1_train.shape[0]):
        X_train.append(X1_train[i])
        Y_train.append(opt[1])
    for i in range(X2_train.shape[0]):
        X_train.append(X2_train[i])
        Y_train.append(opt[2])
    for i in range(X3_train.shape[0]):
        X_train.append(X3_train[i])
        Y_train.append(opt[3])
    for i in range(X4_train.shape[0]):
        X_train.append(X4_train[i])
        Y_train.append(opt[4])
    for i in range(X5_train.shape[0]):
        X_train.append(X5_train[i])
        Y_train.append(opt[5])
    for i in range(X6_train.shape[0]):
        X_train.append(X6_train[i])
        Y_train.append(opt[6])

    print('Generating XY-CV ......')
    for i in range(X0_CV.shape[0]):
        X_CV.append(X0_CV[i])
        Y_CV.append(opt[0])
    for i in range(X1_CV.shape[0]):
        X_CV.append(X1_CV[i])
        Y_CV.append(opt[1])
    for i in range(X2_CV.shape[0]):
        X_CV.append(X2_CV[i])
        Y_CV.append(opt[2])
    for i in range(X3_CV.shape[0]):
        X_CV.append(X3_CV[i])
        Y_CV.append(opt[3])
    for i in range(X4_CV.shape[0]):
        X_CV.append(X4_CV[i])
        Y_CV.append(opt[4])
    for i in range(X5_CV.shape[0]):
        X_CV.append(X5_CV[i])
        Y_CV.append(opt[5])
    for i in range(X6_CV.shape[0]):
        X_CV.append(X6_CV[i])
        Y_CV.append(opt[6])

    print('Generating XY-Test ......')
    for i in range(X0_test.shape[0]):
        X_test.append(X0_test[i])
        Y_test.append(opt[0])
    for i in range(X1_test.shape[0]):
        X_test.append(X1_test[i])
        Y_test.append(opt[1])
    for i in range(X2_test.shape[0]):
        X_test.append(X2_test[i])
        Y_test.append(opt[2])
    for i in range(X3_test.shape[0]):
        X_test.append(X3_test[i])
        Y_test.append(opt[3])
    for i in range(X4_test.shape[0]):
        X_test.append(X4_test[i])
        Y_test.append(opt[4])
    for i in range(X5_test.shape[0]):
        X_test.append(X5_test[i])
        Y_test.append(opt[5])
    for i in range(X6_test.shape[0]):
        X_test.append(X6_test[i])
        Y_test.append(opt[6])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_CV = np.array(X_CV)
    Y_CV = np.array(Y_CV)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    
  


print('Train Shape:')
print(X_train.shape)
print(Y_train.shape)

print('CV Shape:')
print(X_CV.shape)
print(Y_CV.shape)

print('Test Shape:')
print(X_test.shape)
print(Y_test.shape)

len_train = X_train.shape[0]
len_CV = X_CV.shape[0]
len_test = X_test.shape[0]


total_batches = int(len_train/batch_size) 


if(dump):
    print('Dumping X_train')
    pickle.dump(X_train, open('dumps/X_train.p', 'wb'))
    print('Dumping Y_train')
    pickle.dump(Y_train, open('dumps/Y_train.p', 'wb'))
    print('Dumping X_CV')
    pickle.dump(X_CV, open('dumps/X_CV.p', 'wb'))
    print('Dumping Y_CV')
    pickle.dump(Y_CV, open('dumps/Y_CV.p', 'wb'))
    print('Dumping X_test')
    pickle.dump(X_test, open('dumps/X_test.p', 'wb'))
    print('Dumping Y_test')
    pickle.dump(Y_test, open('dumps/Y_test.p', 'wb'))


if(shuffle):
    print('Shuffling Data')
    X_train, X_CNN, Y_train, Y_CNN = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_state)

print('\nCNN CV Shape:')
print(X_CNN.shape)
print(Y_CNN.shape)





##########################################################################################################################
####                                                    Model                                                         ####
##########################################################################################################################


input_shape = (224, 224, 3)

model = model_without_top(input_shape)

model.add(Flatten())

if(not os.path.isfile(filepath)):
    print('\nNo previously saved weights file found')
    print('\nLoading VGG-16 weights\n')
    model.load_weights(root_dir+'/Independent_Study/VGG16Model/vgg16_weights_without_top.h5')
    print('\nSuccessfully loaded VGG-16 weights\n')

    # model.add(Flatten())
    
    for layer in model.layers:
        layer.trainable = False

    # model.add( Dense(1024) )
    # model.add(Activation('relu') )
    # model.add( Dropout(drop_prob) )

    model.add( Dense(256) )
    model.add(Activation('relu') )
    model.add( Dropout(drop_prob) )

    model.add( Dense(len(actions)) )
    model.add( Activation('softmax') )

else:
    print('\nPreviously saved weights file found')

    # model.add(Flatten())

    for layer in model.layers:
        layer.trainable = False

    # model.add( Dense(1024) )
    # model.add(Activation('relu') )
    # model.add( Dropout(drop_prob) )

    model.add( Dense(256) )
    model.add(Activation('relu') )
    model.add( Dropout(drop_prob) )
    
    model.add( Dense(len(actions)) )
    model.add( Activation('softmax') )
    
    print('\nLoading saved weights file')
    model.load_weights(filepath) 
    print('\nSuccessfully loaded weights')



model.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\nseq: ", seq,
      "\nLearning Rate: ", learning_rate,
      "\nDropout: ", drop_prob,
      "\nDecay Rate: ", decay_rate,  
      "\nBatch Size: ", batch_size, 
      "\nTotal Batches: ", total_batches,     
      "\nEpochs: ", n_epochs,
      "\nSamples per Video: ", samples_per_video,
      "\nflip: ", flip, 
      "\ninclude_both: ", include_both)




checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, verbose=1, mode='max')
callbacks_list = [checkpoint]

hist_obj = model.fit(X_train,Y_train, validation_data=[X_CV, Y_CV], epochs=n_epochs, batch_size=batch_size, callbacks=callbacks_list)

train_loss = hist_obj.history['loss']
val_loss = hist_obj.history['val_loss']
train_acc = hist_obj.history['acc']
val_acc = hist_obj.history['val_acc']

score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)



score, acc = model.evaluate(X_CNN, Y_CNN, batch_size=batch_size)
print('CNN Test score:', score)
print('CNN Test accuracy:', acc)




