from torch.utils.data import Dataset
import numpy as np
import skimage.io
import os, glob, random
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

img_size = 224

class ClassificationTrainDataset(Dataset):
    def __init__(self, config, transform=None):

        self.config = config
        self.batch_size = self.config['model']['batch_size']

        self.path_action0, _, self.files_action0 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action0') ))
        self.path_action1, _, self.files_action1 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action1') ))
        self.path_action2, _, self.files_action2 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action2') ))
        self.path_action3, _, self.files_action3 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action3') ))
        self.path_action4, _, self.files_action4 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action4') ))
        self.path_action5, _, self.files_action5 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action5') ))
        self.path_action6, _, self.files_action6 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action6') ))

        self.paths_train0 = glob.glob( os.path.join(self.path_action0, '*') )
        self.paths_train1 = glob.glob( os.path.join(self.path_action1, '*') )
        self.paths_train2 = glob.glob( os.path.join(self.path_action2, '*') )
        self.paths_train3 = glob.glob( os.path.join(self.path_action3, '*') )
        self.paths_train4 = glob.glob( os.path.join(self.path_action4, '*') )
        self.paths_train5 = glob.glob( os.path.join(self.path_action5, '*') )
        self.paths_train6 = glob.glob( os.path.join(self.path_action6, '*') )

        self.train_paths = self.paths_train0 + self.paths_train1 + self.paths_train2 + self.paths_train3 + \
                           self.paths_train4 + self.paths_train5 + self.paths_train6

        self.train_outputs = [-1] * len(self.train_paths)

        print('train_paths : ', len(self.train_paths))

        print('Generating output')
        for i in range(len(self.train_paths)):
            if "action0" in self.train_paths[i]:
                self.train_outputs[i] = 0
            elif "action1" in self.train_paths[i]:
                self.train_outputs[i] = 1
            elif "action2" in self.train_paths[i]:
                self.train_outputs[i] = 2
            elif "action3" in self.train_paths[i]:
                self.train_outputs[i] = 3
            elif "action4" in self.train_paths[i]:
                self.train_outputs[i] = 4
            elif "action5" in self.train_paths[i]:
                self.train_outputs[i] = 5
            elif "action6" in self.train_paths[i]:
                self.train_outputs[i] = 6

        print('Shuffling')
        z = list(zip(self.train_paths, self.train_outputs))
        random.shuffle(z)
        self.train_paths, self.train_outputs = zip(*z)

        if transform == None:
            self.transform = transforms.Compose([
                             transforms.Resize(( img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                             ])
        else:
            self.transform = transform


    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, idx):

        img = skimage.io.imread(self.train_paths[idx])
        img = F.to_pil_image(img)
        img = self.transform(img)
        label = self.train_outputs[idx]

        return (img, label)






class ClassificationValDataset(Dataset):
    def __init__(self, config, transform=None):

        self.config = config
        self.batch_size = self.config['model']['batch_size']

        self.path_action0, _, self.files_action0 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action0') ))
        self.path_action1, _, self.files_action1 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action1') ))
        self.path_action2, _, self.files_action2 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action2') ))
        self.path_action3, _, self.files_action3 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action3') ))
        self.path_action4, _, self.files_action4 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action4') ))
        self.path_action5, _, self.files_action5 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action5') ))
        self.path_action6, _, self.files_action6 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action6') ))

        self.paths_val0 = glob.glob( os.path.join(self.path_action0, '*') )
        self.paths_val1 = glob.glob( os.path.join(self.path_action1, '*') )
        self.paths_val2 = glob.glob( os.path.join(self.path_action2, '*') )
        self.paths_val3 = glob.glob( os.path.join(self.path_action3, '*') )
        self.paths_val4 = glob.glob( os.path.join(self.path_action4, '*') )
        self.paths_val5 = glob.glob( os.path.join(self.path_action5, '*') )
        self.paths_val6 = glob.glob( os.path.join(self.path_action6, '*') )

        self.val_paths = self.paths_val0 + self.paths_val1 + self.paths_val2 + self.paths_val3 + \
                           self.paths_val4 + self.paths_val5 + self.paths_val6

        self.val_outputs = [-1] * len(self.val_paths)

        print('val_paths : ', len(self.val_paths))

        print('Generating output')
        for i in range(len(self.val_paths)):
            if "action0" in self.val_paths[i]:
                self.val_outputs[i] = 0
            elif "action1" in self.val_paths[i]:
                self.val_outputs[i] = 1
            elif "action2" in self.val_paths[i]:
                self.val_outputs[i] = 2
            elif "action3" in self.val_paths[i]:
                self.val_outputs[i] = 3
            elif "action4" in self.val_paths[i]:
                self.val_outputs[i] = 4
            elif "action5" in self.val_paths[i]:
                self.val_outputs[i] = 5
            elif "action6" in self.val_paths[i]:
                self.val_outputs[i] = 6

        if transform == None:
            self.transform = transforms.Compose([
                             transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                             ])
        else:
            self.transform = transform


    def __len__(self):
        return len(self.val_paths)

    def __getitem__(self, idx):

        img = skimage.io.imread(self.val_paths[idx])
        img = F.to_pil_image(img)
        img = self.transform(img)
        label = self.val_outputs[idx]

        return (img, label)




class ClassificationTestDataset(Dataset):
    def __init__(self, config, transform=None):

        self.config = config
        self.batch_size = self.config['model']['batch_size']

        self.path_action0, _, self.files_action0 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action0') ))
        self.path_action1, _, self.files_action1 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action1') ))
        self.path_action2, _, self.files_action2 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action2') ))
        self.path_action3, _, self.files_action3 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action3') ))
        self.path_action4, _, self.files_action4 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action4') ))
        self.path_action5, _, self.files_action5 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action5') ))
        self.path_action6, _, self.files_action6 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action6') ))

        self.paths_test0 = glob.glob( os.path.join(self.path_action0, '*') )
        self.paths_test1 = glob.glob( os.path.join(self.path_action1, '*') )
        self.paths_test2 = glob.glob( os.path.join(self.path_action2, '*') )
        self.paths_test3 = glob.glob( os.path.join(self.path_action3, '*') )
        self.paths_test4 = glob.glob( os.path.join(self.path_action4, '*') )
        self.paths_test5 = glob.glob( os.path.join(self.path_action5, '*') )
        self.paths_test6 = glob.glob( os.path.join(self.path_action6, '*') )

        self.test_paths = self.paths_test0 + self.paths_test1 + self.paths_test2 + self.paths_test3 + \
                           self.paths_test4 + self.paths_test5 + self.paths_test6

        self.test_outputs = [-1] * len(self.test_paths)

        print('test_paths : ', len(self.test_paths))

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

        if transform == None:
            self.transform = transforms.Compose([
                             transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                             ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.test_paths)

    def __getitem__(self, idx):

        img = skimage.io.imread(self.test_paths[idx])
        img = F.to_pil_image(img)
        img = self.transform(img)
        label = self.test_outputs[idx]

        return (img, label)







class LRCNTrainDataset(Dataset):
    def __init__(self, config, transform=None):

        self.config = config
        self.batch_size = self.config['model']['batch_size']
        self.seq_len = self.config['model']['seq_len']

        self.path_action0, _, self.files_action0 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action0') ))
        self.path_action1, _, self.files_action1 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action1') ))
        self.path_action2, _, self.files_action2 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action2') ))
        self.path_action3, _, self.files_action3 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action3') ))
        self.path_action4, _, self.files_action4 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action4') ))
        self.path_action5, _, self.files_action5 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action5') ))
        self.path_action6, _, self.files_action6 = next(os.walk( os.path.join(self.config['data_path'], 'train', 'action6') ))

        self.paths_train0 = glob.glob( os.path.join(self.path_action0, '*') )
        self.paths_train1 = glob.glob( os.path.join(self.path_action1, '*') )
        self.paths_train2 = glob.glob( os.path.join(self.path_action2, '*') )
        self.paths_train3 = glob.glob( os.path.join(self.path_action3, '*') )
        self.paths_train4 = glob.glob( os.path.join(self.path_action4, '*') )
        self.paths_train5 = glob.glob( os.path.join(self.path_action5, '*') )
        self.paths_train6 = glob.glob( os.path.join(self.path_action6, '*') )

        print('Action0 : ', len(self.paths_train0))
        print('Action1 : ', len(self.paths_train1))
        print('Action2 : ', len(self.paths_train2))
        print('Action3 : ', len(self.paths_train3))
        print('Action4 : ', len(self.paths_train4))
        print('Action5 : ', len(self.paths_train5))
        print('Action6 : ', len(self.paths_train6))

        self.train_paths = self.paths_train0 + self.paths_train1 + self.paths_train2 + self.paths_train3 + \
                           self.paths_train4 + self.paths_train5 + self.paths_train6


        print('Sorting')
        self.train_paths = sorted(self.train_paths)

        self.train_outputs = [-1] * len(self.train_paths)

        print('Generating output')
        for i in range(len(self.train_paths)):
            if "action0" in self.train_paths[i]:
                self.train_outputs[i] = 0
            elif "action1" in self.train_paths[i]:
                self.train_outputs[i] = 1
            elif "action2" in self.train_paths[i]:
                self.train_outputs[i] = 2
            elif "action3" in self.train_paths[i]:
                self.train_outputs[i] = 3
            elif "action4" in self.train_paths[i]:
                self.train_outputs[i] = 4
            elif "action5" in self.train_paths[i]:
                self.train_outputs[i] = 5
            elif "action6" in self.train_paths[i]:
                self.train_outputs[i] = 6



        paths, outputs = [], []
        for i in range(len(self.train_paths) // self.seq_len):
            paths.append( self.train_paths[i*self.seq_len : (i+1)*self.seq_len] )
            op = self.train_outputs[ i*self.seq_len : (i+1)*self.seq_len ]
            outputs.append( max(set(op), key=op.count) )

        self.train_paths = paths
        self.train_outputs = outputs


        print('train_paths : ', len(self.train_paths))

        print('Shuffling')
        z = list(zip(self.train_paths, self.train_outputs))
        random.shuffle(z)
        self.train_paths, self.train_outputs = zip(*z)

        if transform == None:
            self.transform = transforms.Compose([
                             transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                             ])
        else:
            self.transform = transform


    def __len__(self):
        return len(self.train_paths)


    def __getitem__(self, idx):

        X = np.zeros((self.seq_len, 3, img_size, img_size))
        for i in range(self.seq_len):
            img = skimage.io.imread(self.train_paths[idx][i])
            img = F.to_pil_image(img)
            X[i] = self.transform(img)

        label = self.train_outputs[idx]

        return (X, label)







class LRCNValDataset(Dataset):
    def __init__(self, config, transform=None):

        self.config = config
        self.batch_size = self.config['model']['batch_size']
        self.seq_len = self.config['model']['seq_len']

        self.path_action0, _, self.files_action0 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action0') ))
        self.path_action1, _, self.files_action1 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action1') ))
        self.path_action2, _, self.files_action2 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action2') ))
        self.path_action3, _, self.files_action3 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action3') ))
        self.path_action4, _, self.files_action4 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action4') ))
        self.path_action5, _, self.files_action5 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action5') ))
        self.path_action6, _, self.files_action6 = next(os.walk( os.path.join(self.config['data_path'], 'val', 'action6') ))

        self.paths_val0 = glob.glob( os.path.join(self.path_action0, '*') )
        self.paths_val1 = glob.glob( os.path.join(self.path_action1, '*') )
        self.paths_val2 = glob.glob( os.path.join(self.path_action2, '*') )
        self.paths_val3 = glob.glob( os.path.join(self.path_action3, '*') )
        self.paths_val4 = glob.glob( os.path.join(self.path_action4, '*') )
        self.paths_val5 = glob.glob( os.path.join(self.path_action5, '*') )
        self.paths_val6 = glob.glob( os.path.join(self.path_action6, '*') )

        self.val_paths = self.paths_val0 + self.paths_val1 + self.paths_val2 + self.paths_val3 + \
                           self.paths_val4 + self.paths_val5 + self.paths_val6

        print('Sorting')
        self.val_paths = sorted(self.val_paths)

        self.val_outputs = [-1] * len(self.val_paths)

        print('Generating output')
        for i in range(len(self.val_paths)):
            if "action0" in self.val_paths[i]:
                self.val_outputs[i] = 0
            elif "action1" in self.val_paths[i]:
                self.val_outputs[i] = 1
            elif "action2" in self.val_paths[i]:
                self.val_outputs[i] = 2
            elif "action3" in self.val_paths[i]:
                self.val_outputs[i] = 3
            elif "action4" in self.val_paths[i]:
                self.val_outputs[i] = 4
            elif "action5" in self.val_paths[i]:
                self.val_outputs[i] = 5
            elif "action6" in self.val_paths[i]:
                self.val_outputs[i] = 6


        paths, outputs = [], []
        for i in range(len(self.val_paths) // self.seq_len):
            paths.append( self.val_paths[i*self.seq_len : (i+1)*self.seq_len] )
            op = self.val_outputs[ i*self.seq_len : (i+1)*self.seq_len ]
            outputs.append( max(set(op), key=op.count) )

        self.val_paths = paths
        self.val_outputs = outputs


        print('val_paths : ', len(self.val_paths))



        if transform == None:
            self.transform = transforms.Compose([
                             transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                             ])
        else:
            self.transform = transform


    def __len__(self):
        return len(self.val_paths)

    def __getitem__(self, idx):

        X = np.zeros((self.seq_len, 3, img_size, img_size))
        for i in range(self.seq_len):
            img = skimage.io.imread(self.val_paths[idx][i])
            img = F.to_pil_image(img)
            X[i] = self.transform(img)

        label = self.val_outputs[idx]

        return (X, label)





class LRCNTestDataset(Dataset):
    def __init__(self, config, transform=None):

        self.config = config
        self.batch_size = self.config['model']['batch_size']
        self.seq_len = self.config['model']['seq_len']

        self.path_action0, _, self.files_action0 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action0') ))
        self.path_action1, _, self.files_action1 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action1') ))
        self.path_action2, _, self.files_action2 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action2') ))
        self.path_action3, _, self.files_action3 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action3') ))
        self.path_action4, _, self.files_action4 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action4') ))
        self.path_action5, _, self.files_action5 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action5') ))
        self.path_action6, _, self.files_action6 = next(os.walk( os.path.join(self.config['data_path'], 'test', 'action6') ))

        self.paths_test0 = glob.glob( os.path.join(self.path_action0, '*') )
        self.paths_test1 = glob.glob( os.path.join(self.path_action1, '*') )
        self.paths_test2 = glob.glob( os.path.join(self.path_action2, '*') )
        self.paths_test3 = glob.glob( os.path.join(self.path_action3, '*') )
        self.paths_test4 = glob.glob( os.path.join(self.path_action4, '*') )
        self.paths_test5 = glob.glob( os.path.join(self.path_action5, '*') )
        self.paths_test6 = glob.glob( os.path.join(self.path_action6, '*') )

        self.test_paths = self.paths_test0 + self.paths_test1 + self.paths_test2 + self.paths_test3 + \
                           self.paths_test4 + self.paths_test5 + self.paths_test6

        print('Sorting')
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


        paths, outputs = [], []
        for i in range(len(self.test_paths) // self.seq_len):
            paths.append( self.test_paths[i*self.seq_len : (i+1)*self.seq_len] )
            op = self.test_outputs[ i*self.seq_len : (i+1)*self.seq_len ]
            outputs.append( max(set(op), key=op.count) )

        self.test_paths = paths
        self.test_outputs = outputs


        print('test_paths : ', len(self.test_paths))


        print('Shuffling')
        z = list(zip(self.test_paths, self.test_outputs))
        random.shuffle(z)
        self.test_paths, self.test_outputs = zip(*z)



        if transform == None:
            self.transform = transforms.Compose([
                             transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                             ])
        else:
            self.transform = transform


    def __len__(self):
        return len(self.test_paths)


    def __getitem__(self, idx):

        X = np.zeros((self.seq_len, 3, img_size, img_size))
        for i in range(self.seq_len):
            img = skimage.io.imread(self.test_paths[idx][i])
            img = F.to_pil_image(img)
            X[i] = self.transform(img)

        label = self.test_outputs[idx]

        return (X, label)
