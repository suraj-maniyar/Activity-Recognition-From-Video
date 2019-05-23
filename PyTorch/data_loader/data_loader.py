from torch.utils.data import Dataset
import numpy as np
import skimage.io
import os, glob, random
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


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
                             transforms.Resize((224, 224)),
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
        img = np.moveaxis(img, 0, 2)
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
                             transforms.Resize((224, 224)),
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
        img = np.moveaxis(img, 0, 2)
        img = F.to_pil_image(img)
        img = self.transform(img)
        label = self.val_outputs[idx]

        return (img, label)
