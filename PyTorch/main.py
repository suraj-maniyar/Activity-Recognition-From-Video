from model.model import ClassificationModel, LRCNModel
from utils.utils import get_config_from_json
from data_loader.data_loader import ClassificationTrainDataset, ClassificationValDataset, \
                                    LRCNTrainDataset, LRCNValDataset
from torch.utils.data import DataLoader
from trainer.trainer import trainClassification, trainLRCN
import torch
import os

def mainClassification():

    # Get configurations
    config = get_config_from_json('config/classification.config')


    # Loading Model
    model = ClassificationModel(config)

    if torch.cuda.is_available():
        model.cuda()

    model.load()

    if os.path.isfile(config['model']['checkpoint']):
        model.load_state_dict( torch.load(config['model']['checkpoint']) )
        print('Weigths loaded from checkpoint : %s' % config['model']['checkpoint'])

    # Loading Dataset
    train_dataset = ClassificationTrainDataset(config)
    val_dataset = ClassificationValDataset(config)

    # Creating Data Loader
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = config['model']['batch_size'])

    val_loader = DataLoader(dataset = val_dataset,
                            batch_size = config['model']['batch_size'])

    # Train model
    trainClassification(config, model, train_loader, val_loader)




def mainLRCN():

    # Get configurations
    config = get_config_from_json('config/lrcn.config')


    # Loading Model
    model = LRCNModel(config)
    model = model.double()

    if torch.cuda.is_available():
        model.cuda()

    model.load()

    print(model)

    if os.path.isfile(config['model']['checkpoint']):
        model.load_state_dict( torch.load(config['model']['checkpoint']) )
        print('Weigths loaded from checkpoint : %s' % config['model']['checkpoint'])

    # Loading Dataset
    train_dataset = LRCNTrainDataset(config)
    val_dataset = LRCNValDataset(config)

    # Creating Data Loader
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = config['model']['batch_size'],
                              num_workers=1)

    val_loader = DataLoader(dataset = val_dataset,
                            batch_size = config['model']['batch_size'],
                            num_workers=1)

    # Train model
    trainLRCN(config, model, train_loader, val_loader)


if __name__ == "__main__":
    mainLRCN()



import matplotlib.pyplot as plt
import  numpy as np

# Get configurations
config = get_config_from_json('config/classification.config')

# Loading Dataset
train_dataset = ClassificationTrainDataset(config)
val_dataset = ClassificationValDataset(config)

# Creating Data Loader
train_loader = DataLoader(dataset = train_dataset,
                          batch_size = config['model']['batch_size'])
val_loader = DataLoader(dataset = val_dataset,
                        batch_size = config['model']['batch_size'])

it = iter(train_loader)
x, y = next(it)

def display(x, y, index):
    x = x.numpy()
    x = np.moveaxis(x, 1, 3)
    print(y[index])
    plt.imshow(x[index])
    plt.show()
    
    return x[index]
