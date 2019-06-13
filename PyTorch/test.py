from model.models import VGG16, VGG16LRCN
from utils.utils import get_config_from_json
from data_loader.data_loader import ClassificationTestDataset, LRCNTestDataset
from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch
import torch.nn as nn
import os



def testClassification():

    # Get configurations
    config = get_config_from_json('config/classification.config')


    # Loading Model
    model = VGG16(config)

    if torch.cuda.is_available():
        model = model.cuda()

    if os.path.isfile(config['model']['checkpoint']):
        model.load_state_dict( torch.load(config['model']['checkpoint']) )
        print('Weigths loaded from checkpoint : %s' % config['model']['checkpoint'])
    else:
        print('############### No checkpoint found ####################')

    # Loading Dataset
    test_dataset = ClassificationTestDataset(config)

    # Creating Data Loader
    test_loader = DataLoader(dataset = test_dataset,
                             batch_size = config['model']['batch_size'],
                              num_workers=4   )


    criterion = nn.CrossEntropyLoss()

    model.eval()
    y_pred, y_true = [], []
    loss_test = 0

    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())

        if i%5 == 0:
            correct = sum( [y_true[i]==y_pred[i] for i in range(len(y_true))] )
            total = len(y_true)
            print(correct, total)
            print(i, '  ACC : ', 100.0*correct/total)
        test_loss = criterion(outputs, labels.long())

        loss_test += test_loss.item()

    print('\nAcc : ', 100.0*float(correct)/total)
    print('test loss : ', loss_test/len(test_loader))



def testLRCN():

    # Get configurations
    config = get_config_from_json('config/lrcn.config')


    # Loading Model
    model = VGG16LRCN(config)
    model = model.double()

    #if torch.cuda.is_available():
    #    model = model.cuda()

    if os.path.isfile(config['model']['checkpoint']):
        model.load_state_dict( torch.load(config['model']['checkpoint'], map_location=lambda storage, loc: storage) )
        print('Weigths loaded from checkpoint : %s' % config['model']['checkpoint'])
    else:
        print('############### No checkpoint found ####################')


    # Loading Dataset
    test_dataset = LRCNTestDataset(config)

    # Creating Data Loader
    test_loader = DataLoader(dataset = test_dataset,
                             batch_size = config['model']['batch_size'])


    criterion = nn.CrossEntropyLoss()

    model.eval()
    y_pred, y_true = [], []
    loss_test = 0

    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        outputs = model(images)
        

        _, predicted = torch.max(outputs.data, 1)

        
        print(predicted.tolist())
        print(labels.tolist())
        print('-'*80)
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())

        if i%5 == 0:
            correct = sum( [y_true[i]==y_pred[i] for i in range(len(y_true))] )
            total = len(y_true)
            print(correct, total)
            print(i, '  ACC : ', 100.0*correct/total)

        test_loss = criterion(outputs, labels.long())

        loss_test += test_loss.item()

    print('\nACC : ', 100.0*float(correct)/total)
    print('test loss : ', loss_test/len(test_loader))



if __name__ == "__main__":
    testLRCN()


