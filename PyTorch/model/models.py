import torchvision
import torch
import torch.nn as nn
import os
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, config):
        super(VGG16, self).__init__()
        self.config = config

        self.basemodel = torch.load('pretrained/vgg16-top.pth')
        print('Loaded pretrained VGG16 weights') 

        self.linear1 = nn.Linear(512*7*7, 128)
        self.drop1 = nn.Dropout(self.config['model']['dropout']) 
        self.linear2 = nn.Linear(128, 7)
        

    def forward(self, x):
        out = self.basemodel(x)
        out = out.view(out.size(0), -1)  
        out = self.drop1(out)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        
        return out   




class VGG16LRCN(nn.Module):
    def __init__(self, config):
        super(VGG16LRCN, self).__init__()
        self.config = config
        self.lstm_hidden_dim = self.config['model']['lstm_hidden_dim']
        self.num_lstm_layers = self.config['model']['num_lstm_layers']

        self.basemodel = torch.load('pretrained/vgg16-top.pth')
        print('Loaded pretrained VGG16 weights')

        self.drop1 = nn.Dropout(self.config['model']['dropout'])
        self.linear1 = nn.Linear(512*7*7, 128)

        self.lstm1 = nn.LSTM(128, self.lstm_hidden_dim, self.num_lstm_layers, batch_first=True)
        self.linear2 = nn.Linear(self.lstm_hidden_dim, 7)

        count = 0                                                                                                                              
        for param in self.parameters():                                                                                                        
           if count < 26:
               param.requires_grad = False                                                                                                    
           count += 1                                                                                                                        
        print('made VGG16 non-trainable')

    def forward(self, x):

        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size*seq_len, c, h, w)
 
        out = self.basemodel(x)

        out = out.view(out.size(0), -1)

        out = self.drop1(out)
        out = self.linear1(out)
        out = F.relu(out)
        
        out = out.view(batch_size, seq_len, -1)
        out, hs = self.lstm1(out)

        out = out[:, -1]

        out = self.linear2(out)

        return out








class AlexNet(nn.Module):
    def __init__(self, config):
        super(AlexNet, self).__init__()
        self.config = config

        self.basemodel = torch.load('pretrained/alexnet-top.pth')
        print('Loaded pretrained AlexNet weights')        

        self.linear1 = nn.Linear(256*6*6, 128)
        self.drop1 = nn.Dropout(self.config['model']['dropout']) 
        self.linear2 = nn.Linear(128, 7)
        

    def forward(self, x):
        out = self.basemodel(x)
        out = out.view(out.size(0), -1)  
        out = self.linear1(out)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.linear2(out)
        
        return out   



class ResNet18(nn.Module):
    def __init__(self, config):
        super(ResNet18, self).__init__()
        self.config = config          

        self.basemodel = torch.load('pretrained/resnet18-top.pth')
        print('Loaded pretrained ResNet18 weights') 

        self.linear1 = nn.Linear(512*1*1, 128)
        self.drop1 = nn.Dropout(self.config['model']['dropout'])
        self.linear2 = nn.Linear(128, 7)
        

    def forward(self, x):
        out = self.basemodel(x)
        out = out.view(out.size(0), -1)  
        out = self.linear1(out)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.linear2(out)
        
        return out  

