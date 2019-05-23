import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationModel(nn.Module):

    def __init__(self, config):
        super(ClassificationModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self, n_classes=7):

        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        self.drop1 = nn.Dropout(self.config['model']['dropout'])
        self.linear1 = nn.Linear(512*14*14, 128)
        self.linear2 = nn.Linear(128, n_classes)

     

    def forward(self, x):

        out = F.relu(self.conv1_1(x))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out)

        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)

        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        conv4_3_feats = out
        out = self.pool4(out)

        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)

        out = out.view(out.size(0), -1)

        out = self.drop1(out)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)

        return out


    def load(self, path='checkpoint/vgg16-top.pth'):

        # Current state Base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG Base
        pretrained_state_dict = torch.load(path)
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i in range(26):
            state_dict[param_names[i]] = pretrained_state_dict[ pretrained_param_names[i] ]

        print('Loaded VGG16 pretrained base weights')
