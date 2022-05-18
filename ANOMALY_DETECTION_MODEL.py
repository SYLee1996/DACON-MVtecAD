import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class Network(torch.nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.model = config['model']
        self.class_num = config['class_num']
        self.pretrain = config['pretrain']
        
        if self.model == 'efficientnet_b7':
            self.model =  EfficientNet.from_pretrained('efficientnet-b7', num_classes=self.class_num)
            
    def forward(self, x):
        x = self.model(x)
        return x