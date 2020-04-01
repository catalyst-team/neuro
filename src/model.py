
import torch
import torch.nn as nn
from catalyst.dl import registry


@registry.Model
class MeshNet(nn.Module):
    """
    MeshNet Neural Network
    Arguments:
        config: config of the neural network
        bn_before: apply batch normalization before activation function
        weight_initilization: weight intilization type
    """
 # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
    def __init__(self, weight_initilization='xavier_uniform'):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels=1, kernel_size=3, out_channels=71, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(71),
            nn.Conv3d(in_channels=71, kernel_size=3, out_channels=71, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(71),
            nn.Conv3d(in_channels=71, kernel_size=3, out_channels=71, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(71),
            nn.Conv3d(in_channels=71, kernel_size=3, out_channels=71, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(71),
            nn.Conv3d(in_channels=71, kernel_size=3, out_channels=71, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(71),
            nn.Conv3d(in_channels=71, kernel_size=3, out_channels=71, padding=8, dilation=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(71),
            nn.Conv3d(in_channels=71, kernel_size=3, out_channels=71, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(71),
            nn.Conv3d(in_channels=71, kernel_size=1, out_channels=1, padding=0, dilation=1),
        )

        # weight initilization
        if weight_initilization == 'xavier_uniform':
            for m in self.model.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.constant_(m.bias, 0.)
        elif weight_initilization == 'xavier_normal':
            for m in self.model.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.constant_(m.bias, 0.)
        elif weight_initilization == 'identity':
            for m in self.model.modules():
                if isinstance(m, nn.Conv3d):
                    temp = torch.FloatTensor(m.weight.size())
                    nn.init.xavier_uniform_(temp, gain=nn.init.calculate_gain('relu'))
                    temp[:, :, 0, 0, 0] += 1
                    m.weight = torch.nn.Parameter(temp)
                    nn.init.constant_(m.bias, 0.)
        elif weight_initilization == 'kaiming_uniform':
            for m in self.model.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    nn.init.constant_(m.bias, 0.)
        elif weight_initilization == 'kaiming_normal':
            for m in self.model.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.constant_(m.bias, 0.)
        else:
            assert False, '{} initilization isn\'t defined'.format(weight_initilization)



    def forward(self, x):
        """
        Forward propagation.
        Arguments:
            x: input
        """
        x = self.model(x)
        return x
