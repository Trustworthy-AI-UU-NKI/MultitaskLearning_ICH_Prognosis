import torch
import torch.nn as nn
import functools
import operator
from sklearn.decomposition import PCA
import torch.nn.functional as F

class AmaiaModel(nn.Module):
    def __init__(self, depth=40, width=301, height=301, in_channels=1, out_channels=1, initializer_seed=1):
        super(AmaiaModel, self).__init__()
        input_dim = (in_channels, depth, width, height)
        self.initializer_seed = initializer_seed
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels=16, kernel_size=3, padding=0, stride = 1, bias=False),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Dropout3d(p=0.5),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
    
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.AdaptiveMaxPool3d(1)
        )
        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape))
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_features=num_features_before_fcnn, out_features=out_channels)
        )
        self._initialize_weights()
    def _initialize_weights(self):
        torch.manual_seed(self.initializer_seed)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x.view(-1, 1)