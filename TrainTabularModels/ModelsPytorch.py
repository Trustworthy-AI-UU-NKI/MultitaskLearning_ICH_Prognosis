
import torch
import torch.nn as nn
import functools
import operator

class TabularModel(nn.Module):
    def __init__(self, input_size, out_channels=1, threshold=0.5):
        super(TabularModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            # nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16, out_channels)
        )
        
    def forward(self, inputs):
        # Flatten before passing it to fully connected layers
        x = inputs.view(inputs.size(0), -1)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x.view(-1, 1).squeeze(1)
    
    def predict(self, inputs):
        # Apply softmax to output. 
        # pred = F.softmax(self.forward(inputs), dim=1)
        pred = self.forward(inputs)
        pred = torch.Sigmoid()(pred)
        ans = (pred > self.threshold).int()
        return ans