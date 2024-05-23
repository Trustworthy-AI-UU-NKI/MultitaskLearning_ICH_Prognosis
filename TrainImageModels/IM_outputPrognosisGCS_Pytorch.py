import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
import functools
import operator

class PrognosisICH_GCS_Model(nn.Module):
    def __init__(self, image_shape, depth, spatial_dims=3, in_channels=1, num_classes_binary=1, num_classes_ordinal=13, dropout_prob=0.0):
        super(PrognosisICH_GCS_Model, self).__init__()
        input_dim = (in_channels, depth, image_shape, image_shape)
        # Initialize the DenseNet121 without its final layer
        self.feature_extractor = DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=14, dropout_prob=dropout_prob, pretrained=False)
        # Remove the last linear layer
        self.feature_extractor.class_layers = nn.Identity()
        self.num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape))
        
        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_binary),
        )
        
        # Ordinal regression head
        self.ordinal_head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_ordinal),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        binary_output = self.binary_head(features)
        ordinal_output = self.ordinal_head(features)
        return binary_output, ordinal_output
    
class PrognosisICH_BinaryGCS_Model(nn.Module):
    def __init__(self, image_shape, depth, spatial_dims=3, in_channels=1, num_classes_binary=1, num_classes_ordinal=1, dropout_prob=0.0):
        super(PrognosisICH_BinaryGCS_Model, self).__init__()
        input_dim = (in_channels, depth, image_shape, image_shape)
        # Initialize the DenseNet121 without its final layer
        self.feature_extractor = DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=14, dropout_prob=dropout_prob, pretrained=False)
        # Remove the last linear layer
        self.feature_extractor.class_layers = nn.Identity()

        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_binary),
        )
        
        # Ordinal regression head
        self.ordinal_head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_ordinal),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        binary_output = self.binary_head(features)
        ordinal_output = self.ordinal_head(features)
        return binary_output, ordinal_output ##### only return binary output for medcam
    
class PrognosisICH_BinaryGCS_ModelConstrainedNonNegativeWeights(nn.Module):
    def __init__(self, image_shape, depth, spatial_dims=3, in_channels=1, num_classes_binary=1, num_classes_ordinal=1, dropout_prob=0.0):
        super(PrognosisICH_BinaryGCS_ModelConstrainedNonNegativeWeights, self).__init__()
        input_dim = (in_channels, depth, image_shape, image_shape)
        # Initialize the DenseNet121 without its final layer
        self.feature_extractor = DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=14, dropout_prob=dropout_prob, pretrained=False)
        # Remove the last linear layer
        self.feature_extractor.class_layers = nn.Identity()

        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.NonNegLinear(1024, num_classes_binary),
        )
        
        # Ordinal regression head
        self.ordinal_head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.NonNegLinear(1024, num_classes_ordinal),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        binary_output = self.binary_head(features)
        ordinal_output = self.ordinal_head(features)
        return binary_output, ordinal_output
    
class NonNegLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight.clamp(min=0.), self.bias.clamp(min=0.))


    
class PrognosisICH_ThreeClassGCS_Model(nn.Module):
    def __init__(self, image_shape, depth, spatial_dims=3, in_channels=1, num_classes_binary=1, num_classes_ordinal=2, dropout_prob=0.0):
        super(PrognosisICH_ThreeClassGCS_Model, self).__init__()
        input_dim = (in_channels, depth, image_shape, image_shape)
        # Initialize the DenseNet121 without its final layer
        self.feature_extractor = DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=14, dropout_prob=dropout_prob, pretrained=False)
        # Remove the last linear layer
        self.feature_extractor.class_layers = nn.Identity()
        self.num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, *input_dim)).shape))
        
        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_binary),
        )
        
        # Ordinal regression head
        self.ordinal_head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_ordinal),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        binary_output = self.binary_head(features)
        ordinal_output = self.ordinal_head(features)
        return binary_output, ordinal_output
    
class PrognosisICH_BinaryGCSBinaryAge_Model(nn.Module):
    def __init__(self, image_shape, depth, spatial_dims=3, in_channels=1, num_classes_binary=1, num_classes_ordinalGCS=1, num_classes_ordinalAge=1, dropout_prob=0.0, saliency_maps=None):
        super(PrognosisICH_BinaryGCSBinaryAge_Model, self).__init__()
        input_dim = (in_channels, depth, image_shape, image_shape)
        self.saliency_maps = saliency_maps
        # Initialize the DenseNet121 without its final layer
        self.feature_extractor = DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=1, dropout_prob=dropout_prob, pretrained=False)
        # Remove the last linear layer
        self.feature_extractor.class_layers = nn.Identity()

        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_binary),
        )
        
        # Ordinal regression head GCS
        self.ordinal_headGCS = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_ordinalGCS),
        )

        # Ordinal regression head Age
        self.ordinal_headAge = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_ordinalAge),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        binary_output = self.binary_head(features)
        ordinal_outputGCS = self.ordinal_headGCS(features)
        ordinal_outputAge = self.ordinal_headAge(features)
        if self.saliency_maps == 'Prognosis':
            return binary_output
        elif self.saliency_maps == 'GCS':
            return ordinal_outputGCS
        elif self.saliency_maps == 'Age':
            return ordinal_outputAge
        else:
            return binary_output, ordinal_outputGCS, ordinal_outputAge
class PrognosisICH_BinaryAge_Model(nn.Module):
    def __init__(self, image_shape, depth, spatial_dims=3, in_channels=1, num_classes_binary=1, num_classes_ordinalAge=1, dropout_prob=0.0, saliency_maps=None):
        super(PrognosisICH_BinaryAge_Model, self).__init__()
        input_dim = (in_channels, depth, image_shape, image_shape)
        self.saliency_maps = saliency_maps
        # Initialize the DenseNet121 without its final layer
        self.feature_extractor = DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=1, dropout_prob=dropout_prob, pretrained=False)
        # Remove the last linear layer
        self.feature_extractor.class_layers = nn.Identity()

        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_binary),
        )

        # Ordinal regression head Age
        self.ordinal_headAge = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_ordinalAge),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        binary_output = self.binary_head(features)
        ordinal_outputAge = self.ordinal_headAge(features)
        if self.saliency_maps == 'Prognosis':
            return binary_output
        elif self.saliency_maps == 'Age':
            return ordinal_outputAge
        else:
            return binary_output, ordinal_outputAge
        
class PrognosisICH_ThreeClassGCSBinaryAge_Model(nn.Module):
    def __init__(self, image_shape, depth, spatial_dims=3, in_channels=1, num_classes_binary=1, num_classes_ordinalGCS=2, num_classes_ordinalAge=1, dropout_prob=0.0, saliency_maps=None):
        super(PrognosisICH_ThreeClassGCSBinaryAge_Model, self).__init__()
        input_dim = (in_channels, depth, image_shape, image_shape)
        self.saliency_maps = saliency_maps
        # Initialize the DenseNet121 without its final layer
        self.feature_extractor = DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=1, dropout_prob=dropout_prob, pretrained=False)
        # Remove the last linear layer
        self.feature_extractor.class_layers = nn.Identity()

        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_binary),
        )
        
        # Ordinal regression head GCS
        self.ordinal_headGCS = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_ordinalGCS),
        )

        # Ordinal regression head Age
        self.ordinal_headAge = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_ordinalAge),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        binary_output = self.binary_head(features)
        ordinal_outputGCS = self.ordinal_headGCS(features)
        ordinal_outputAge = self.ordinal_headAge(features)
        if self.saliency_maps == 'Prognosis':
            return binary_output
        elif self.saliency_maps == 'GCS':
            return ordinal_outputGCS
        elif self.saliency_maps == 'Age':
            return ordinal_outputAge
        else:
            return binary_output, ordinal_outputGCS, ordinal_outputAge
        
class BinaryGCS_Model(nn.Module):
    def __init__(self, image_shape, depth, spatial_dims=3, in_channels=1, num_classes_ordinalGCS=1, dropout_prob=0.0):
        super(BinaryGCS_Model, self).__init__()
        input_dim = (in_channels, depth, image_shape, image_shape)
        # Initialize the DenseNet121 without its final layer
        self.feature_extractor = DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=1, dropout_prob=dropout_prob, pretrained=False)
        # Remove the last linear layer
        self.feature_extractor.class_layers = nn.Identity()
        
        # Ordinal regression head GCS
        self.ordinal_headGCS = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_ordinalGCS),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        ordinal_outputGCS = self.ordinal_headGCS(features)

        return ordinal_outputGCS
    
class BinaryAge_Model(nn.Module):
    def __init__(self, image_shape, depth, spatial_dims=3, in_channels=1, num_classes_ordinalAge=1, dropout_prob=0.0):
        super(BinaryAge_Model, self).__init__()
        input_dim = (in_channels, depth, image_shape, image_shape)
        # Initialize the DenseNet121 without its final layer
        self.feature_extractor = DenseNet121(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=1, dropout_prob=dropout_prob, pretrained=False)
        # Remove the last linear layer
        self.feature_extractor.class_layers = nn.Identity()
        
        # Ordinal regression head GCS
        self.ordinal_headAge = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes_ordinalAge),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        ordinal_outputAge = self.ordinal_headAge(features)

        return ordinal_outputAge