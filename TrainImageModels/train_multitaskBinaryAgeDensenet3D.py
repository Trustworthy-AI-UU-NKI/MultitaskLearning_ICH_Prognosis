import logging
import os
import sys
import shutil
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
import torch.nn as nn
import functools
import operator
from sklearn.model_selection import StratifiedKFold

from bootstrapDEF import bootstrap_metric_ci
from torchsummary import summary

from IM_outputPrognosisGCS_Pytorch import PrognosisICH_BinaryAge_Model

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
import math
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, Dataset, CacheDataset
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    DivisiblePadd,
    Flipd,
    ThresholdIntensityd,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    SpatialPadd,
    Compose,
    Rotate90d,
    RandRotate90,
    Resized,
    ScaleIntensity,
    Transposed,
    AsDiscrete,
    Activations,
    Transpose,
    RandRotate90d,
    RandGaussianNoised,
    Rand3DElasticd, 
    RandRotated,
    RandAffined,
    Zoomd
)
from evaluate_thresholds import EvaluateThresholds
import torch.nn.functional as F

import json

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

# Load the config file
config = load_config('/home/ubuntu/tenerife/miriam/MultitaskLearning_ICH_Prognosis/config.json')

# Access the variables from the config dictionary
path_to_save_model_dir = config["path_to_save_model_dir"]
path_to_save_results = config["path_to_save_results"]
name_file = config["name_file"]

np.set_printoptions(precision=3)

# path_to_save_model_dir = "/home/ubuntu/tenerife/data/ZZ_ICH_PrognosisMICCAI/Models/BinaryGCS_BinaryAge_Prognosis"
# path_to_save_results = '/home/ubuntu/tenerife/data/ZZ_ICH_PrognosisMICCAI/Results/BinaryAge_Prognosis'
# name_file = "BinaryAge_Prognosis_MulticlassOutput"

def ordinal_encode(y, num_classes):
    """
    Encode labels in a custom ordinal fashion follwing this approach:

        class 1 is represented as [0 0 0 0 ...]

        class 2 is represented as [1 0 0 0 ...]

        class 3 is represented as [1 1 0 0 ...]

        etc.
    
    Args:
    - y (array-like): Array of class labels (integers starting from 0).
    - num_classes (int): Total number of classes.
    
    Returns:
    - np.ndarray: Ordinally encoded labels.
    """
    # Initialize the encoded labels array
    encoded_labels = np.zeros((len(y), num_classes), dtype=int)
    
    # Encode each label
    for i, label in enumerate(y):
        if label > 0:  # No encoding needed for class 0, as it's already [0, 0, 0]
            encoded_labels[i, :label] = 1
            
    return encoded_labels


sys.stdout=open(os.path.join(path_to_save_results, "run_out_"+name_file+"_10fold.txt"),'w')
# save prints in a txt file
original_stdout = sys.stdout

pin_memory = torch.cuda.is_available()
str_cuda="cuda:0"
device = torch.device(str_cuda if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

def print_specific_layer_weights(model, layer_name):
    # print("Name layers in model:")
    for name, param in model.named_parameters():
        # print(name)
        if name == layer_name:
            print(f"Weights of {name}:", param.data)

image_shape=301
depth=40

# Set data directory
directory = "/home/ubuntu/tenerife/data/ICH_nii_StrippedSkull"
# read images and corresponding labels from directory
goodPrognosis_images = sorted(os.listdir(os.path.join(directory, "GOOD_PROGNOSIS")))
print(f"Good prognosis images(0): {len(goodPrognosis_images)}")
PoorPrognosis_images = sorted(os.listdir(os.path.join(directory, "POOR_PROGNOSIS")))
print(f"Poor prognosis images (1): {len(PoorPrognosis_images)}")

# read CLINICAL DATA
clinical_data = pd.read_csv('/home/ubuntu/tenerife/data/ICH_tabular_data/CLINICAL_DATA_ICH.csv', delimiter=',')
clinical_data = clinical_data.rename(columns={'PatientID': 'Patient'})
# drop row where Patient is 213
clinical_data_filtered = clinical_data[~clinical_data['Patient'].isin([213])]
clinical_data_filtered = clinical_data_filtered.reset_index(drop=True)

images_all=[]
labels_all=[]
# loop over Patient in clinical_data
for i in range(len(clinical_data)):
    # get patient ID
    patientID = clinical_data['Patient'][i]
    # get label
    label = clinical_data['Label (poor_prognosis)'][i]
    # read corresponding image
    if label==0:
        # check that image exists
        if os.path.isfile(os.path.join(directory, "GOOD_PROGNOSIS", str(patientID) + "_brain.nii.gz")):
            images_all.append(os.path.join(directory, "GOOD_PROGNOSIS", str(patientID) + "_brain.nii.gz"))
            labels_all.append(label)
    elif label==1:
        # check that image exists
        if os.path.isfile(os.path.join(directory, "POOR_PROGNOSIS", str(patientID) + "_brain.nii.gz")):
            images_all.append(os.path.join(directory, "POOR_PROGNOSIS", str(patientID) + "_brain.nii.gz"))
            labels_all.append(label)
    else:
        print("ERROR: for Patient", patientID, "label not found")
print("Number of images:", len(images_all))
print("Number of labels:", len(labels_all))
images_all=np.array(images_all)
labels_all=np.array(labels_all)

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# create 5 fold cross validation

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
print("=" * 80)
# save best metrics for each fold
fold_metrics_df = pd.DataFrame(columns=['Fold', 'AUC', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score',
                                         'AUC_Age', 'Balanced_accuracy_Age', 'Accuracy_Age', 'Specificity_Age', 'NPV_Age', 'Precision_Age', 'Recall_Age', 'F1-score_Age'])
# save metrics for best threshold regarding recall for each fold
fold_metrics_recall_df = pd.DataFrame(columns=['Fold', 'AUC', 'Threshold', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score'])
# save metrics for best threshold regarding f1-score for each fold
fold_metrics_f1_df = pd.DataFrame(columns=['Fold', 'AUC', 'Threshold', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score'])
# save test labels and predictions for each fold
test_labels_df = pd.DataFrame(columns=['Fold', 'True_labels', 'Probabilities_labels_test', 'Predicted_labels_test_th0.5',
                                       'True_labels_Age', 'Probabilities_labels_test_Age', 'Predicted_labels_test_th0.5_Age'])

# Iterate over the folds
for fold, (train_index, test_index) in enumerate(skf.split(images_all, labels_all)):
    print(f"Fold {fold + 1}:")

    # Split the data into train and test sets
    images_train, images_test = images_all[train_index], images_all[test_index]
    labels_train, labels_test = labels_all[train_index], labels_all[test_index]

    # Further split the training set into train and validation sets
    images_train, images_val, labels_train, labels_val = train_test_split(
        images_train, labels_train, test_size=0.1, random_state=1, stratify=labels_train
    )

    # convert back to list
    images_train = images_train.tolist()
    images_val = images_val.tolist()
    images_test = images_test.tolist()
    labels_train = labels_train.tolist()
    labels_val = labels_val.tolist()
    labels_test = labels_test.tolist()

    # Print information for each fold
    print("Training set:", len(images_train), "images,", labels_train.count(0), "good prognosis,", labels_train.count(1), "poor prognosis")
    print("Validation set:", len(images_val), "images,", labels_val.count(0), "good prognosis,", labels_val.count(1), "poor prognosis")
    print("Test set:", len(images_test), "images,", labels_test.count(0), "good prognosis,", labels_test.count(1), "poor prognosis")
    print("=" * 80)

    # Count the occurrences of each class in the training set
    class_counts = {0: labels_train.count(0), 1: labels_train.count(1)}

    # Determine the target count for balancing
    target_count = max(class_counts.values())

    # Perform random oversampling on the minority class (class 1) ### TODO the oversampling technique could be improved
    ros = RandomOverSampler(sampling_strategy={0: target_count, 1: target_count}, random_state=2)
    images_res, labels_res = ros.fit_resample(np.array(images_train).reshape(-1, 1), np.array(labels_train).reshape(-1, 1))
    labels_res = labels_res.tolist()
    images_res = [item for sublist in images_res for item in sublist]
    print("Training set after resampling:", len(images_res), "images,", labels_res.count(0), "good prognosis,", labels_res.count(1), "poor prognosis")
    
    # clinical data
    df = clinical_data_filtered
    # rename patient column
    df = df.rename(columns={'Patient': 'PatientID'})
    print("Shape clinical data dataframe:", df.shape)

    # loop over images_train, images_val and images_test to get the corresponding clinical data
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    X_val = pd.DataFrame()
    for i in images_res:
        patientID = int(i.split('/')[-1].split('_brain.')[0])
        # get all columns in df for this patient
        X_train = pd.concat([X_train, df[df['PatientID']==patientID]])
    for i in images_test:
        patientID = int(i.split('/')[-1].split('_brain.')[0])
        # get all columns in df for this patient
        X_test = pd.concat([X_test, df[df['PatientID']==patientID]])
    for i in images_val:
        patientID = int(i.split('/')[-1].split('_brain.')[0])
        # get all columns in df for this patient
        X_val = pd.concat([X_val, df[df['PatientID']==patientID]])

    # remove PatientID from X_train, X_test and X_val
    X_train = X_train.drop(columns=['PatientID', 'Label (poor_prognosis)'])
    X_test = X_test.drop(columns=['PatientID', 'Label (poor_prognosis)'])
    X_val = X_val.drop(columns=['PatientID', 'Label (poor_prognosis)'])
    
    only_GCS=False
    only_Age=True

    if only_GCS==True:
        gcs_train = X_train[['GCS']]
        gcs_test = X_test[['GCS']]
        gcs_val = X_val[['GCS']]
        ordinal_categorical_var=['GCS']
        # binary encode, if gCS <=8 then 0, if GCS >8 then 1
        gcs_train['GCS'] = gcs_train['GCS'].apply(lambda x: 1 if x<=8 else 0)
        gcs_test['GCS'] = gcs_test['GCS'].apply(lambda x: 1 if x<=8 else 0)
        gcs_val['GCS'] = gcs_val['GCS'].apply(lambda x: 1 if x<=8 else 0)
        print("Number of ordinal categorical variables:", len(ordinal_categorical_var))
        print("Distribution of values of GCS after binaryzation")
        print("Train:", gcs_train['GCS'].value_counts())
        print("Val:", gcs_val['GCS'].value_counts())
        print("Test:", gcs_test['GCS'].value_counts())
        
        # convert to tensor
        gcs_train_tensor = gcs_train['GCS'].values.tolist()
        gcs_val_tensor = gcs_val['GCS'].values.tolist()
        gcs_test_tensor = gcs_test['GCS'].values.tolist()

        # for the weights:
        num_positive_gcs = sum(label == 1 for label in gcs_train_tensor)
        num_negative_gcs = sum(label == 0 for label in gcs_train_tensor)
        # Calculate the weight for the positive class
        pos_weight_gcs = torch.tensor([num_negative_gcs / num_positive_gcs], dtype=torch.float32)
        print("Positive weight GCS:", pos_weight_gcs)
    
    if only_Age==True:
        age_train = X_train[['Age']]
        age_test = X_test[['Age']]
        age_val = X_val[['Age']]
        ordinal_regression_var=['Age']
        # binary encode, if age <=65 then 0, if age >65 then 1
        age_train['Age'] = age_train['Age'].apply(lambda x: 1 if x>=80 else 0)
        age_test['Age'] = age_test['Age'].apply(lambda x: 1 if x>=80 else 0)
        age_val['Age'] = age_val['Age'].apply(lambda x: 1 if x>=80 else 0)
        print("Number of ordinal categorical variables:", len(ordinal_regression_var))
        print("Distribution of values of Age after binaryzation")
        print("Train:", age_train['Age'].value_counts())
        print("Val:", age_val['Age'].value_counts())
        print("Test:", age_test['Age'].value_counts())
        
        # convert to tensor
        age_train_tensor = age_train['Age'].values.tolist()
        age_val_tensor = age_val['Age'].values.tolist()
        age_test_tensor = age_test['Age'].values.tolist()

        # for the weights:
        num_positive_age = sum(label == 1 for label in age_train_tensor)
        num_negative_age = sum(label == 0 for label in age_train_tensor)

        # Calculate the weight for the positive class
        pos_weight_age = torch.tensor([num_negative_age / num_positive_age], dtype=torch.float32)
        print("Positive weight Age:", pos_weight_age)
    
    train_files = [{"image": image_name, "label": label_name, "age": age_name} for image_name, label_name, age_name in zip(images_res, labels_res, age_train_tensor)]
    val_files = [{"image": image_name, "label": label_name, "age": age_name} for image_name, label_name, age_name in zip(images_val, labels_val, age_val_tensor)]
    test_files = [{"image": image_name, "label": label_name, "age": age_name} for image_name, label_name, age_name in zip(images_test, labels_test, age_test_tensor)]
    
    # https://github.com/Project-MONAI/tutorials/blob/main/modules/load_medical_images.ipynb

    # Define transforms
    train_transforms = Compose(
        [LoadImaged(keys="image", image_only=True, ensure_channel_first=True, reader="ITKReader"),
        Rotate90d(keys="image", k=3), 
        Flipd(keys="image", spatial_axis=1),
        NormalizeIntensityd(keys="image", subtrahend=15, divisor=85),
        ThresholdIntensityd(keys="image", threshold=0, above=True, cval=0.0),
        ThresholdIntensityd(keys="image", threshold=1, above=False, cval=0.0),
        SpatialPadd(keys="image", spatial_size=[-1, -1, depth], mode=('constant'), method= ("symmetric")),
        Resized(keys="image", spatial_size=[image_shape, image_shape, depth]),
        RandRotated(keys="image", prob=0.5, range_z=np.pi/36, range_y=np.pi/36, range_x=np.pi/36), # randzoom, rand3delastic could also be used
        # RandAffined(keys="image", scale_range=(0.9, 1), rotate_range=(np.pi/18, np.pi/18), prob=0.5),
        Zoomd(keys="image", zoom=1.1, prob=0.5),
        # Rand3DElasticd(keys="image", sigma_range=(2, 3), magnitude_range=(100, 200), prob=0.1),
        RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=0.01),
        Transposed(keys="image", indices=[0, 3, 1, 2])
        ])

    val_transforms = Compose(
        [LoadImaged(keys="image", image_only=True, ensure_channel_first=True, reader="ITKReader"), 
        Rotate90d(keys="image", k=3), 
        Flipd(keys="image", spatial_axis=1),
        NormalizeIntensityd(keys="image", subtrahend=15, divisor=85),
        ThresholdIntensityd(keys="image", threshold=0, above=True, cval=0.0),
        ThresholdIntensityd(keys="image", threshold=1, above=False, cval=0.0),
        SpatialPadd(keys="image", spatial_size=[-1, -1, depth], mode=('constant'), method= ("symmetric")),
        Resized(keys="image", spatial_size=[image_shape, image_shape, depth]),
        Transposed(keys="image", indices=[0, 3, 1, 2])
        ])

    # Define nifti dataset, data loader
    check_ds = Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    try:
        result = monai.utils.misc.first(check_loader)
        print(type(result['image']), result['image'].shape, result['label'])
    except RuntimeError as e:
        print("Error occurred when applying transform:", e)
    # create a training data loader
    # train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=pin_memory)

    # you can use Dataset or CacheDataset, according to 
    # this https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb
    # the later is faster

    # create a validation data loader
    # val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=2, pin_memory=pin_memory)
    path_to_save_model=os.path.join(path_to_save_model_dir,
                                                "PrognosisModelICH_DenseNet_"+name_file+"_fold"+str(fold)+".pth")
    if not os.path.isfile(path_to_save_model):
        model = PrognosisICH_BinaryAge_Model(image_shape=image_shape, depth=depth, spatial_dims=3, in_channels=1, num_classes_binary=1, dropout_prob=0.2)
        # print the name of the layers in the model
        # print("Name layers in model:")
        # for name, param in model.named_parameters():
        #     print(name)
        
        total_params = sum(
            param.numel() for param in model.parameters()
        )
        print("Total params model", total_params)

        # model = model.cuda()

        threshold = 0.5
        ###
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs")
        #     model= torch.nn.DataParallel(model)

        model.to(device)
        # print("Model's summary:")
        # print(summary(model, (1, image_shape, image_shape, depth)))

        # loss_function = torch.nn.CrossEntropyLoss() # this is for 2 out_channels output
        binary_loss_function = torch.nn.BCEWithLogitsLoss().to(device) # also works with this data
        ordinal_loss_function_Age = torch.nn.BCEWithLogitsLoss().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=True)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.98, verbose=False)
        #### understand well all the training and revise
        #### create custom model to append at the end tabular data and image features
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        best_age_metric = -1
        best_age_metric_epoch = -1
        epoch_loss_values = []
        epoch_auc_values = []
        epoch_accuracy_values = []
        epoch_age_accuracy_values = []
        epoch_age_auc_values = []
        auc_values = []
        accuracy_values = []
        val_age_auc_values = []
        val_age_accuracy_values = []

        max_epochs = 100

        # writer = SummaryWriter()

        # early stopping
        patience = 20
        epochs_no_improve = 0
        early_stop = False
        accumulation_steps = 3

        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            print('Epoch-{0} lr: {1}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
            model.train()
            epoch_loss = 0
            step = 0
            all_predictions = []  # To store predicted values
            all_probabilities = []  # To store corresponding probabilities
            all_labels = []  # To store ground truth

            all_age_predictions = []  # To store predicted values for Age
            all_age_probabilities = []  # To store corresponding probabilities for Age
            all_age_labels = []  # To store ground truth for Age

            predictions_train=[]
            labels_train=[]
            probabilities_train=[]

            predictions_age_train=[]
            labels_age_train=[]
            probabilities_age_train=[]

            optimizer.zero_grad()
            i=0
            for batch_data in train_loader:
                i+=1
                step += 1
                inputs, labels, labels_age = batch_data["image"].to(device), batch_data["label"].to(device), batch_data["age"].to(device)
                # optimizer.zero_grad()
                # print(inputs.shape)
                binary_output, output_age = model(inputs)
                ### sequeeze to get the right output shape
                binary_output = binary_output.squeeze()
                binary_loss = binary_loss_function(binary_output, labels.float())
                # print("Predicted probabilities binary output:", binary_output)

                output_age = output_age.squeeze()
                ordinal_loss_age = ordinal_loss_function_Age(output_age, labels_age.float())
                
                # Combine losses
                combined_loss = 0.5*binary_loss + 0.3*ordinal_loss_age # Consider weighting losses if necessary

                combined_loss = combined_loss / accumulation_steps  # Normalize loss for accumulation
                combined_loss.backward()  # Accumulate gradients

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()  # Update model parameters
                    # model.binary_head[-1].weight.data.clamp_(min=0)
                    # model.binary_head[-1].bias.data.clamp_(min=0)
                    # model.ordinal_head[-1].weight.data.clamp_(min=0)
                    # model.ordinal_head[-1].bias.data.clamp_(min=0)
                    scheduler.step(combined_loss)  # Scheduler step (if scheduler requires loss)
                    optimizer.zero_grad()  # Reset gradients
                    # print("Already clamped weights and biases")
                    # print("Model binary head weights after clamping:", model.binary_head[-1].weight.data.min())
                    # print("Model binary head biases after clamping:", model.binary_head[-1].bias.data.min())
                    # print("Model ordinal head weights after clamping:", model.ordinal_head[-1].weight.data.min())
                    # print("Model ordinal head biases after clamping:", model.ordinal_head[-1].bias.data.min())

                epoch_loss += combined_loss.item() * accumulation_steps  # Correct loss scaling
                epoch_len = len(train_ds) // train_loader.batch_size

                train_probabilities = nn.Sigmoid()(binary_output)
                train_predictions = (train_probabilities >= threshold).float()
                probabilities_train.extend(train_probabilities.detach().cpu().numpy())
                labels_train.extend(labels.cpu().numpy())
                predictions_train.extend(train_predictions.detach().cpu().numpy())
                # for Age
                labels_age_train.extend(labels_age.cpu().numpy())
                ordinal_probabilities_age = nn.Sigmoid()(output_age)
                probabilities_age_train.extend(ordinal_probabilities_age.detach().cpu().numpy())
                # print("Predicted probabilities ordinal output Age:", ordinal_probabilities_age)
                # Predicted classes
                predicted_ordinal_classes_age = (ordinal_probabilities_age >= threshold).float()
                predictions_age_train.extend(predicted_ordinal_classes_age.detach().cpu().numpy())
            
            # Evaluation of binary prognosis
            train_auc=roc_auc_score(labels_train, probabilities_train)
            train_accuracy = balanced_accuracy_score(labels_train, predictions_train)
            unique_train, counts_train = np.unique(predictions_train, return_counts=True)
            # print("Train predictions count:", dict(zip(unique_train, counts_train)))
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            epoch_auc_values.append(train_auc)
            epoch_accuracy_values.append(train_accuracy)
            print(f"Train epoch {epoch + 1} average loss: {epoch_loss:.3f}, train_auc: {train_auc:.3f}, train_accuracy: {train_accuracy:.3f}")
            # Evaluation of age prediction
            age_accuracy_train = accuracy_score(labels_age_train, predictions_age_train)
            unique_age_train, counts_age_train = np.unique(labels_age_train, return_counts=True)
            print("labels_age_train:", dict(zip(unique_age_train, counts_age_train)))
            print("probabilities_age_train:", probabilities_age_train)
            age_auc_train = roc_auc_score(labels_age_train, probabilities_age_train)
            print(f"Train epoch {epoch + 1} Age accuracy: {age_accuracy_train:.3f}, Age AUC: {age_auc_train:.3f}")
            epoch_age_accuracy_values.append(age_accuracy_train)
            epoch_age_auc_values.append(age_auc_train)
            
            if (epoch + 1) % val_interval == 0:
                model.eval()

                num_correct = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels, val_labels_age = val_data["image"].to(device), val_data["label"].to(device), val_data["age"].to(device)
                    with torch.no_grad():
                        binary_val_outputs, val_outputs_age = model(val_images)
                        binary_val_outputs = binary_val_outputs.squeeze() ### sequeeze to get the right output shape

                        # _, predicted = val_outputs.max(1)
                        probabilities = nn.Sigmoid()(binary_val_outputs)
                        predicted = (probabilities >= threshold).float()
                        # value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(val_labels.cpu().numpy())
                        all_probabilities.extend(probabilities.cpu().numpy())

                        # for age
                        val_outputs_age = val_outputs_age.squeeze()
                        val_probabilities_age = nn.Sigmoid()(val_outputs_age)
                        predicted_val_ordinal_classes_age = (val_probabilities_age >= threshold).float()
                        all_age_predictions.extend(predicted_val_ordinal_classes_age.cpu().numpy())
                        all_age_labels.extend(val_labels_age.cpu().numpy())
                        all_age_probabilities.extend(val_probabilities_age.cpu().numpy())

                all_predictions = np.array(all_predictions).astype(int)
                all_probabilities = np.array(all_probabilities)
                all_labels = np.array(all_labels)
                unique, counts = np.unique(all_predictions, return_counts=True)
                print("Predictions count:", dict(zip(unique, counts)))
                print("Probabilities in validation:", all_probabilities)
                # Calculate metrics
                roc_auc = roc_auc_score(all_labels, all_probabilities,average='weighted')
                accuracy = balanced_accuracy_score(all_labels, all_predictions)
                precision = precision_score(all_labels, all_predictions, average='weighted')
                recall = recall_score(all_labels, all_predictions, average='weighted')
                f1 = f1_score(all_labels, all_predictions, average='weighted')
                auc_values.append(roc_auc)
                accuracy_values.append(accuracy)

                # for Age
                all_age_predictions = np.array(all_age_predictions).astype(int)
                all_age_probabilities = np.array(all_age_probabilities)
                all_age_labels = np.array(all_age_labels)
                unique_age, counts_age = np.unique(all_age_predictions, return_counts=True)
                print("Predictions count Age:", dict(zip(unique_age, counts_age)))
                print("Probabilities in validation Age:", all_age_probabilities)
                # Calculate metrics
                roc_auc_age = roc_auc_score(all_age_labels, all_age_probabilities,average='weighted')
                accuracy_age = balanced_accuracy_score(all_age_labels, all_age_predictions)
                precision_age = precision_score(all_age_labels, all_age_predictions, average='weighted')
                recall_age = recall_score(all_age_labels, all_age_predictions, average='weighted')
                f1_age = f1_score(all_age_labels, all_age_predictions, average='weighted')
                val_age_auc_values.append(roc_auc_age)
                val_age_accuracy_values.append(accuracy_age)

                # to perform early-stopping we select the best metric in prognosis
                if accuracy > best_metric:
                    best_metric = accuracy
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), path_to_save_model)
                    print("saved new best metric model")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    # Check early stopping condition
                    if epochs_no_improve == patience:
                        print('Early stopping!')
                        early_stop = True
                        break

                if accuracy_age > best_age_metric:
                    best_age_metric = accuracy_age
                    best_age_metric_epoch = epoch + 1

                # Print metrics for each epoch
                print(f"Epoch {epoch + 1}/{max_epochs} - AUC: {roc_auc:.3f}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")

                print(
                        f"current epoch: {epoch + 1} current AUC: {roc_auc:.3f}"
                        f" current accuracy: {accuracy:.3f}"
                        f" best accuracy: {best_metric:.3f}"
                        f" at epoch: {best_metric_epoch}"
                )
                print(
                        f" current Age AUC: {roc_auc_age:.3f}"
                        f" current Age accuracy: {accuracy_age:.3f}"
                        f" best Age accuracy: {best_age_metric:.3f}"
                        f" at epoch: {best_age_metric_epoch}"
                )
                # for age
                print(f"Epoch {epoch + 1}/{max_epochs} - Age AUC: {roc_auc_age:.3f}, Age Accuracy: {accuracy_age:.3f}, Age Precision: {precision_age:.3f}, Age Recall: {recall_age:.3f}, Age F1-score: {f1_age:.3f}")
                # writer.add_scalar("auc", roc_auc, epoch + 1)

        print(f"Training completed, best_metric in prognosis: {best_metric:.3f} at epoch: {best_metric_epoch}")
        print("-" * 50)
        # writer.close()
        plt.figure("train", (12, 12))
        plt.subplot(1, 3, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y)

        plt.subplot(1, 3, 2)
        plt.title("AUC prognosis")
        x = [i + 1 for i in range(len(epoch_auc_values))]
        y = epoch_auc_values
        plt.plot(x, y)
        x = [val_interval * (i + 1) for i in range(len(auc_values))]
        y = auc_values
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.ylabel("AUC")
        plt.legend(['train', 'val'], loc='best')

        plt.subplot(1, 3, 3)
        plt.title("AUC Age")
        x = [i + 1 for i in range(len(epoch_age_auc_values))]
        y = epoch_age_auc_values
        plt.plot(x, y)
        x = [val_interval * (i + 1) for i in range(len(val_age_auc_values))]
        y = val_age_auc_values
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.ylabel("AUC")
        plt.legend(['train_Age', 'val_Age'], loc='best')
        plt.savefig(os.path.join(path_to_save_results, "TrainAndVal_LossAUC_"+name_file+"_fold"+str(fold)+".png"))
        plt.close()

        plt.figure("train", (12, 12))
        plt.subplot(1, 2, 1)
        plt.title("Balanced accuracy")
        x = [i + 1 for i in range(len(epoch_accuracy_values))]
        y = epoch_accuracy_values
        plt.plot(x, y)
        x = [val_interval * (i + 1) for i in range(len(accuracy_values))]
        y = accuracy_values
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.ylabel("Balanced accuracy")
        plt.legend(['train', 'val'], loc='best')

        plt.subplot(1, 2, 2)
        x = [i + 1 for i in range(len(epoch_age_accuracy_values))]
        y = epoch_age_accuracy_values
        plt.plot(x, y)
        x = [val_interval * (i + 1) for i in range(len(val_age_accuracy_values))]
        y = val_age_accuracy_values
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.ylabel("Balanced accuracy")
        plt.legend(['train_Age', 'val_Age'], loc='best')
        plt.savefig(os.path.join(path_to_save_results, "TrainAndVal_BalAcc_"+name_file+"_fold"+str(fold)+".png"))
        plt.close()

    print("Inference in test")
    threshold = 0.5

    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=8)

    model = PrognosisICH_BinaryAge_Model(image_shape=image_shape, depth=depth, spatial_dims=3, in_channels=1, num_classes_binary=1, dropout_prob=0.2)
    model.to(device)
    model.load_state_dict(torch.load(path_to_save_model))
    model.eval()
    predicted_labels_test = []
    all_probabilities_test = []
    labels_test_tensor = []
    # for age
    predicted_labels_age_test = []
    all_probabilities_age_test = []
    labels_age_test_tensor = []
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels, test_labels_age = test_data["image"].to(device), test_data["label"].to(device), test_data["age"].to(device)
            outputs_test, outputs_age = model(test_images)
            outputs_test = outputs_test.squeeze() ### sequeeze to get the right output shape
            probabilities_test = nn.Sigmoid()(outputs_test)
            predicted_test = (probabilities_test >= threshold).float()
            print("Probabilities test:", probabilities_test.cpu().numpy())
            all_probabilities_test.extend(probabilities_test.cpu().numpy())
            predicted_labels_test.extend(predicted_test.cpu().numpy())
            labels_test_tensor.extend(test_labels.cpu().numpy())
            # for age
            outputs_age = outputs_age.squeeze()
            probabilities_age_test = nn.Sigmoid()(outputs_age)
            predicted_age_test = (probabilities_age_test >= threshold).float()
            all_probabilities_age_test.extend(probabilities_age_test.cpu().numpy())
            predicted_labels_age_test.extend(predicted_age_test.cpu().numpy())
            labels_age_test_tensor.extend(test_labels_age.cpu().numpy())
            

    all_probabilities_test = np.array(all_probabilities_test)
    labels_test_tensor = np.array(labels_test_tensor).astype(int)
    predicted_labels_test=np.array(predicted_labels_test).astype(int)
    # for age
    all_probabilities_age_test = np.array(all_probabilities_age_test)
    labels_age_test_tensor = np.array(labels_age_test_tensor).astype(int)
    predicted_labels_age_test=np.array(predicted_labels_age_test).astype(int)
    # create a fold_array that repeates the fold number as many times as test samples
    fold_array = np.full((len(labels_test_tensor)), fold)
    combined=np.column_stack((fold_array, labels_test_tensor, all_probabilities_test, predicted_labels_test,
                              labels_age_test_tensor, all_probabilities_age_test, predicted_labels_age_test))
    test_labels_df = pd.concat([test_labels_df, pd.DataFrame(combined, columns=test_labels_df.columns)], ignore_index=True)
    test_balanced_accuracy = balanced_accuracy_score(labels_test_tensor, predicted_labels_test)
    test_accuracy = accuracy_score(labels_test_tensor, predicted_labels_test)
    test_auc = roc_auc_score(labels_test_tensor, all_probabilities_test)
    test_precision = precision_score(labels_test_tensor, predicted_labels_test)
    test_recall = recall_score(labels_test_tensor, predicted_labels_test)
    test_f1 = f1_score(labels_test_tensor, predicted_labels_test)

    tn, fp, fn, tp = confusion_matrix(labels_test_tensor, predicted_labels_test, labels=[0, 1]).ravel()
    test_NPV=tn/(tn+fn)
    test_specificity=tn/(tn+fp)
    # for age
    test_age_balanced_accuracy = balanced_accuracy_score(labels_age_test_tensor, predicted_labels_age_test)
    test_age_accuracy = accuracy_score(labels_age_test_tensor, predicted_labels_age_test)
    test_age_auc = roc_auc_score(labels_age_test_tensor, all_probabilities_age_test)
    test_age_precision = precision_score(labels_age_test_tensor, predicted_labels_age_test)
    test_age_recall = recall_score(labels_age_test_tensor, predicted_labels_age_test)
    test_age_f1 = f1_score(labels_age_test_tensor, predicted_labels_age_test)

    tn_age, fp_age, fn_age, tp_age = confusion_matrix(labels_age_test_tensor, predicted_labels_age_test, labels=[0, 1]).ravel()
    test_age_NPV=tn_age/(tn_age+fn_age)
    test_age_specificity=tn_age/(tn_age+fp_age)

    # save in fold_metrics_df
    fold_metrics_df = pd.concat([fold_metrics_df, pd.DataFrame([{'Fold': fold, 'AUC': test_auc, 
                'Balanced_accuracy': test_balanced_accuracy, 'Accuracy': test_accuracy, 'Specificity': test_specificity, 
                'NPV': test_NPV, 'Precision': test_precision, 'Recall': test_recall, 'F1-score': test_f1, 
                'AUC_Age': test_age_auc, 'Accuracy_Age': test_age_accuracy,
                'Balanced_accuracy_Age': test_age_balanced_accuracy, 'Precision_Age': test_age_precision,
                'NPV_Age': test_age_NPV, 'Specificity_Age': test_age_specificity,
                'Recall_Age': test_age_recall, 'F1-score_Age': test_age_f1}])], ignore_index=True)

    print("Probabilities test in prognosis:", all_probabilities_test)
    unique, counts = np.unique(predicted_labels_test, return_counts=True)
    print("Predictions count test:", dict(zip(unique, counts)))
    print(f'Test Accuracy: {test_accuracy:.2%}')
    print(f'Test ROC AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

    print("Probabilities test in Age:", all_probabilities_age_test)
    unique_age, counts_age = np.unique(predicted_labels_age_test, return_counts=True)
    print("Predictions count test Age:", dict(zip(unique_age, counts_age)))
    print(f'Test Age Accuracy: {test_age_accuracy:.2%}')
    print(f'Test Age ROC AUC: {test_age_auc:.4f}, Precision: {test_age_precision:.4f}, Recall: {test_age_recall:.4f}, F1: {test_age_f1:.4f}')

    # Save predicted labels for test set
    predicted_labels_df = pd.DataFrame({'True Labels': labels_test, 'Predicted Labels': predicted_labels_test})
    # predicted_labels_df.to_csv('predicted_labels.csv', index=False)

    # plot summary report
    print(classification_report(labels_test_tensor, predicted_labels_test, target_names=['Good prognosis', 'Poor prognosis']))

    evaluator = EvaluateThresholds(all_probabilities_test, labels_test_tensor,
             path_to_save_auc_plot=os.path.join(path_to_save_results,"AUC_plots",
                                                 "AUC_testplot_"+name_file+"_fold"+str(fold)+".png"), fold=fold)
    auc_score = evaluator.plot_roc_curve()
    print("AUC score calculated by evaluator:", auc_score)
    best_threshold_recall, best_metrics_recall = evaluator.evaluate_metrics_recall()
    print("Best threshold based on F1-score:", best_threshold_recall, "Best metrics based on recall:", best_metrics_recall)
    fold_metrics_recall_df = pd.concat([fold_metrics_recall_df, pd.DataFrame([best_metrics_recall])], ignore_index=True)
    best_threshold_f1, best_metrics_f1 = evaluator.evaluate_metrics_f1()
    print("Best threshold based on F1-score:", best_threshold_f1, "Best metrics based on f1-score:", best_metrics_f1)
    fold_metrics_f1_df = pd.concat([fold_metrics_f1_df, pd.DataFrame([best_metrics_f1])], ignore_index=True)

    # try:
    #     test_auc_boots=bootstrapping(y_true=labels_test_tensor, y_pred=all_probabilities_test, y_pred_threshold=predicted_labels_test, 
    #                 path_to_save_metrics='/home/ubuntu/tenerife/data/ICH_results/tabularData_model', 
    #                 metrics = 'AUC', confidence = 0.95, n_bootstraps = 1000)
    #     print(test_auc_boots)

    #     all_metrics_boots=bootstrapping(y_true=labels_test_tensor, y_pred=all_probabilities_test, y_pred_threshold=predicted_labels_test, 
    #                 path_to_save_metrics='/home/ubuntu/tenerife/data/ICH_results/tabularData_model', 
    #                 metrics = 'METRICS', confidence = 0.95, n_bootstraps = 1000)
    #     print(all_metrics_boots)
    # except Exception as e:
    #     print(e)
    bootstrap_metric_ci(y_true = labels_test_tensor, y_pred_threshold = predicted_labels_test)

    # save fold_metrics_df
    fold_metrics_df.to_csv(os.path.join(path_to_save_results, "fold_metrics_df_"+name_file+".csv"), index=False)
    test_labels_df.to_csv(os.path.join(path_to_save_results, "test_labels_df_"+name_file+".csv"), index=False)
    fold_metrics_recall_df.to_csv(os.path.join(path_to_save_results, "fold_metrics_recall_"+name_file+".csv"), index=False)
    fold_metrics_f1_df.to_csv(os.path.join(path_to_save_results, "fold_metrics_f1_"+name_file+".csv"), index=False)

    # torch.cuda.empty_cache()
    print("=" * 80)
    

print("=" * 80)
# calculate mean and std of metrics for all folds
mean_auc = fold_metrics_df['AUC'].mean()
std_auc = fold_metrics_df['AUC'].std()
mean_accuracy = fold_metrics_df['Accuracy'].mean()
std_accuracy = fold_metrics_df['Accuracy'].std()
mean_balanced_accuracy = fold_metrics_df['Balanced_accuracy'].mean()
std_balanced_accuracy = fold_metrics_df['Balanced_accuracy'].std()
mean_specificity = fold_metrics_df['Specificity'].mean()
std_specificity = fold_metrics_df['Specificity'].std()
mean_NPV = fold_metrics_df['NPV'].mean()
std_NPV = fold_metrics_df['NPV'].std()
mean_precision = fold_metrics_df['Precision'].mean()
std_precision = fold_metrics_df['Precision'].std()
mean_recall = fold_metrics_df['Recall'].mean()
std_recall = fold_metrics_df['Recall'].std()
mean_f1 = fold_metrics_df['F1-score'].mean()
std_f1 = fold_metrics_df['F1-score'].std()
# print metrics
print("Mean AUC:", mean_auc, "Std AUC:", std_auc)
print("Mean Accuracy:", mean_accuracy, "Std Accuracy:", std_accuracy)
print("Mean Balanced accuracy:", mean_balanced_accuracy, "Std Balanced accuracy:", std_balanced_accuracy)
print("Mean Specificity:", mean_specificity, "Std Specificity:", std_specificity)
print("Mean NPV:", mean_NPV, "Std NPV:", std_NPV)
print("Mean Precision:", mean_precision, "Std Precision:", std_precision)
print("Mean Recall:", mean_recall, "Std Recall:", std_recall)
print("Mean F1-score:", mean_f1, "Std F1-score:", std_f1)
# for age
mean_auc_age = fold_metrics_df['AUC_Age'].mean()
std_auc_age = fold_metrics_df['AUC_Age'].std()
mean_accuracy_age = fold_metrics_df['Accuracy_Age'].mean()
std_accuracy_age = fold_metrics_df['Accuracy_Age'].std()
mean_balanced_accuracy_age = fold_metrics_df['Balanced_accuracy_Age'].mean()
std_balanced_accuracy_age = fold_metrics_df['Balanced_accuracy_Age'].std()
mean_specificity_age = fold_metrics_df['Specificity_Age'].mean()
std_specificity_age = fold_metrics_df['Specificity_Age'].std()
mean_NPV_age = fold_metrics_df['NPV_Age'].mean()
std_NPV_age = fold_metrics_df['NPV_Age'].std()
mean_precision_age = fold_metrics_df['Precision_Age'].mean()
std_precision_age = fold_metrics_df['Precision_Age'].std()
mean_recall_age = fold_metrics_df['Recall_Age'].mean()
std_recall_age = fold_metrics_df['Recall_Age'].std()
mean_f1_age = fold_metrics_df['F1-score_Age'].mean()
std_f1_age = fold_metrics_df['F1-score_Age'].std()
# print metrics
print("Mean AUC Age:", mean_auc_age, "Std AUC Age:", std_auc_age)
print("Mean Accuracy Age:", mean_accuracy_age, "Std Accuracy Age:", std_accuracy_age)
print("Mean Balanced accuracy Age:", mean_balanced_accuracy_age, "Std Balanced accuracy Age:", std_balanced_accuracy_age)
print("Mean Specificity Age:", mean_specificity_age, "Std Specificity Age:", std_specificity_age)
print("Mean NPV Age:", mean_NPV_age, "Std NPV Age:", std_NPV_age)
print("Mean Precision Age:", mean_precision_age, "Std Precision Age:", std_precision_age)
print("Mean Recall Age:", mean_recall_age, "Std Recall Age:", std_recall_age)
print("Mean F1-score Age:", mean_f1_age, "Std F1-score Age:", std_f1_age)
print("=" * 80)
print("=" * 80)
# calculate best metrics for best threshold based on recall
mean_auc_recall = fold_metrics_recall_df['AUC'].mean()
std_auc_recall = fold_metrics_recall_df['AUC'].std()
mean_threshold_recall = fold_metrics_recall_df['Threshold'].mean()
std_threshold_recall = fold_metrics_recall_df['Threshold'].std()
mean_accuracy_recall = fold_metrics_recall_df['Accuracy'].mean()
std_accuracy_recall = fold_metrics_recall_df['Accuracy'].std()
mean_balanced_accuracy_recall = fold_metrics_recall_df['Balanced_accuracy'].mean()
std_balanced_accuracy_recall = fold_metrics_recall_df['Balanced_accuracy'].std()
mean_specificity_recall = fold_metrics_recall_df['Specificity'].mean()
std_specificity_recall = fold_metrics_recall_df['Specificity'].std()
mean_NPV_recall = fold_metrics_recall_df['NPV'].mean()
std_NPV_recall = fold_metrics_recall_df['NPV'].std()
mean_precision_recall = fold_metrics_recall_df['Precision'].mean()
std_precision_recall = fold_metrics_recall_df['Precision'].std()
mean_recall_recall = fold_metrics_recall_df['Recall'].mean()
std_recall_recall = fold_metrics_recall_df['Recall'].std()
mean_f1_recall = fold_metrics_recall_df['F1-score'].mean()
std_f1_recall = fold_metrics_recall_df['F1-score'].std()
# print metrics
print("=" * 80)
print("Mean AUC (recall):", mean_auc_recall, "Std AUC recall:", std_auc_recall)
print("Mean Threshold optimized for recall:", mean_threshold_recall, "Std Threshold recall:", std_threshold_recall)
print("Mean Accuracy optimized for recall:", mean_accuracy_recall, "Std Accuracy recall:", std_accuracy_recall)
print("Mean Balanced accuracy optimized for recall:", mean_balanced_accuracy_recall, "Std Balanced accuracy recall:", std_balanced_accuracy_recall)
print("Mean Specificity optimized for recall:", mean_specificity_recall, "Std Specificity recall:", std_specificity_recall)
print("Mean NPV optimized for recall:", mean_NPV_recall, "Std NPV recall:", std_NPV_recall)
print("Mean Precision optimized for recall:", mean_precision_recall, "Std Precision recall:", std_precision_recall)
print("Mean Recall optimized for recall:", mean_recall_recall, "Std Recall recall:", std_recall_recall)
print("Mean F1-score optimized for recall:", mean_f1_recall, "Std F1-score recall:", std_f1_recall)

# calculate best metrics for best threshold based on f1-score
mean_auc_f1 = fold_metrics_f1_df['AUC'].mean()
std_auc_f1 = fold_metrics_f1_df['AUC'].std()
mean_threshold_f1 = fold_metrics_f1_df['Threshold'].mean()
std_threshold_f1 = fold_metrics_f1_df['Threshold'].std()
mean_accuracy_f1 = fold_metrics_f1_df['Accuracy'].mean()
std_accuracy_f1 = fold_metrics_f1_df['Accuracy'].std()
mean_balanced_accuracy_f1 = fold_metrics_f1_df['Balanced_accuracy'].mean()
std_balanced_accuracy_f1 = fold_metrics_f1_df['Balanced_accuracy'].std()
mean_specificity_f1 = fold_metrics_f1_df['Specificity'].mean()
std_specificity_f1 = fold_metrics_f1_df['Specificity'].std()
mean_NPV_f1 = fold_metrics_f1_df['NPV'].mean()
std_NPV_f1 = fold_metrics_f1_df['NPV'].std()
mean_precision_f1 = fold_metrics_f1_df['Precision'].mean()
std_precision_f1 = fold_metrics_f1_df['Precision'].std()
mean_recall_f1 = fold_metrics_f1_df['Recall'].mean()
std_recall_f1 = fold_metrics_f1_df['Recall'].std()
mean_f1_f1 = fold_metrics_f1_df['F1-score'].mean()
std_f1_f1 = fold_metrics_f1_df['F1-score'].std()
# print metrics
print("=" * 80)
print("Mean AUC (f1):", mean_auc_f1, "Std AUC f1:", std_auc_f1)
print("Mean Threshold optimized for f1:", mean_threshold_f1, "Std Threshold f1:", std_threshold_f1)
print("Mean Accuracy optimized for f1:", mean_accuracy_f1, "Std Accuracy f1:", std_accuracy_f1)
print("Mean Balanced accuracy optimized for f1:", mean_balanced_accuracy_f1, "Std Balanced accuracy f1:", std_balanced_accuracy_f1)
print("Mean Specificity optimized for f1:", mean_specificity_f1, "Std Specificity f1:", std_specificity_f1)
print("Mean NPV optimized for f1:", mean_NPV_f1, "Std NPV f1:", std_NPV_f1)
print("Mean Precision optimized for f1:", mean_precision_f1, "Std Precision f1:", std_precision_f1)
print("Mean Recall optimized for f1:", mean_recall_f1, "Std Recall f1:", std_recall_f1)
print("Mean F1-score optimized for f1:", mean_f1_f1, "Std F1-score f1:", std_f1_f1)

# save fold_metrics_df
fold_metrics_df.to_csv(os.path.join(path_to_save_results, "fold_metrics_df_"+name_file+".csv"), index=False)
test_labels_df.to_csv(os.path.join(path_to_save_results, "test_labels_df_"+name_file+".csv"), index=False)
fold_metrics_recall_df.to_csv(os.path.join(path_to_save_results, "fold_metrics_recall_"+name_file+".csv"), index=False)
fold_metrics_f1_df.to_csv(os.path.join(path_to_save_results, "fold_metrics_f1_"+name_file+".csv"), index=False)

sys.stdout.close()