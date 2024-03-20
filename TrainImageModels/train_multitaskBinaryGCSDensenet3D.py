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
import torch
import torch.nn as nn
import functools
import operator
from sklearn.model_selection import StratifiedKFold

from torchsummary import summary

from IM_outputPrognosisGCS_Pytorch import PrognosisICH_BinaryGCS_Model

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

np.set_printoptions(precision=3)

path_to_save_model_dir = "/home/ubuntu/tenerife/data/ZZ_ICH_PrognosisMICCAI/Models/BinaryGCS_Prognosis"
path_to_save_results = '/home/ubuntu/tenerife/data/ZZ_ICH_PrognosisMICCAI/Results/BinaryGCS_Prognosis'
name_file = "BinaryGCS_Prognosis_MulticlassOutput"

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

torch.cuda.empty_cache()

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

# decide which loss function to apply
constrained_loss=False

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# create 5 fold cross validation

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
print("=" * 80)
# save best metrics for each fold
fold_metrics_df = pd.DataFrame(columns=['Fold', 'AUC', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score',
                                         'AUC_GCS', 'Balanced_accuracy_GCS', 'Accuracy_GCS', 'Specificity_GCS', 'NPV_GCS', 'Precision_GCS', 'Recall_GCS', 'F1-score_GCS'])
# save metrics for best threshold regarding recall for each fold
fold_metrics_recall_df = pd.DataFrame(columns=['Fold', 'AUC', 'Threshold', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score'])
# save metrics for best threshold regarding f1-score for each fold
fold_metrics_f1_df = pd.DataFrame(columns=['Fold', 'AUC', 'Threshold', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score'])
# save test labels and predictions for each fold
test_labels_df = pd.DataFrame(columns=['Fold', 'True_labels', 'Probabilities_labels_test', 'Predicted_labels_test_th0.5',
                                       'True_labels_GCS', 'Probabilities_labels_test_GCS', 'Predicted_labels_test_th0.5_GCS'])

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
    
    only_GCS=True

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
        print("Distribution ov values of GCS after binaryzation")
        print("Train:", gcs_train['GCS'].value_counts())
        print("Val:", gcs_val['GCS'].value_counts())
        print("Test:", gcs_test['GCS'].value_counts())
        
        # convert to tensor
        gcs_train_tensor = gcs_train['GCS'].values.tolist()
        gcs_val_tensor = gcs_val['GCS'].values.tolist()
        gcs_test_tensor = gcs_test['GCS'].values.tolist()
    
    train_files = [{"image": image_name, "label": label_name, "gcs":gcs_name} for image_name, label_name, gcs_name in zip(images_res, labels_res, gcs_train_tensor)]
    val_files = [{"image": image_name, "label": label_name, "gcs":gcs_name} for image_name, label_name, gcs_name in zip(images_val, labels_val, gcs_val_tensor)]
    test_files = [{"image": image_name, "label": label_name, "gcs":gcs_name} for image_name, label_name, gcs_name in zip(images_test, labels_test, gcs_test_tensor)]
    
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
        model = PrognosisICH_BinaryGCS_Model(image_shape=image_shape, depth=depth, spatial_dims=3, in_channels=1, num_classes_binary=1, dropout_prob=0.2)
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
        ordinal_loss_function = torch.nn.BCEWithLogitsLoss().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=True)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.98, verbose=False)
        #### understand well all the training and revise
        #### create custom model to append at the end tabular data and image features
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        best_gcs_metric = -1
        best_gcs_metric_epoch = -1
        epoch_loss_values = []
        epoch_auc_values = []
        epoch_accuracy_values = []
        epoch_gcs_accuracy_values = []
        epoch_gcs_auc_values = []
        auc_values = []
        val_gcs_auc_values = []
        val_gcs_accuracy_values = []
        accuracy_values = []

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

            all_gcs_predictions = []  # To store predicted values for GCS
            all_gcs_probabilities = []  # To store corresponding probabilities for GCS
            all_gcs_labels = []  # To store ground truth for GCS

            predictions_train=[]
            labels_train=[]
            probabilities_train=[]

            predictions_gcs_train=[]
            labels_gcs_train=[]
            probabilities_gcs_train=[]

            optimizer.zero_grad()
            i=0
            for batch_data in train_loader:
                i+=1
                step += 1
                inputs, labels, labels_gcs = batch_data["image"].to(device), batch_data["label"].to(device), batch_data["gcs"].to(device)
                # optimizer.zero_grad()
                # print(inputs.shape)
                binary_output, ordinal_output = model(inputs)
                ### sequeeze to get the right output shape
                binary_output = binary_output.squeeze()
                binary_loss = binary_loss_function(binary_output, labels.float())
                # print("Predicted probabilities binary output:", binary_output)
                ordinal_output = ordinal_output.squeeze()
                # print("Labels GCS:", labels_gcs.squeeze().long())
                ordinal_loss = ordinal_loss_function(ordinal_output, labels_gcs.float())
                # Combine losses
                if constrained_loss==True:
                    combined_loss = (binary_loss + ordinal_loss)*(1+abs(binary_loss-ordinal_loss)) # constrained_loss
                else:
                    combined_loss = binary_loss + ordinal_loss
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
                # for GCS
                labels_gcs_train.extend(labels_gcs.cpu().numpy())
                ordinal_probabilities = nn.Sigmoid()(ordinal_output)
                probabilities_gcs_train.extend(ordinal_probabilities.detach().cpu().numpy())
                # print("Predicted probabilities ordinal output GCS:", ordinal_probabilities)
                # Predicted classes
                predicted_ordinal_classes = (ordinal_probabilities >= threshold).float()
                predictions_gcs_train.extend(predicted_ordinal_classes.detach().cpu().numpy())
                # print("Predicted classes ordinal output GCS:", predicted_ordinal_classes)
                # print(labels.cpu().numpy(), train_predictions.detach().cpu().numpy())
            
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
            # Evaluation of gcs prediction
            gcs_accuracy_train = accuracy_score(labels_gcs_train, predictions_gcs_train)
            # weighted_kappa_train = cohen_kappa_score(labels_gcs_train, predictions_gcs_train, weights='quadratic')
            gcs_auc_train = roc_auc_score(labels_gcs_train, probabilities_gcs_train)
            print(f"Train epoch {epoch + 1} GCS accuracy: {gcs_accuracy_train:.3f}, GCS AUC: {gcs_auc_train:.3f}")
            # print(f"Train epoch {epoch + 1} GCS accuracy: {gcs_accuracy_train:.3f}, weighted (quadratic) kappa: {weighted_kappa_train:.3f}")
            epoch_gcs_accuracy_values.append(gcs_accuracy_train)
            epoch_gcs_auc_values.append(gcs_auc_train)
            if (epoch + 1) % val_interval == 0:
                model.eval()

                num_correct = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels, val_labels_gcs = val_data["image"].to(device), val_data["label"].to(device), val_data["gcs"].to(device)
                    with torch.no_grad():
                        binary_val_outputs, ordinal_val_outputs = model(val_images)
                        binary_val_outputs = binary_val_outputs.squeeze() ### sequeeze to get the right output shape

                        # _, predicted = val_outputs.max(1)
                        probabilities = nn.Sigmoid()(binary_val_outputs)
                        predicted = (probabilities >= threshold).float()
                        # value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(val_labels.cpu().numpy())
                        all_probabilities.extend(probabilities.cpu().numpy())

                        # for GCS
                        ordinal_val_outputs = ordinal_val_outputs.squeeze()
                        ordinal_val_probabilities = nn.Sigmoid()(ordinal_val_outputs)
                        predicted_val_ordinal_classes = (ordinal_val_probabilities >= threshold).float()
                        all_gcs_predictions.extend(predicted_val_ordinal_classes.cpu().numpy())
                        all_gcs_labels.extend(val_labels_gcs.cpu().numpy())
                        all_gcs_probabilities.extend(ordinal_val_probabilities.cpu().numpy())

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

                # for GCS
                all_gcs_predictions = np.array(all_gcs_predictions).astype(int)
                all_gcs_probabilities = np.array(all_gcs_probabilities)
                all_gcs_labels = np.array(all_gcs_labels)
                unique_gcs, counts_gcs = np.unique(all_gcs_predictions, return_counts=True)
                print("Predictions count GCS:", dict(zip(unique_gcs, counts_gcs)))
                print("Probabilities in validation GCS:", all_gcs_probabilities)
                # Calculate metrics
                roc_auc_gcs = roc_auc_score(all_gcs_labels, all_gcs_probabilities,average='weighted')
                accuracy_gcs = balanced_accuracy_score(all_gcs_labels, all_gcs_predictions)
                precision_gcs = precision_score(all_gcs_labels, all_gcs_predictions, average='weighted')
                recall_gcs = recall_score(all_gcs_labels, all_gcs_predictions, average='weighted')
                f1_gcs = f1_score(all_gcs_labels, all_gcs_predictions, average='weighted')
                val_gcs_auc_values.append(roc_auc_gcs)
                val_gcs_accuracy_values.append(accuracy_gcs)
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

                if accuracy_gcs > best_gcs_metric:
                    best_gcs_metric = accuracy_gcs
                    best_gcs_metric_epoch = epoch + 1

                # Print metrics for each epoch
                print(f"Epoch {epoch + 1}/{max_epochs} - AUC: {roc_auc:.3f}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")

                print(
                        f"current epoch: {epoch + 1} current AUC: {roc_auc:.3f}"
                        f" current accuracy: {accuracy:.3f}"
                        f" best accuracy: {best_metric:.3f}"
                        f" at epoch: {best_metric_epoch}"
                        f" current GCS AUC: {roc_auc_gcs:.3f}"
                        f" current GCS accuracy: {accuracy_gcs:.3f}"
                        f" best GCS accuracy: {best_gcs_metric:.3f}"
                        f" at epoch: {best_gcs_metric_epoch}"
                    )
                # for GCS
                print(f"Epoch {epoch + 1}/{max_epochs} - GCS AUC: {roc_auc_gcs:.3f}, GCS Accuracy: {accuracy_gcs:.3f}, GCS Precision: {precision_gcs:.3f}, GCS Recall: {recall_gcs:.3f}, GCS F1-score: {f1_gcs:.3f}")

                # writer.add_scalar("auc", roc_auc, epoch + 1)

        print(f"Training completed, best_metric in prognosis: {best_metric:.3f} at epoch: {best_metric_epoch}")
        print("-" * 50)
        # writer.close()
        plt.figure("train", (12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.subplot(1, 3, 2)
        plt.title("AUC")
        x = [i + 1 for i in range(len(epoch_auc_values))]
        y = epoch_auc_values
        plt.plot(x, y)
        x = [i + 1 for i in range(len(epoch_gcs_auc_values))]
        y = epoch_gcs_auc_values
        plt.plot(x, y)
        x = [val_interval * (i + 1) for i in range(len(auc_values))]
        y = auc_values
        plt.plot(x, y)
        x = [val_interval * (i + 1) for i in range(len(auc_values))]
        y = val_gcs_auc_values
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.legend(['train', 'train_GCS', 'val', 'val_GCS'], loc='best')
        plt.subplot(1, 3, 3)
        plt.title("Balanced ccuracy")
        x = [i + 1 for i in range(len(epoch_accuracy_values))]
        y = epoch_accuracy_values
        plt.plot(x, y)
        x = [i + 1 for i in range(len(epoch_gcs_accuracy_values))]
        y = epoch_gcs_accuracy_values
        plt.plot(x, y)
        x = [val_interval * (i + 1) for i in range(len(accuracy_values))]
        y = accuracy_values
        plt.plot(x, y)
        x = [val_interval * (i + 1) for i in range(len(accuracy_values))]
        y = val_gcs_accuracy_values
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.legend(['train', 'train_GCS', 'val', 'val_GCS'], loc='best')
        plt.savefig(os.path.join(path_to_save_results, "TrainAndVal_"+name_file+"_fold"+str(fold)+".png"))
        plt.close()

    print("Inference in test")
    threshold = 0.5

    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=8)

    model = PrognosisICH_BinaryGCS_Model(image_shape=image_shape, depth=depth, spatial_dims=3, in_channels=1, num_classes_binary=1, dropout_prob=0.2)
    model.to(device)
    model.load_state_dict(torch.load(path_to_save_model))
    model.eval()
    predicted_labels_test = []
    all_probabilities_test = []
    labels_test_tensor = []
    # for GCS
    predicted_labels_gcs_test = []
    all_probabilities_gcs_test = []
    labels_gcs_test_tensor = []
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels, test_labels_gcs = test_data["image"].to(device), test_data["label"].to(device), test_data["gcs"].to(device)
            outputs_test, ordinal_test_outputs = model(test_images)
            outputs_test = outputs_test.squeeze() ### sequeeze to get the right output shape
            probabilities_test = nn.Sigmoid()(outputs_test)
            predicted_test = (probabilities_test >= threshold).float()
            print("Probabilities test:", probabilities_test.cpu().numpy())
            all_probabilities_test.extend(probabilities_test.cpu().numpy())
            predicted_labels_test.extend(predicted_test.cpu().numpy())
            labels_test_tensor.extend(test_labels.cpu().numpy())
            #  for GCS
            ordinal_test_outputs = ordinal_test_outputs.squeeze()
            ordinal_probabilities_test = nn.Sigmoid()(ordinal_test_outputs)
            predicted_ordinal_classes_test = (ordinal_probabilities_test >= threshold).float()
            all_probabilities_gcs_test.extend(ordinal_probabilities_test.cpu().numpy())
            predicted_labels_gcs_test.extend(predicted_ordinal_classes_test.cpu().numpy())
            labels_gcs_test_tensor.extend(test_labels_gcs.cpu().numpy())


    all_probabilities_test = np.array(all_probabilities_test)
    labels_test_tensor = np.array(labels_test_tensor).astype(int)
    predicted_labels_test=np.array(predicted_labels_test).astype(int)
    # for GCS
    all_probabilities_gcs_test = np.array(all_probabilities_gcs_test)
    labels_gcs_test_tensor = np.array(labels_gcs_test_tensor).astype(int)
    predicted_labels_gcs_test=np.array(predicted_labels_gcs_test).astype(int)
    # create a fold_array that repeates the fold number as many times as test samples
    fold_array = np.full((len(labels_test_tensor)), fold)
    combined=np.column_stack((fold_array, labels_test_tensor, all_probabilities_test, predicted_labels_test, 
                              labels_gcs_test_tensor, all_probabilities_gcs_test, predicted_labels_gcs_test))
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
    # for GCS
    test_gcs_balanced_accuracy = balanced_accuracy_score(labels_gcs_test_tensor, predicted_labels_gcs_test)
    test_gcs_accuracy = accuracy_score(labels_gcs_test_tensor, predicted_labels_gcs_test)
    test_gcs_auc = roc_auc_score(labels_gcs_test_tensor, all_probabilities_gcs_test)
    test_gcs_precision = precision_score(labels_gcs_test_tensor, predicted_labels_gcs_test)
    test_gcs_recall = recall_score(labels_gcs_test_tensor, predicted_labels_gcs_test)
    test_gcs_f1 = f1_score(labels_gcs_test_tensor, predicted_labels_gcs_test)

    tn_gcs, fp_gcs, fn_gcs, tp_gcs = confusion_matrix(labels_gcs_test_tensor, predicted_labels_gcs_test, labels=[0, 1]).ravel()
    test_gcs_NPV=tn_gcs/(tn_gcs+fn_gcs)
    test_gcs_specificity=tn_gcs/(tn_gcs+fp_gcs)

    # save in fold_metrics_df
    fold_metrics_df = pd.concat([fold_metrics_df, pd.DataFrame([{'Fold': fold, 'AUC': test_auc, 
                'Balanced_accuracy': test_balanced_accuracy, 'Accuracy': test_accuracy, 'Specificity': test_specificity, 
                'NPV': test_NPV, 'Precision': test_precision, 'Recall': test_recall, 'F1-score': test_f1, 
                'AUC_GCS':test_gcs_auc, 'Balanced_accuracy_GCS': test_gcs_balanced_accuracy,
                'Accuracy_GCS': test_gcs_accuracy, 'Specificity_GCS': test_gcs_specificity, 
                'NPV_GCS': test_gcs_NPV, 'Precision_GCS': test_gcs_precision, 'Recall_GCS': test_gcs_recall,
                'F1-score_GCS': test_gcs_f1}])], ignore_index=True)

    print("Probabilities test in prognosis:", all_probabilities_test)
    unique, counts = np.unique(predicted_labels_test, return_counts=True)
    print("Predictions count test:", dict(zip(unique, counts)))
    print(f'Test Accuracy: {test_accuracy:.2%}')
    print(f'Test ROC AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

    print("Probabilities test in GCS:", all_probabilities_gcs_test)
    unique_gcs, counts_gcs = np.unique(predicted_labels_gcs_test, return_counts=True)
    print("Predictions count test GCS:", dict(zip(unique_gcs, counts_gcs)))
    print(f'Test GCS Accuracy: {test_gcs_accuracy:.2%}')
    print(f'Test GCS ROC AUC: {test_gcs_auc:.4f}, Precision: {test_gcs_precision:.4f}, Recall: {test_gcs_recall:.4f}, F1: {test_gcs_f1:.4f}')

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

    try:
        test_auc_boots=bootstrapping(y_true=labels_test_tensor, y_pred=all_probabilities_test, y_pred_threshold=predicted_labels_test, 
                    path_to_save_metrics='/home/ubuntu/tenerife/data/ICH_results/tabularData_model', 
                    metrics = 'AUC', confidence = 0.95, n_bootstraps = 1000)
        print(test_auc_boots)

        all_metrics_boots=bootstrapping(y_true=labels_test_tensor, y_pred=all_probabilities_test, y_pred_threshold=predicted_labels_test, 
                    path_to_save_metrics='/home/ubuntu/tenerife/data/ICH_results/tabularData_model', 
                    metrics = 'METRICS', confidence = 0.95, n_bootstraps = 1000)
        print(all_metrics_boots)
    except Exception as e:
        print(e)

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
# for GCS
mean_auc_gcs = fold_metrics_df['AUC_GCS'].mean()
std_auc_gcs = fold_metrics_df['AUC_GCS'].std()
mean_accuracy_gcs = fold_metrics_df['Accuracy_GCS'].mean()
std_accuracy_gcs = fold_metrics_df['Accuracy_GCS'].std()
mean_balanced_accuracy_gcs = fold_metrics_df['Balanced_accuracy_GCS'].mean()
std_balanced_accuracy_gcs = fold_metrics_df['Balanced_accuracy_GCS'].std()
mean_specificity_gcs = fold_metrics_df['Specificity_GCS'].mean()
std_specificity_gcs = fold_metrics_df['Specificity_GCS'].std()
mean_NPV_gcs = fold_metrics_df['NPV_GCS'].mean()
std_NPV_gcs = fold_metrics_df['NPV_GCS'].std()
mean_precision_gcs = fold_metrics_df['Precision_GCS'].mean()
std_precision_gcs = fold_metrics_df['Precision_GCS'].std()
mean_recall_gcs = fold_metrics_df['Recall_GCS'].mean()
std_recall_gcs = fold_metrics_df['Recall_GCS'].std()
mean_f1_gcs = fold_metrics_df['F1-score_GCS'].mean()
std_f1_gcs = fold_metrics_df['F1-score_GCS'].std()
print("=" * 80)
print("=" * 80)
# print metrics
print("Mean AUC GCS:", mean_auc_gcs, "Std AUC GCS:", std_auc_gcs)
print("Mean Accuracy GCS:", mean_accuracy_gcs, "Std Accuracy GCS:", std_accuracy_gcs)
print("Mean Balanced accuracy GCS:", mean_balanced_accuracy_gcs, "Std Balanced accuracy GCS:", std_balanced_accuracy_gcs)
print("Mean Specificity GCS:", mean_specificity_gcs, "Std Specificity GCS:", std_specificity_gcs)
print("Mean NPV GCS:", mean_NPV_gcs, "Std NPV GCS:", std_NPV_gcs)
print("Mean Precision GCS:", mean_precision_gcs, "Std Precision GCS:", std_precision_gcs)
print("Mean Recall GCS:", mean_recall_gcs, "Std Recall GCS:", std_recall_gcs)
print("Mean F1-score GCS:", mean_f1_gcs, "Std F1-score GCS:", std_f1_gcs)

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