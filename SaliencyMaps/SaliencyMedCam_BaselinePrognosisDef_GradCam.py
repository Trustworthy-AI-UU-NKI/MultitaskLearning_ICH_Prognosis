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

from pretrainDiagnosisMonaiDensenet3D import TrainDiagnosisModel

from bootstrapping import bootstrapping
from torchsummary import summary

from IM_outputPrognosisGCS_Pytorch import PrognosisICH_BinaryGCSBinaryAge_Model

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
from medcam import medcam

np.set_printoptions(precision=3)

path_to_save_model_dir = "/home/ubuntu/tenerife/data/ZZ_ICH_PrognosisMICCAI/Models/BaselineDenseNet_Prognosis/Dropout0.2WeightDecay0.0001"
path_to_save_results = '/home/ubuntu/tenerife/data/ZZ_ICH_PrognosisMICCAI/Results/BaselineDenseNet_Prognosis/Dropout0.2WeightDecay0.0001/SaliencyMapsPrognosis'
name_file_saved_model = "BaselineDenseNet_Prognosis301_40_Dropout0.2WeightDecay0.0001"
name_file="SaliencyMaps_Guided-Grad-Cam_BaselinePrognosis"

dir_to_save_saliency_maps=os.path.join(path_to_save_results,"MedCam")

sys.stdout=open(os.path.join(path_to_save_results, "run_out_"+name_file+".txt"),'w')
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

#### read diagnosis images to pretrain the DenseNet model
run_pretraining = False

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# create 5 fold cross validation

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
print("=" * 80)
# save best metrics for each fold
fold_metrics_df = pd.DataFrame(columns=['Fold', 'AUC', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score',
                                         'AUC_GCS', 'Balanced_accuracy_GCS', 'Accuracy_GCS', 'Specificity_GCS', 'NPV_GCS', 'Precision_GCS', 'Recall_GCS', 'F1-score_GCS',
                                         'AUC_Age', 'Balanced_accuracy_Age', 'Accuracy_Age', 'Specificity_Age', 'NPV_Age', 'Precision_Age', 'Recall_Age', 'F1-score_Age'])
# save metrics for best threshold regarding recall for each fold
fold_metrics_recall_df = pd.DataFrame(columns=['Fold', 'AUC', 'Threshold', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score'])
# save metrics for best threshold regarding f1-score for each fold
fold_metrics_f1_df = pd.DataFrame(columns=['Fold', 'AUC', 'Threshold', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score'])
# save test labels and predictions for each fold
test_labels_df = pd.DataFrame(columns=['Fold', 'True_labels', 'Probabilities_labels_test', 'Predicted_labels_test_th0.5',
                                       'True_labels_GCS', 'Probabilities_labels_test_GCS', 'Predicted_labels_test_th0.5_GCS',
                                       'True_labels_Age', 'Probabilities_labels_test_Age', 'Predicted_labels_test_th0.5_Age'])

# Iterate over the folds
for fold, (train_index, test_index) in enumerate(skf.split(images_all, labels_all)):
    print(f"Fold {fold + 1}:")
    path_to_save_saliency_maps = dir_to_save_saliency_maps + "/Fold" + str(fold)
    if not os.path.exists(path_to_save_saliency_maps):
        os.makedirs(path_to_save_saliency_maps)

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

    path_to_pretrained_model="/home/ubuntu/tenerife/data/ICH_diagnosis/Models/Densenet3DMonai/DiagnosisModelICH_Densenet_pretrained_fold"+str(fold)+".pth"
    if run_pretraining:
        # if path_to_pretrained_model does not exist, train the model
        if not os.path.isfile(path_to_pretrained_model):
            diagnosis_directory = "/home/ubuntu/tenerife/data/ICH_diagnosis/3D_dataset/PREPROCESSED IMAGES"
            # read images and corresponding labels from directory
            goodDiagnosis_images = sorted(os.listdir(os.path.join(diagnosis_directory, "CONTROLS")))
            print(f"Control images (0): {len(goodDiagnosis_images)}")
            PoorDiagnosis_images = sorted(os.listdir(os.path.join(diagnosis_directory, "ICH")))
            print(f"ICH images (1): {len(PoorDiagnosis_images)}")

            # read CLINICAL DATA
            clinical_data_diagnosis = pd.read_csv('/home/ubuntu/tenerife/data/ICH_diagnosis/3D_dataset/CSV/ALL_PATIENTS.csv', delimiter=',')
            # filter clinical_data_diagnosis to remove the IDs that are in test in order to avoid data leakage
            print("Len clinical data diagnosis before removing patients in test:", len(clinical_data_diagnosis))
            ids_test = X_test['PatientID'].values
            renamed_ids=[f"HIC_{id}" for id in ids_test]
            print("Removed ids: ", clinical_data_diagnosis.loc[clinical_data_diagnosis['Patient'].isin(renamed_ids), 'Patient'].values)
            clinical_data_diagnosis = clinical_data_diagnosis[~clinical_data_diagnosis['Patient'].isin(renamed_ids)]
            print("Len clinical data diagnosis after removing patients in test:", len(clinical_data_diagnosis))
            clinical_data_diagnosis.reset_index(drop=True, inplace=True)

            images_all_diagnosis=[]
            labels_all_diagnosis=[]
            # loop over Patient in clinical_data
            for i in range(len(clinical_data_diagnosis)):
                # get patient ID
                patientID_diagnosis = clinical_data_diagnosis['Patient'][i]
                # get label
                label_diagnosis= clinical_data_diagnosis['Label'][i]
                # read corresponding image
                if label_diagnosis==0:
                    # check that image exists
                    if os.path.isfile(os.path.join(diagnosis_directory, "CONTROLS", str(patientID_diagnosis) + ".npy")):
                        images_all_diagnosis.append(os.path.join(diagnosis_directory, "CONTROLS", str(patientID_diagnosis) + ".npy"))
                        labels_all_diagnosis.append(label_diagnosis)
                elif label_diagnosis==1:
                    # check that image exists
                    if os.path.isfile(os.path.join(diagnosis_directory, "ICH", str(patientID_diagnosis) + ".npy")):
                        images_all_diagnosis.append(os.path.join(diagnosis_directory, "ICH", str(patientID_diagnosis) + ".npy"))
                        labels_all_diagnosis.append(label_diagnosis)
                else:
                    print("ERROR: for Patient", patientID, "label not found")
            print("Total number of images diagnosis:", len(images_all_diagnosis))
            print("Total number of labels diagnosis:", len(labels_all_diagnosis))
            images_all_diagnosis=np.array(images_all_diagnosis)
            labels_all_diagnosis=np.array(labels_all_diagnosis)
            pretrainedModel=TrainDiagnosisModel()
            print("Starting now training of the diagnostic model")
            pretrainedModel.train(images_all=images_all_diagnosis, labels_all=labels_all_diagnosis, 
                                loc_save_model=path_to_pretrained_model, fold=fold, str_cuda=str_cuda, seed=1)
            sys.stdout = original_stdout
            print("Finished training of the diagnostic model")

    X_test_patients = X_test['PatientID']
    
    # remove PatientID from X_train, X_test and X_val
    X_train = X_train.drop(columns=['PatientID', 'Label (poor_prognosis)'])
    X_test = X_test.drop(columns=['PatientID', 'Label (poor_prognosis)'])
    X_val = X_val.drop(columns=['PatientID', 'Label (poor_prognosis)'])
    
    only_GCS=True
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
    
    train_files = [{"image": image_name, "label": label_name, "gcs":gcs_name, "age": age_name} for image_name, label_name, gcs_name, age_name in zip(images_res, labels_res, gcs_train_tensor, age_train_tensor)]
    val_files = [{"image": image_name, "label": label_name, "gcs":gcs_name, "age": age_name} for image_name, label_name, gcs_name, age_name in zip(images_val, labels_val, gcs_val_tensor, age_val_tensor)]
    test_files = [{"image": image_name, "label": label_name, "gcs":gcs_name, "age": age_name,
                   "patient":patient} for image_name, label_name, gcs_name, age_name,
                     patient in zip(images_test, labels_test, gcs_test_tensor, age_test_tensor, X_test_patients)]

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
    path_to_save_model=os.path.join(path_to_save_model_dir,
                                                name_file_saved_model+"_fold"+str(fold)+".pth")
    
    print("Inference in test")
    threshold = 0.5

    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=0.2)
    model.to(device)
    model.load_state_dict(torch.load(path_to_save_model))
    layer='features.transition3.conv'
    model = medcam.inject(model, backend='ggcam', layer=layer, output_dir=path_to_save_saliency_maps, save_maps=True, return_score=False, cudnn=True)
    map_type = "Guided-Grad-Cam"
    # Note: Guided-Backpropagation ignores parameter layer.
    print("Layer names in model:", medcam.get_layers(model))
    model.eval()
    predicted_labels_test = []
    all_probabilities_test = []
    labels_test_tensor = []
    # for GCS
    predicted_labels_gcs_test = []
    all_probabilities_gcs_test = []
    labels_gcs_test_tensor = []
    # for age
    predicted_labels_age_test = []
    all_probabilities_age_test = []
    labels_age_test_tensor = []
    i = 0
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels, test_labels_gcs, test_labels_age, test_patientID = test_data["image"].to(device), test_data["label"].to(device), test_data["gcs"], test_data["age"], test_data["patient"]
            test_patientID = test_patientID.item()
            print("Computing saliency maps for patient:", test_patientID)
            test_labels = test_labels.unsqueeze(1).float()
            if test_labels==0:
                label_name="GOOD_PROGNOSIS"
            else:
                label_name="POOR_PROGNOSIS"
            outputs_test = model(test_images)
            # rename saliency maps file to patientID
            shutil.move(os.path.join(path_to_save_saliency_maps, layer, "attention_map_"+str(i)+"_0_0.nii.gz"), os.path.join(path_to_save_saliency_maps, layer, str(test_patientID)+"_"+map_type+"-label-"+label_name+".nii.gz"))
            i += 1
            # outputs_test = model(test_images)
            # outputs_test = outputs_test.squeeze() ### sequeeze to get the right output shape
            # probabilities_test = nn.Sigmoid()(outputs_test)
            # predicted_test = (probabilities_test >= threshold).float()
            # print("Probabilities test:", probabilities_test.cpu().numpy())
            # all_probabilities_test.extend(probabilities_test.cpu().numpy())
            # predicted_labels_test.extend(predicted_test.cpu().numpy())
            # labels_test_tensor.extend(test_labels.cpu().numpy())
            # #  for GCS
            # ordinal_test_outputs = ordinal_test_outputs.squeeze()
            # ordinal_probabilities_test = nn.Sigmoid()(ordinal_test_outputs)
            # predicted_ordinal_classes_test = (ordinal_probabilities_test >= threshold).float()
            # all_probabilities_gcs_test.extend(ordinal_probabilities_test.cpu().numpy())
            # predicted_labels_gcs_test.extend(predicted_ordinal_classes_test.cpu().numpy())
            # labels_gcs_test_tensor.extend(test_labels_gcs.cpu().numpy())
            # # for age
            # outputs_age = outputs_age.squeeze()
            # probabilities_age_test = nn.Sigmoid()(outputs_age)
            # predicted_age_test = (probabilities_age_test >= threshold).float()
            # all_probabilities_age_test.extend(probabilities_age_test.cpu().numpy())
            # predicted_labels_age_test.extend(predicted_age_test.cpu().numpy())
            # labels_age_test_tensor.extend(test_labels_age.cpu().numpy())
            

    # all_probabilities_test = np.array(all_probabilities_test)
    # labels_test_tensor = np.array(labels_test_tensor).astype(int)
    # predicted_labels_test=np.array(predicted_labels_test).astype(int)
    # # for GCS
    # all_probabilities_gcs_test = np.array(all_probabilities_gcs_test)
    # labels_gcs_test_tensor = np.array(labels_gcs_test_tensor).astype(int)
    # predicted_labels_gcs_test=np.array(predicted_labels_gcs_test).astype(int)
    # # for age
    # all_probabilities_age_test = np.array(all_probabilities_age_test)
    # labels_age_test_tensor = np.array(labels_age_test_tensor).astype(int)
    # predicted_labels_age_test=np.array(predicted_labels_age_test).astype(int)
    # # create a fold_array that repeates the fold number as many times as test samples
    # fold_array = np.full((len(labels_test_tensor)), fold)
    # combined=np.column_stack((fold_array, labels_test_tensor, all_probabilities_test, predicted_labels_test, 
    #                           labels_gcs_test_tensor, all_probabilities_gcs_test, predicted_labels_gcs_test,
    #                           labels_age_test_tensor, all_probabilities_age_test, predicted_labels_age_test))
    # test_labels_df = pd.concat([test_labels_df, pd.DataFrame(combined, columns=test_labels_df.columns)], ignore_index=True)
    # test_balanced_accuracy = balanced_accuracy_score(labels_test_tensor, predicted_labels_test)
    # test_accuracy = accuracy_score(labels_test_tensor, predicted_labels_test)
    # test_auc = roc_auc_score(labels_test_tensor, all_probabilities_test)
    # test_precision = precision_score(labels_test_tensor, predicted_labels_test)
    # test_recall = recall_score(labels_test_tensor, predicted_labels_test)
    # test_f1 = f1_score(labels_test_tensor, predicted_labels_test)

    # tn, fp, fn, tp = confusion_matrix(labels_test_tensor, predicted_labels_test, labels=[0, 1]).ravel()
    # test_NPV=tn/(tn+fn)
    # test_specificity=tn/(tn+fp)
    # # for GCS
    # test_gcs_balanced_accuracy = balanced_accuracy_score(labels_gcs_test_tensor, predicted_labels_gcs_test)
    # test_gcs_accuracy = accuracy_score(labels_gcs_test_tensor, predicted_labels_gcs_test)
    # test_gcs_auc = roc_auc_score(labels_gcs_test_tensor, all_probabilities_gcs_test)
    # test_gcs_precision = precision_score(labels_gcs_test_tensor, predicted_labels_gcs_test)
    # test_gcs_recall = recall_score(labels_gcs_test_tensor, predicted_labels_gcs_test)
    # test_gcs_f1 = f1_score(labels_gcs_test_tensor, predicted_labels_gcs_test)

    # tn_gcs, fp_gcs, fn_gcs, tp_gcs = confusion_matrix(labels_gcs_test_tensor, predicted_labels_gcs_test, labels=[0, 1]).ravel()
    # test_gcs_NPV=tn_gcs/(tn_gcs+fn_gcs)
    # test_gcs_specificity=tn_gcs/(tn_gcs+fp_gcs)
    # # for age
    # test_age_balanced_accuracy = balanced_accuracy_score(labels_age_test_tensor, predicted_labels_age_test)
    # test_age_accuracy = accuracy_score(labels_age_test_tensor, predicted_labels_age_test)
    # test_age_auc = roc_auc_score(labels_age_test_tensor, all_probabilities_age_test)
    # test_age_precision = precision_score(labels_age_test_tensor, predicted_labels_age_test)
    # test_age_recall = recall_score(labels_age_test_tensor, predicted_labels_age_test)
    # test_age_f1 = f1_score(labels_age_test_tensor, predicted_labels_age_test)

    # tn_age, fp_age, fn_age, tp_age = confusion_matrix(labels_age_test_tensor, predicted_labels_age_test, labels=[0, 1]).ravel()
    # test_age_NPV=tn_age/(tn_age+fn_age)
    # test_age_specificity=tn_age/(tn_age+fp_age)

    break
    print("=" * 80)
    

print("=" * 80)
# calculate mean and std of metrics for all folds
# mean_auc = fold_metrics_df['AUC'].mean()
# std_auc = fold_metrics_df['AUC'].std()
# mean_accuracy = fold_metrics_df['Accuracy'].mean()
# std_accuracy = fold_metrics_df['Accuracy'].std()
# mean_balanced_accuracy = fold_metrics_df['Balanced_accuracy'].mean()
# std_balanced_accuracy = fold_metrics_df['Balanced_accuracy'].std()
# mean_specificity = fold_metrics_df['Specificity'].mean()
# std_specificity = fold_metrics_df['Specificity'].std()
# mean_NPV = fold_metrics_df['NPV'].mean()
# std_NPV = fold_metrics_df['NPV'].std()
# mean_precision = fold_metrics_df['Precision'].mean()
# std_precision = fold_metrics_df['Precision'].std()
# mean_recall = fold_metrics_df['Recall'].mean()
# std_recall = fold_metrics_df['Recall'].std()
# mean_f1 = fold_metrics_df['F1-score'].mean()
# std_f1 = fold_metrics_df['F1-score'].std()
# # print metrics
# print("Mean AUC:", mean_auc, "Std AUC:", std_auc)
# print("Mean Accuracy:", mean_accuracy, "Std Accuracy:", std_accuracy)
# print("Mean Balanced accuracy:", mean_balanced_accuracy, "Std Balanced accuracy:", std_balanced_accuracy)
# print("Mean Specificity:", mean_specificity, "Std Specificity:", std_specificity)
# print("Mean NPV:", mean_NPV, "Std NPV:", std_NPV)
# print("Mean Precision:", mean_precision, "Std Precision:", std_precision)
# print("Mean Recall:", mean_recall, "Std Recall:", std_recall)
# print("Mean F1-score:", mean_f1, "Std F1-score:", std_f1)
# # for GCS
# print("=" * 80)
# mean_auc_gcs = fold_metrics_df['AUC_GCS'].mean()
# std_auc_gcs = fold_metrics_df['AUC_GCS'].std()
# mean_accuracy_gcs = fold_metrics_df['Accuracy_GCS'].mean()
# std_accuracy_gcs = fold_metrics_df['Accuracy_GCS'].std()
# mean_balanced_accuracy_gcs = fold_metrics_df['Balanced_accuracy_GCS'].mean()
# std_balanced_accuracy_gcs = fold_metrics_df['Balanced_accuracy_GCS'].std()
# mean_specificity_gcs = fold_metrics_df['Specificity_GCS'].mean()
# std_specificity_gcs = fold_metrics_df['Specificity_GCS'].std()
# mean_NPV_gcs = fold_metrics_df['NPV_GCS'].mean()
# std_NPV_gcs = fold_metrics_df['NPV_GCS'].std()
# mean_precision_gcs = fold_metrics_df['Precision_GCS'].mean()
# std_precision_gcs = fold_metrics_df['Precision_GCS'].std()
# mean_recall_gcs = fold_metrics_df['Recall_GCS'].mean()
# std_recall_gcs = fold_metrics_df['Recall_GCS'].std()
# mean_f1_gcs = fold_metrics_df['F1-score_GCS'].mean()
# std_f1_gcs = fold_metrics_df['F1-score_GCS'].std()
# # print metrics
# print("Mean AUC GCS:", mean_auc_gcs, "Std AUC GCS:", std_auc_gcs)
# print("Mean Accuracy GCS:", mean_accuracy_gcs, "Std Accuracy GCS:", std_accuracy_gcs)
# print("Mean Balanced accuracy GCS:", mean_balanced_accuracy_gcs, "Std Balanced accuracy GCS:", std_balanced_accuracy_gcs)
# print("Mean Specificity GCS:", mean_specificity_gcs, "Std Specificity GCS:", std_specificity_gcs)
# print("Mean NPV GCS:", mean_NPV_gcs, "Std NPV GCS:", std_NPV_gcs)
# print("Mean Precision GCS:", mean_precision_gcs, "Std Precision GCS:", std_precision_gcs)
# print("Mean Recall GCS:", mean_recall_gcs, "Std Recall GCS:", std_recall_gcs)
# print("Mean F1-score GCS:", mean_f1_gcs, "Std F1-score GCS:", std_f1_gcs)
# print("=" * 80)
# # for age
# mean_auc_age = fold_metrics_df['AUC_Age'].mean()
# std_auc_age = fold_metrics_df['AUC_Age'].std()
# mean_accuracy_age = fold_metrics_df['Accuracy_Age'].mean()
# std_accuracy_age = fold_metrics_df['Accuracy_Age'].std()
# mean_balanced_accuracy_age = fold_metrics_df['Balanced_accuracy_Age'].mean()
# std_balanced_accuracy_age = fold_metrics_df['Balanced_accuracy_Age'].std()
# mean_specificity_age = fold_metrics_df['Specificity_Age'].mean()
# std_specificity_age = fold_metrics_df['Specificity_Age'].std()
# mean_NPV_age = fold_metrics_df['NPV_Age'].mean()
# std_NPV_age = fold_metrics_df['NPV_Age'].std()
# mean_precision_age = fold_metrics_df['Precision_Age'].mean()
# std_precision_age = fold_metrics_df['Precision_Age'].std()
# mean_recall_age = fold_metrics_df['Recall_Age'].mean()
# std_recall_age = fold_metrics_df['Recall_Age'].std()
# mean_f1_age = fold_metrics_df['F1-score_Age'].mean()
# std_f1_age = fold_metrics_df['F1-score_Age'].std()
# # print metrics
# print("Mean AUC Age:", mean_auc_age, "Std AUC Age:", std_auc_age)
# print("Mean Accuracy Age:", mean_accuracy_age, "Std Accuracy Age:", std_accuracy_age)
# print("Mean Balanced accuracy Age:", mean_balanced_accuracy_age, "Std Balanced accuracy Age:", std_balanced_accuracy_age)
# print("Mean Specificity Age:", mean_specificity_age, "Std Specificity Age:", std_specificity_age)
# print("Mean NPV Age:", mean_NPV_age, "Std NPV Age:", std_NPV_age)
# print("Mean Precision Age:", mean_precision_age, "Std Precision Age:", std_precision_age)
# print("Mean Recall Age:", mean_recall_age, "Std Recall Age:", std_recall_age)
# print("Mean F1-score Age:", mean_f1_age, "Std F1-score Age:", std_f1_age)
# print("=" * 80)
# print("=" * 80)
# # calculate best metrics for best threshold based on recall
# mean_auc_recall = fold_metrics_recall_df['AUC'].mean()
# std_auc_recall = fold_metrics_recall_df['AUC'].std()
# mean_threshold_recall = fold_metrics_recall_df['Threshold'].mean()
# std_threshold_recall = fold_metrics_recall_df['Threshold'].std()
# mean_accuracy_recall = fold_metrics_recall_df['Accuracy'].mean()
# std_accuracy_recall = fold_metrics_recall_df['Accuracy'].std()
# mean_balanced_accuracy_recall = fold_metrics_recall_df['Balanced_accuracy'].mean()
# std_balanced_accuracy_recall = fold_metrics_recall_df['Balanced_accuracy'].std()
# mean_specificity_recall = fold_metrics_recall_df['Specificity'].mean()
# std_specificity_recall = fold_metrics_recall_df['Specificity'].std()
# mean_NPV_recall = fold_metrics_recall_df['NPV'].mean()
# std_NPV_recall = fold_metrics_recall_df['NPV'].std()
# mean_precision_recall = fold_metrics_recall_df['Precision'].mean()
# std_precision_recall = fold_metrics_recall_df['Precision'].std()
# mean_recall_recall = fold_metrics_recall_df['Recall'].mean()
# std_recall_recall = fold_metrics_recall_df['Recall'].std()
# mean_f1_recall = fold_metrics_recall_df['F1-score'].mean()
# std_f1_recall = fold_metrics_recall_df['F1-score'].std()
# # print metrics
# print("=" * 80)
# print("Mean AUC (recall):", mean_auc_recall, "Std AUC recall:", std_auc_recall)
# print("Mean Threshold optimized for recall:", mean_threshold_recall, "Std Threshold recall:", std_threshold_recall)
# print("Mean Accuracy optimized for recall:", mean_accuracy_recall, "Std Accuracy recall:", std_accuracy_recall)
# print("Mean Balanced accuracy optimized for recall:", mean_balanced_accuracy_recall, "Std Balanced accuracy recall:", std_balanced_accuracy_recall)
# print("Mean Specificity optimized for recall:", mean_specificity_recall, "Std Specificity recall:", std_specificity_recall)
# print("Mean NPV optimized for recall:", mean_NPV_recall, "Std NPV recall:", std_NPV_recall)
# print("Mean Precision optimized for recall:", mean_precision_recall, "Std Precision recall:", std_precision_recall)
# print("Mean Recall optimized for recall:", mean_recall_recall, "Std Recall recall:", std_recall_recall)
# print("Mean F1-score optimized for recall:", mean_f1_recall, "Std F1-score recall:", std_f1_recall)

# # calculate best metrics for best threshold based on f1-score
# mean_auc_f1 = fold_metrics_f1_df['AUC'].mean()
# std_auc_f1 = fold_metrics_f1_df['AUC'].std()
# mean_threshold_f1 = fold_metrics_f1_df['Threshold'].mean()
# std_threshold_f1 = fold_metrics_f1_df['Threshold'].std()
# mean_accuracy_f1 = fold_metrics_f1_df['Accuracy'].mean()
# std_accuracy_f1 = fold_metrics_f1_df['Accuracy'].std()
# mean_balanced_accuracy_f1 = fold_metrics_f1_df['Balanced_accuracy'].mean()
# std_balanced_accuracy_f1 = fold_metrics_f1_df['Balanced_accuracy'].std()
# mean_specificity_f1 = fold_metrics_f1_df['Specificity'].mean()
# std_specificity_f1 = fold_metrics_f1_df['Specificity'].std()
# mean_NPV_f1 = fold_metrics_f1_df['NPV'].mean()
# std_NPV_f1 = fold_metrics_f1_df['NPV'].std()
# mean_precision_f1 = fold_metrics_f1_df['Precision'].mean()
# std_precision_f1 = fold_metrics_f1_df['Precision'].std()
# mean_recall_f1 = fold_metrics_f1_df['Recall'].mean()
# std_recall_f1 = fold_metrics_f1_df['Recall'].std()
# mean_f1_f1 = fold_metrics_f1_df['F1-score'].mean()
# std_f1_f1 = fold_metrics_f1_df['F1-score'].std()
# # print metrics
# print("=" * 80)
# print("Mean AUC (f1):", mean_auc_f1, "Std AUC f1:", std_auc_f1)
# print("Mean Threshold optimized for f1:", mean_threshold_f1, "Std Threshold f1:", std_threshold_f1)
# print("Mean Accuracy optimized for f1:", mean_accuracy_f1, "Std Accuracy f1:", std_accuracy_f1)
# print("Mean Balanced accuracy optimized for f1:", mean_balanced_accuracy_f1, "Std Balanced accuracy f1:", std_balanced_accuracy_f1)
# print("Mean Specificity optimized for f1:", mean_specificity_f1, "Std Specificity f1:", std_specificity_f1)
# print("Mean NPV optimized for f1:", mean_NPV_f1, "Std NPV f1:", std_NPV_f1)
# print("Mean Precision optimized for f1:", mean_precision_f1, "Std Precision f1:", std_precision_f1)
# print("Mean Recall optimized for f1:", mean_recall_f1, "Std Recall f1:", std_recall_f1)
# print("Mean F1-score optimized for f1:", mean_f1_f1, "Std F1-score f1:", std_f1_f1)

# # save fold_metrics_df
# fold_metrics_df.to_csv(os.path.join(path_to_save_results, "fold_metrics_df_"+name_file+".csv"), index=False)
# test_labels_df.to_csv(os.path.join(path_to_save_results, "test_labels_df_"+name_file+".csv"), index=False)
# fold_metrics_recall_df.to_csv(os.path.join(path_to_save_results, "fold_metrics_recall_"+name_file+".csv"), index=False)
# fold_metrics_f1_df.to_csv(os.path.join(path_to_save_results, "fold_metrics_f1_"+name_file+".csv"), index=False)

sys.stdout.close()