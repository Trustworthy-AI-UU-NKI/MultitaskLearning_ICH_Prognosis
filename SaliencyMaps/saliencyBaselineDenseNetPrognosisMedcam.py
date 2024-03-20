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

path_to_save_model_dir = "/home/ubuntu/tenerife/data/ICH_models/ICH_3DMonaiClassificationDenseNet"
path_to_save_results = "/home/ubuntu/tenerife/data/ICH_results/3DClassificationImageModelMonai/Saliency_maps"
name_file_saved_model = "PrognosisModelICH_DenseNet_SkullStripping_SmallDataAug"
name_file = "SaliencyMaps_Baseline301_40Prognosis_DenseNet"

# save prints in a txt file
sys.stdout=open(os.path.join(path_to_save_results, "run_out_"+name_file+".txt"),'w')

dir_to_save_saliency_maps=os.path.join(path_to_save_results,"MedCam")

pin_memory = torch.cuda.is_available()
str_cuda="cuda:1"
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
fold_metrics_df = pd.DataFrame(columns=['Fold', 'AUC', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score'])
# save metrics for best threshold regarding recall for each fold
fold_metrics_recall_df = pd.DataFrame(columns=['Fold', 'AUC', 'Threshold', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score'])
# save metrics for best threshold regarding f1-score for each fold
fold_metrics_f1_df = pd.DataFrame(columns=['Fold', 'AUC', 'Threshold', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score'])
# save test labels and predictions for each fold
test_labels_df = pd.DataFrame(columns=['Fold', 'True_labels', 'Probabilities_labels_test', 'Predicted_labels_test_th0.5'])

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

    X_test_patients = X_test['PatientID']
   
    train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images_res, labels_res)]
    val_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images_val, labels_val)]
    test_files = [{"image": image_name, "label": label_name, "patient":patient} for image_name, label_name, patient in zip(images_test, labels_test, X_test_patients)]
    
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

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=2, pin_memory=pin_memory)
    path_to_save_model=os.path.join(path_to_save_model_dir,
                                                name_file_saved_model+"_fold"+str(fold)+".pth")

    print("Inference in test")
    threshold = 0.5

    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=0.2)
    model.to(device)
    model.load_state_dict(torch.load(path_to_save_model))
    model = medcam.inject(model, backend='gbp', output_dir=path_to_save_saliency_maps, save_maps=True, return_score=False, cudnn=True)
    map_type = "guidedBackProp"
    # Note: Guided-Backpropagation ignores parameter layer.
    print("Layer names in model:", medcam.get_layers(model))
    model.eval()
    predicted_labels_test = []
    all_probabilities_test = []
    labels_test_tensor = []
    predicted_labels_test = []
    all_probabilities_test = []
    labels_test_tensor = []
    i = 0
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels, test_patientID = test_data["image"].to(device), test_data["label"].to(device), test_data["patient"]
            test_patientID = test_patientID.item()
            print("Computing saliency maps for patient:", test_patientID)
            test_labels = test_labels.unsqueeze(1).float()
            if test_labels==0:
                label_name="GOOD_PROGNOSIS"
            else:
                label_name="POOR_PROGNOSIS"
            outputs_test = model(test_images)
            # rename saliency maps file to patientID
            shutil.move(os.path.join(path_to_save_saliency_maps, "attention_map_"+str(i)+"_0_0.nii.gz"), os.path.join(path_to_save_saliency_maps, str(test_patientID)+"_"+map_type+"-label-"+label_name+".nii.gz"))
            i += 1
    #         outputs_test = outputs_test.squeeze() ### sequeeze to get the right output shape
    #         probabilities_test = nn.Sigmoid()(outputs_test)
    #         predicted_test = (probabilities_test >= threshold).float()
    #         print("Probabilities test:", probabilities_test.cpu().numpy())
    #         all_probabilities_test.extend(probabilities_test.cpu().numpy())
    #         predicted_labels_test.extend(predicted_test.cpu().numpy())
    #         labels_test_tensor.extend(test_labels.cpu().numpy())

    # all_probabilities_test = np.array(all_probabilities_test)
    # labels_test_tensor = np.array(labels_test_tensor).astype(int)
    # predicted_labels_test=np.array(predicted_labels_test).astype(int)
    # # create a fold_array that repeates the fold number as many times as test samples
    # fold_array = np.full((len(labels_test_tensor)), fold)
    # combined=np.column_stack((fold_array, labels_test_tensor, all_probabilities_test, predicted_labels_test))
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

    # # save in fold_metrics_df
    # fold_metrics_df = pd.concat([fold_metrics_df, pd.DataFrame([{'Fold': fold, 'AUC': test_auc, 
    #             'Balanced_accuracy': test_balanced_accuracy, 'Accuracy': test_accuracy, 'Specificity': test_specificity, 
    #             'NPV': test_NPV, 'Precision': test_precision, 'Recall': test_recall, 'F1-score': test_f1}])], ignore_index=True)

    # print("Probabilities test in prognosis:", all_probabilities_test)
    # unique, counts = np.unique(predicted_labels_test, return_counts=True)
    # print("Predictions count test:", dict(zip(unique, counts)))
    # print(f'Test Accuracy: {test_accuracy:.2%}')
    # print(f'Test ROC AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

    # # Save predicted labels for test set
    # predicted_labels_df = pd.DataFrame({'True Labels': labels_test, 'Predicted Labels': predicted_labels_test})
    # # predicted_labels_df.to_csv('predicted_labels.csv', index=False)

    # # plot summary report
    # print(classification_report(labels_test_tensor, predicted_labels_test, target_names=['Good prognosis', 'Poor prognosis']))

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
    break
    torch.cuda.empty_cache()
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

sys.stdout.close()