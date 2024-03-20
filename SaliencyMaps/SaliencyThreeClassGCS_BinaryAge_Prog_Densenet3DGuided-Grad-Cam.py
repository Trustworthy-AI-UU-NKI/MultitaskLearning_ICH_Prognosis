import logging
import os
import sys
import shutil
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import functools
import operator
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from IM_outputPrognosisGCS_Pytorch import PrognosisICH_ThreeClassGCSBinaryAge_Model

from bootstrapping import bootstrapping
from torchsummary import summary
from imbalancedOrdinalMetric import imbalanced_ordinal_classification_index, uoc_index

from IM_outputPrognosisGCS_Pytorch import BinaryGCS_Model

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

path_to_save_model_dir = "/home/ubuntu/tenerife/data/ZZ_ICH_PrognosisMICCAI/Models/ThreeClassGCS_BinaryAge_Prognosis_SameHP"
path_to_save_results = '/home/ubuntu/tenerife/data/ZZ_ICH_PrognosisMICCAI/Results/ThreeClassGCS_BinaryAge_Prognosis_SameHP/SaliencyMapsPrognosis'
name_file_saved_model = "ThreeClassGCS_BinaryAge_Prognosis_SAMEPARAMETERS_asBinaryGCS_BinaryAge_Prognosis_No_pos_weight"
name_file="SaliencyMaps_ThreeClassGCS_BinaryAge_Prognosis_GradCAM"

dir_to_save_saliency_maps=os.path.join(path_to_save_results,"MedCam")

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

#### read diagnosis images to pretrain the DenseNet model
run_pretraining = False

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# create 5 fold cross validation

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
print("=" * 80)
# save best metrics for each fold
fold_metrics_df = pd.DataFrame(columns=['Fold', 'Balanced_accuracy_GCS', 'Accuracy_GCS', 'MAE_GCS', 'RMSE_GCS', 'UOC_index_GCS', 'cohen_kappa_score_GCS',
                                        'AUC', 'Balanced_accuracy', 'Accuracy', 'Specificity', 'NPV', 'Precision', 'Recall', 'F1-score',
                                         'AUC_Age', 'Balanced_accuracy_Age', 'Accuracy_Age', 'Specificity_Age', 'NPV_Age', 'Precision_Age', 'Recall_Age', 'F1-score_Age'])
# save test labels and predictions for each fold
test_labels_df = pd.DataFrame(columns=['Fold', 'True_labels_GCS', 'Predicted_labels_test_th0.5_GCS',
                                       'True_labels', 'Probabilities_labels_test', 'Predicted_labels_test_th0.5',
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
        gcs_train['GCS'] = gcs_train['GCS'].apply(lambda x: 2 if x<=8 else (1 if x <= 12 else 0))
        gcs_test['GCS'] = gcs_test['GCS'].apply(lambda x: 2 if x<=8 else (1 if x <= 12 else 0))
        gcs_val['GCS'] = gcs_val['GCS'].apply(lambda x: 2 if x<=8 else (1 if x <= 12 else 0))
        # print("GCS train before encoding:", gcs_train['GCS'])
        # three class encode GCS using ordinal_encode
        gcs_train_tensor = ordinal_encode(gcs_train['GCS'].values, 2)
        gcs_val_tensor = ordinal_encode(gcs_val['GCS'].values, 2)
        gcs_test_tensor = ordinal_encode(gcs_test['GCS'].values, 2)
        print("Number of ordinal categorical variables:", len(ordinal_categorical_var))
        print("Distribution ov values of GCS after binaryzation")
        print("Train:", gcs_train['GCS'].value_counts())
        print("Val:", gcs_val['GCS'].value_counts())
        print("Test:", gcs_test['GCS'].value_counts())
        # print("GCS train tensor after encoding:", gcs_train_tensor)
        # for the weights:
        tuples_list = [tuple(item) for item in gcs_train_tensor]
        occurrences = Counter(tuples_list)
        occurrences_dict = dict(occurrences)
        print("Occurrences of each class in the training set:", occurrences_dict)
        total_negative_first = occurrences_dict[(0, 0)]
        total_positive_first = occurrences_dict[(1, 0)] + occurrences_dict[(1, 1)]
        total_negative_second = occurrences_dict[(1, 0)]
        total_positive_second = occurrences_dict[(1, 1)]
        # Calculate the weight for the positive class
        pos_weight_gcs_first = total_negative_first / total_positive_first
        pos_weight_gcs_second = total_negative_second / total_positive_second
        pos_weight_gcs = torch.tensor([pos_weight_gcs_first, pos_weight_gcs_second], dtype=torch.float32)
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
        # print("Number of ordinal categorical variables:", len(ordinal_regression_var))
        # print("Distribution of values of Age after binaryzation")
        # print("Train:", age_train['Age'].value_counts())
        # print("Val:", age_val['Age'].value_counts())
        # print("Test:", age_test['Age'].value_counts())
        
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
    
    train_files = [{"image": image_name, "label": label_name, "gcs":gcs_name, "age": age_name} for image_name, label_name, gcs_name, age_name in zip(images_res, labels_res, gcs_train_tensor, age_train_tensor)]
    val_files = [{"image": image_name, "label": label_name, "gcs":gcs_name, "age": age_name} for image_name, label_name, gcs_name, age_name in zip(images_val, labels_val, gcs_val_tensor, age_val_tensor)]
    test_files = [{"image": image_name, "label": label_name, "gcs":gcs_name, "age": age_name,
                   "patient":patient} for image_name, label_name, gcs_name, age_name,
                     patient in zip(images_test, labels_test, gcs_test_tensor, age_test_tensor, X_test_patients)]
    
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
                                                "DenseNet_"+name_file_saved_model+"_fold"+str(fold)+".pth")

    print("Inference in test")

    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)
    threshold = 0.5

    model = PrognosisICH_ThreeClassGCSBinaryAge_Model(image_shape=image_shape, depth=depth, spatial_dims=3, in_channels=1, num_classes_binary=1, dropout_prob=0.2, saliency_maps='Prognosis')
    model.to(device)
    model.load_state_dict(torch.load(path_to_save_model))
    layer='feature_extractor.features.transition3.conv'
    model = medcam.inject(model, layer=layer, backend='ggcam', output_dir=path_to_save_saliency_maps, save_maps=True, return_score=False, cudnn=True)
    map_type = "Guided-Grad-Cam"
    # Note: Guided-Backpropagation ignores parameter layer.
    print("Layer names in model:", medcam.get_layers(model))
    
    model.eval()
    # for prognosis
    predicted_labels_test = []
    all_probabilities_test = []
    labels_test_tensor = []
    # for GCS
    predicted_labels_gcs_test = []
    labels_gcs_test_tensor = []
    # for age
    predicted_labels_age_test = []
    all_probabilities_age_test = []
    labels_age_test_tensor = []
    i = 0
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels, test_labels_gcs, test_labels_age, test_patientID = test_data["image"].to(device), test_data["label"].to(device), test_data["gcs"].to(device), test_data["age"].to(device), test_data["patient"]
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
            print("Path saliency map:", os.path.join(path_to_save_saliency_maps, str(test_patientID)+"_"+map_type+"-label-"+label_name+".nii.gz"))
            i += 1
    
    break
    print("=" * 80)
    

print("=" * 80)

sys.stdout.close()