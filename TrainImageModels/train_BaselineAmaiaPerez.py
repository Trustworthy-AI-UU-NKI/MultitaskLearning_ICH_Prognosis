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

from bootstrapping import bootstrapping
from bootstrapDEF import bootstrap_metric_ci
from torchsummary import summary

from BaselineAmaiaPytorch import AmaiaModel

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
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
name_file_model = config["name_file_model"]

# path_to_save_model_dir = "/home/ubuntu/tenerife/data/ZZ_ICH_PrognosisMICCAI/Models/BaselineAmaia"
# path_to_save_results = "/home/ubuntu/tenerife/data/ZZ_ICH_PrognosisMICCAI/Results/BaselineAmaia"
# name_file_model = "BaselineAmaia_Prognosis301_40"
# name_file = "RepeatCI_Test_BaselineAmaia_Prognosis301_40"

# save prints in a txt file
sys.stdout=open(os.path.join(path_to_save_results, "run_out_"+name_file+"_10fold.txt"),'w')

pin_memory = torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

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
test_labels_df = pd.DataFrame(columns=['Fold', 'True_labels', 'Probabilities_label1_test', 'Predicted_labels_test_th0.5'])

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

    train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images_res, labels_res)]
    val_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images_val, labels_val)]
    test_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images_test, labels_test)]

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
                                                name_file_model+str(fold)+".pth") ###
    threshold = 0.5
    if not os.path.isfile(path_to_save_model):

        model = AmaiaModel(depth=40, width=301, height=301, in_channels=1, out_channels=1, initializer_seed=seed)

        total_params = sum(
            param.numel() for param in model.parameters()
        )
        print("Total params model", total_params)

        # model = model.cuda()
        ###
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs")
        #     model= torch.nn.DataParallel(model)

        model.to(device)

        # loss_function = torch.nn.CrossEntropyLoss() # this is for 2 out_channels output
        loss_function = torch.nn.BCEWithLogitsLoss().to(device) # also works with this data

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=True)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.98, verbose=False)
        #### understand well all the training and revise
        #### create custom model to append at the end tabular data and image features
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        epoch_auc_values = []
        epoch_accuracy_values = []
        auc_values = []
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

            predictions_train=[]
            labels_train=[]
            probabilities_train=[]
            optimizer.zero_grad()
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                # optimizer.zero_grad()
                # print(inputs.shape)
                outputs = model(inputs)
                ### sequeeze to get the right output shape
                outputs = outputs.squeeze(1)
                loss = loss_function(outputs, labels.float())
                loss = loss / accumulation_steps  # Normalize loss for accumulation
                loss.backward()  # Accumulate gradients

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()  # Update model parameters
                    scheduler.step(loss)  # Scheduler step (if scheduler requires loss)
                    optimizer.zero_grad()  # Reset gradients

                epoch_loss += loss.item() * accumulation_steps  # Correct loss scaling
                epoch_len = len(train_ds) // train_loader.batch_size

                train_probabilities = nn.Sigmoid()(outputs)
                train_predictions = (train_probabilities >= threshold).float()
                probabilities_train.extend(train_probabilities.detach().cpu().numpy())
                labels_train.extend(labels.cpu().numpy())
                predictions_train.extend(train_predictions.detach().cpu().numpy())
                # print(labels.cpu().numpy(), train_predictions.detach().cpu().numpy())
                
            train_auc=roc_auc_score(labels_train, probabilities_train)
            train_accuracy = balanced_accuracy_score(labels_train, predictions_train)
            unique_train, counts_train = np.unique(predictions_train, return_counts=True)
            # print("Train predictions count:", dict(zip(unique_train, counts_train)))
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            epoch_auc_values.append(train_auc)
            epoch_accuracy_values.append(train_accuracy)
            print(f"Train epoch {epoch + 1} average loss: {epoch_loss:.3f}, train_auc: {train_auc:.3f}, train_accuracy: {train_accuracy:.3f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()

                num_correct = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    with torch.no_grad():
                        val_outputs = model(val_images)
                        val_outputs = val_outputs.squeeze(1) ### sequeeze to get the right output shape
                        # _, predicted = val_outputs.max(1)
                        probabilities = nn.Sigmoid()(val_outputs)
                        predicted = (probabilities >= threshold).float()
                        # value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(val_labels.cpu().numpy())
                        all_probabilities.extend(probabilities.cpu().numpy())

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

                # Print metrics for each epoch
                print(f"Epoch {epoch + 1}/{max_epochs} - AUC: {roc_auc:.3f}, Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")

                print(
                        f"current epoch: {epoch + 1} current AUC: {roc_auc:.3f}"
                        f" current accuracy: {accuracy:.3f}"
                        f" best accuracy: {best_metric:.3f}"
                        f" at epoch: {best_metric_epoch}"
                    )
                # writer.add_scalar("auc", roc_auc, epoch + 1)

        print(f"Training completed, best_metric: {best_metric:.3f} at epoch: {best_metric_epoch}")
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
        x = [val_interval * (i + 1) for i in range(len(auc_values))]
        y = auc_values
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.legend(['train', 'val'], loc='best')
        plt.subplot(1, 3, 3)
        plt.title("Balanced ccuracy")
        x = [i + 1 for i in range(len(epoch_accuracy_values))]
        y = epoch_accuracy_values
        plt.plot(x, y)
        x = [val_interval * (i + 1) for i in range(len(accuracy_values))]
        y = accuracy_values
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.legend(['train', 'val'], loc='best')
        plt.savefig(os.path.join(path_to_save_results, "TrainAndVal_"+name_file+"_fold"+str(fold)+".png"))
        plt.close()

    print("Inference in test")

    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=8)

    model = AmaiaModel(depth=40, width=301, height=301, in_channels=1, out_channels=1, initializer_seed=seed)
    model.to(device)
    model.load_state_dict(torch.load(path_to_save_model))
    model.eval()
    predicted_labels_test = []
    all_probabilities_test = []
    labels_test_tensor = []
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data["image"].to(device), test_data["label"].to(device)
            outputs_test = model(test_images)
            outputs_test = outputs_test.squeeze(1) ### sequeeze to get the right output shape
            probabilities_test = nn.Sigmoid()(outputs_test)
            predicted_test = (probabilities_test >= threshold).float()
            all_probabilities_test.extend(probabilities_test.cpu().numpy())
            predicted_labels_test.extend(predicted_test.cpu().numpy())
            labels_test_tensor.extend(test_labels.cpu().numpy())

    all_probabilities_test = np.array(all_probabilities_test)
    labels_test_tensor = np.array(labels_test_tensor).astype(int)
    predicted_labels_test=np.array(predicted_labels_test).astype(int)
    # create a fold_array that repeates the fold number as many times as test samples
    fold_array = np.full((len(labels_test_tensor)), fold)
    combined=np.column_stack((fold_array, labels_test_tensor, all_probabilities_test, predicted_labels_test))
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

    fold_metrics_df = pd.concat([fold_metrics_df, pd.DataFrame([{'Fold': fold, 'AUC': test_auc, 'Balanced_accuracy': test_balanced_accuracy, 'Accuracy': test_accuracy, 'Specificity': test_specificity, 'NPV': test_NPV, 'Precision': test_precision, 'Recall': test_recall, 'F1-score': test_f1}])], ignore_index=True)

    print("Probabilities test:", all_probabilities_test)
    unique, counts = np.unique(predicted_labels_test, return_counts=True)
    print("Predictions count test:", dict(zip(unique, counts)))
    print(f'Test Accuracy: {test_accuracy:.2%}')
    print(f'Test ROC AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

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