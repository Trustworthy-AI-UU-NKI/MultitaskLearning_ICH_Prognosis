# create MONAI model from npy preprocessed files downloaded from Digital CSIC
# Miriam Cobo, nov 2023

import logging
import os
import sys
import shutil
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
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
    Transpose
)
from ModelsPytorch import TabularModel
from bootstrapping import bootstrapping
from imblearn.over_sampling import RandomOverSampler
from torchsummary import summary
import shap
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import sys
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

# save prints in a txt file
sys.stdout=open("/home/ubuntu/tenerife/data/ICH_results/tabularData_model/run_out_10folds_SameLRRepeatTraining.txt",'w')

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

# Set data directory
directory = "/home/ubuntu/tenerife/data/ICH_nii"
# read images and corresponding labels from directory
goodPrognosis_images = sorted(os.listdir(os.path.join(directory, "GOOD_PROGNOSIS")))
print(f"Good prognosis images(0): {len(goodPrognosis_images)}")
PoorPrognosis_images = sorted(os.listdir(os.path.join(directory, "POOR_PROGNOSIS")))
print(f"Poor prognosis images (1): {len(PoorPrognosis_images)}")

# read CLINICAL DATA
clinical_data = pd.read_csv('/home/ubuntu/tenerife/data/ICH_tabular_data/CLINICAL_DATA_ICH.csv', delimiter=',')
# rename PatientID by Patient
clinical_data = clinical_data.rename(columns={'PatientID': 'Patient'})
clinical_data.columns
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
        if os.path.isfile(os.path.join(directory, "GOOD_PROGNOSIS", str(patientID) + ".nii.gz")):
            images_all.append(os.path.join(directory, "GOOD_PROGNOSIS", str(patientID) + ".nii.gz"))
            labels_all.append(label)
    elif label==1:
        # check that image exists
        if os.path.isfile(os.path.join(directory, "POOR_PROGNOSIS", str(patientID) + ".nii.gz")):
            images_all.append(os.path.join(directory, "POOR_PROGNOSIS", str(patientID) + ".nii.gz"))
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
# save test labels and predictions for each fold
test_labels_df = pd.DataFrame(columns=['Fold', 'True_labels', 'Probabilities_labels_test', 'Predicted_labels_test_th0.5'])

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

    print("Training set:", len(images_train), "images,", labels_train.count(0), "good prognosis,", labels_train.count(1), "poor prognosis")
    print("Validation set:", len(images_val), "images,", labels_val.count(0), "good prognosis,", labels_val.count(1), "poor prognosis")
    print("Test set:", len(images_test), "images,", labels_test.count(0), "good prognosis,", labels_test.count(1), "poor prognosis")

    # Count the occurrences of each class
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
        # Resized(keys="image", spatial_size=[256, 256, -1]),
        SpatialPadd(keys="image", spatial_size=[-1, -1, 52], mode=('constant'), method= ("symmetric")),
        Transposed(keys="image", indices=[0, 3, 1, 2])
        ])
    # SpatialPad(spatial_size=[1, 32, 571, 571], mode=('constant'), method= ("symmetric"))
    val_transforms = Compose(
        [LoadImaged(keys="image", image_only=True, ensure_channel_first=True, reader="ITKReader"), 
        Rotate90d(keys="image", k=3), 
        Flipd(keys="image", spatial_axis=1),
        NormalizeIntensityd(keys="image", subtrahend=15, divisor=85),
        ThresholdIntensityd(keys="image", threshold=0, above=True, cval=0.0),
        ThresholdIntensityd(keys="image", threshold=1, above=False, cval=0.0),
        # Resized(keys="image", spatial_size=[256, 256, -1]),
        SpatialPadd(keys="image", spatial_size=[-1, -1, 52], mode=('constant'), method= ("symmetric")),
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
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=pin_memory)

    # you can use Dataset or CacheDataset, according to 
    # this https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb
    # the later is faster

    # create a validation data loader
    # val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

    # input to 3d cnn https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#conv3d
    # and https://stackoverflow.com/questions/66199022/what-should-be-the-input-shape-for-3d-cnn-on-a-sequence-of-images

    df = clinical_data_filtered
    # rename patient column
    df = df.rename(columns={'Patient': 'PatientID'})
    print(df.shape)

    # loop over images_train, images_val and images_test to get the corresponding clinical data
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    X_val = pd.DataFrame()
    for i in images_res:
        patientID = int(i.split('/')[-1].split('.')[0])
        # get all columns in df for this patient
        X_train = pd.concat([X_train, df[df['PatientID']==patientID]])
    for i in images_test:
        patientID = int(i.split('/')[-1].split('.')[0])
        # get all columns in df for this patient
        X_test = pd.concat([X_test, df[df['PatientID']==patientID]])
    for i in images_val:
        patientID = int(i.split('/')[-1].split('.')[0])
        # get all columns in df for this patient
        X_val = pd.concat([X_val, df[df['PatientID']==patientID]])

    # remove PatientID from X_train, X_test and X_val
    X_train = X_train.drop(columns=['PatientID', 'Label (poor_prognosis)'])
    X_test = X_test.drop(columns=['PatientID', 'Label (poor_prognosis)'])
    X_val = X_val.drop(columns=['PatientID', 'Label (poor_prognosis)'])

    # normalize the data
    categorical_var=['Gender', 'Smoker', 'Alcohol', 'HT',
        'DM', 'Dyslipidemia', 'MH_ICH', 'MH_CVD', 'MH_neuro', 'MH_dementia',
        'MH_cancer', 'MH_hematho', 'MH_others', 'Anticoagulant',
        'Antiaggregant', 'Antihypertensive_drugs', 'Calcium_antag', 'Alpha_blockers', 'PE_neuro', 'Cause_head_trauma']
    ordinal_categorical_var=['GCS']
    numerical_var=['Age', 'Systolic_AP',
        'Dyastolic_AP', 'OxSat', 'Tº', 'HR', 'RF', 'Glucose',
        'Creatinine', 'Urea', 'Sodium', 'Potassium', 'WBC', 'Hgb', 'Platelets',
        'MCV', 'RCDW', 'CMHC', 'MPV', 'INR', 'Fibrinogen']
    print("Number of categorical variables:", len(categorical_var))
    print("Number of numerical variables:", len(numerical_var))
    print("Number of ordinal categorical variables:", len(ordinal_categorical_var))
    # check if categorical_var are one hot encoded
    # for i in categorical_var:
        # print(i, X_train[i].unique())
    # check if numerical_var are normalized between 0 and 1
    # for i in numerical_var:
    #        print(i, X_train[i].min(), X_train[i].max())
    # normalize numerical_var in X_train, X_test and X_val
    scaler = MinMaxScaler()
    X_train[numerical_var] = scaler.fit_transform(X_train[numerical_var])
    X_test[numerical_var] = scaler.transform(X_test[numerical_var])
    X_val[numerical_var] = scaler.transform(X_val[numerical_var])

    gcs_categories = [[i for i in range(3, 16)]]

    # Create the OrdinalEncoder, specifying the order
    encoder = OrdinalEncoder(categories=gcs_categories)

    # Fit and transform the data
    X_train[ordinal_categorical_var] = encoder.fit_transform(X_train[ordinal_categorical_var])
    X_test[ordinal_categorical_var] = encoder.transform(X_test[ordinal_categorical_var])
    X_val[ordinal_categorical_var] = encoder.transform(X_val[ordinal_categorical_var])

    # print shape X_train, X_test and X_val
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("X_val shape:", X_val.shape)

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values.reshape(-1, 1, X_train.shape[1]))  # Add a channel dimension
    labels_train_tensor = torch.LongTensor(labels_res)

    X_val_tensor = torch.FloatTensor(X_val.values.reshape(-1, 1, X_val.shape[1]))
    labels_val_tensor = torch.LongTensor(labels_val)

    X_test_tensor = torch.FloatTensor(X_test.values.reshape(-1, 1, X_test.shape[1]))
    labels_test_tensor = torch.LongTensor(labels_test)

    # Create DataLoader for training, validation, and testing sets
    train_dataset = TensorDataset(X_train_tensor, labels_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, labels_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, labels_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    threshold = 0.5

    # Initialize the model, loss function, and optimizer
    # model = CNNModel(input_size=X_train.shape[1], num_classes=2)
    # model = TabularModel(input_size=X_train.shape[1], num_classes=2, initializer_seed = initializer_seed)
    model = TabularModel(input_size=X_train.shape[1], out_channels=1, threshold=0.5)
    model.apply(reset_weights)
    model.to(device)
    print(summary(model, input_size=(X_train.shape[1],)))
    # criterion = nn.CrossEntropyLoss()

    model = TabularModel(input_size=X_train.shape[1], out_channels=1, threshold=0.5)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001, amsgrad=True) # 0.005 -> updated for SameLR to 0.001
    # weight_decay 0.0001 -> updated for SameLR to 0.001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.98, verbose=True)

    # Training loop with validation
    max_epochs = 100

    # early stopping
    patience = 20
    epochs_no_improve = 0
    val_interval = 1
    early_stop = False
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    epoch_auc_values = []
    epoch_accuracy_values = []
    auc_values = []
    accuracy_values = []

    path_to_save_model=os.path.join("/home/ubuntu/tenerife/data/ICH_models",
                                                "tabular10folds_SameLRRepeatTraining_modelPrognosisICH_fold"+str(fold)+".pth")

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

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            train_probabilities = nn.Sigmoid()(outputs)
            train_predictions = (train_probabilities >= threshold).float()
            probabilities_train.extend(train_probabilities.detach().cpu().numpy())
            labels_train.extend(labels.cpu().numpy())
            predictions_train.extend(train_predictions.detach().cpu().numpy())
        
        train_auc=roc_auc_score(labels_train, probabilities_train)
        train_accuracy = balanced_accuracy_score(labels_train, predictions_train)
        epoch_loss_values.append(epoch_loss)
        epoch_auc_values.append(train_auc)
        epoch_accuracy_values.append(train_accuracy)
        print(f"Train epoch {epoch + 1} average loss: {epoch_loss:.3f}, train_auc: {train_auc:.3f}, train_accuracy: {train_accuracy:.3f}")


        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        predicted_labels_val = []
        all_predictions = []  # To store predicted values
        all_probabilities = []  # To store corresponding probabilities
        all_labels = []  # To store ground truth
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                val_outputs = model(inputs_val)
                loss_val = criterion(val_outputs, labels_val.float())
                val_loss += loss_val.item()

                probabilities = nn.Sigmoid()(val_outputs)
                predicted = (probabilities >= threshold).float()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels_val.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                total_val += labels_val.size(0)
                correct_val += (predicted == labels_val).sum().item()

                predicted_labels_val.extend(predicted.cpu().numpy())

        val_accuracy = correct_val / total_val
        average_val_loss = val_loss / len(val_loader)
        all_probabilities = np.array(all_probabilities)
        # val_auc = roc_auc_score(labels_val_tensor.cpu().numpy(), predicted_labels_val)
        val_precision = precision_score(labels_val_tensor.cpu().numpy(), predicted_labels_val)
        val_recall = recall_score(labels_val_tensor.cpu().numpy(), predicted_labels_val)
        val_f1 = f1_score(labels_val_tensor.cpu().numpy(), predicted_labels_val)

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
            torch.save(model.state_dict(), os.path.join("/home/ubuntu/tenerife/data/ICH_models",
                                                        "best_tabular10folds_SameLRRepeatTraining_model_prognosis.pth"))
            # print("saved new best metric model")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == patience:
                print('Early stopping!')
                early_stop = True
                break
            
        print(f'Epoch [{epoch+1}/{max_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}')
        print(f'Validation ROC AUC: {roc_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
        print(
                    f"current epoch: {epoch + 1} current AUC: {roc_auc:.3f}"
                    f" current accuracy: {accuracy:.3f}"
                    f" best accuracy: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    # Evaluation on the test set
    model.eval()
    predicted_labels_test = []
    all_probabilities_test = []
    number_samples = 0
    with torch.no_grad():
        for inputs_test, labels_test in test_loader:
            number_samples += 1
            outputs_test = model(inputs_test)
            probabilities_test = nn.Sigmoid()(outputs_test)
            predicted_test = (probabilities_test >= threshold).float()
            # all_probabilities_test.extend(probabilities_test.cpu().numpy())
            # all_probabilities_test = np.array(all_probabilities_test)
        all_probabilities_test = np.array(probabilities_test)
        labels_test_tensor = np.array(labels_test_tensor).astype(int)
        predicted_labels_test=np.array(predicted_test).astype(int)
        fold_array = np.full((len(labels_test_tensor)), fold)
        combined=np.column_stack((fold_array, labels_test_tensor, all_probabilities_test, predicted_labels_test))
        test_labels_df = pd.concat([test_labels_df, pd.DataFrame(combined, columns=test_labels_df.columns)], ignore_index=True)
        test_balanced_accuracy = balanced_accuracy_score(labels_test_tensor, predicted_test)
        test_accuracy = accuracy_score(labels_test_tensor, predicted_test)
        test_auc = roc_auc_score(labels_test_tensor, all_probabilities_test)
        test_precision = precision_score(labels_test_tensor, predicted_test)
        test_recall = recall_score(labels_test_tensor, predicted_test)
        test_f1 = f1_score(labels_test_tensor, predicted_test)
        tn, fp, fn, tp = confusion_matrix(labels_test_tensor, predicted_test, labels=[0, 1]).ravel()
        test_NPV=tn/(tn+fn)
        test_specificity=tn/(tn+fp)

    fold_metrics_df = pd.concat([fold_metrics_df, pd.DataFrame([{'Fold': fold, 'AUC': test_auc, 'Balanced_accuracy': test_balanced_accuracy, 'Accuracy': test_accuracy, 'Specificity': test_specificity, 'NPV': test_NPV, 'Precision': test_precision, 'Recall': test_recall, 'F1-score': test_f1}])], ignore_index=True)
    print("Probabilities test:", all_probabilities_test)

    print(f'Test Accuracy: {test_accuracy:.2%}')
    print(f'Test ROC AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')

    # Save predicted labels for test set
    predicted_labels_df = pd.DataFrame({'True Labels': labels_test_tensor, 'Predicted Labels': predicted_test})
    # predicted_labels_df.to_csv('predicted_labels.csv', index=False)
    try:
        print(bootstrapping(y_true=labels_test_tensor, y_pred=probabilities_test, y_pred_threshold=predicted_test, 
                    path_to_save_metrics='/home/ubuntu/tenerife/data/ICH_results/tabularData_model', 
                    metrics = 'AUC', confidence = 0.95, n_bootstraps = 1000))

        print(bootstrapping(y_true=labels_test_tensor, y_pred=probabilities_test, y_pred_threshold=predicted_test, 
                    path_to_save_metrics='/home/ubuntu/tenerife/data/ICH_results/tabularData_model', 
                    metrics = 'METRICS', confidence = 0.95, n_bootstraps = 1000))
    except Exception as e:
        print("Error in bootstrapping:", e)

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
    plt.title("Accuracy")
    x = [i + 1 for i in range(len(epoch_accuracy_values))]
    y = epoch_accuracy_values
    plt.plot(x, y)
    x = [val_interval * (i + 1) for i in range(len(accuracy_values))]
    y = accuracy_values
    plt.plot(x, y)
    plt.xlabel("epoch")
    plt.legend(['train', 'val'], loc='best')
    plt.savefig(os.path.join("/home/ubuntu/tenerife/data/ICH_results/tabularData_model", "TrainAndVal10folds_SameLRRepeatTraining_fold"+str(fold)+".png"))
    plt.close()

    # from torch.autograd import Variable

    # # Define function to wrap model to transform data to tensor
    # f = lambda x: model( Variable( torch.from_numpy(x) ) ).detach().numpy()
    # # Convert my pandas dataframe to numpy
    # data_train = X_train.to_numpy(dtype=np.float32)
    # data_test = X_test.to_numpy(dtype=np.float32)
    # # The explainer doesn't like tensors, hence the f function
    # explainer = shap.KernelExplainer(f, data_train)

    # # Get the shap values from my test data
    # shap_values = explainer.shap_values(data_test)

    # # Enable the plots in jupyter
    # shap.initjs()

    # feature_names = X_test.columns
    # # Plots
    # #shap.force_plot(explainer.expected_value, shap_values[0], feature_names)
    # #shap.dependence_plot("b1_price_avg", shap_values[0], data, feature_names)
    # # shap.summary_plot(shap_values, data_test, feature_names)

    # fig = shap.summary_plot(shap_values, data_test, feature_names, plot_type="bar",plot_size=(10, 10), show=False)
    # plt.savefig('/home/ubuntu/tenerife/data/ICH_results/tabularData_model/shapSameLRRepeatTraining_summary_plot_fold'+str(fold)+'.png')
    # plt.close()
    print("=" * 80)

# calculate mean and std of metrics for all folds
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

# save fold_metrics_df
fold_metrics_df.to_csv(os.path.join("/home/ubuntu/tenerife/data/ICH_results/tabularData_model", "fold_metrics10folds_SameLRRepeatTraining_df.csv"), index=False)
test_labels_df.to_csv(os.path.join("/home/ubuntu/tenerife/data/ICH_results/tabularData_model", "test_labels10folds_SameLRRepeatTraining_df.csv"), index=False)

sys.stdout.close()