# evaluate the metrics of a model for different thersholds and print AUC

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

class EvaluateThresholds:
    def __init__(self, all_probabilities_test, labels_test, path_to_save_auc_plot, fold):
        self.probabilities = all_probabilities_test
        self.labels = labels_test
        self.path_to_save_auc_plot = path_to_save_auc_plot
        self.fold=fold

    def plot_roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.labels, self.probabilities)
        auc_score = roc_auc_score(self.labels, self.probabilities)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.path_to_save_auc_plot)

        return auc_score

    def evaluate_metrics_recall(self):
        thresholds = np.arange(0.1, 0.95, 0.05)
        auc_score = roc_auc_score(self.labels, self.probabilities)
        best_threshold = None
        best_metrics = {
            'Fold': self.fold, 'AUC': auc_score, 'Threshold': 0, 'Balanced_accuracy': 0, 'Accuracy': 0, 'Specificity': 0, 
            'NPV': 0, 'Precision': 0, 'Recall': 0, 'F1-score': 0
        }

        for threshold in thresholds:
            predicted_labels = self.probabilities >= threshold
            tn, fp, fn, tp = confusion_matrix(self.labels, predicted_labels).ravel()

            accuracy = accuracy_score(self.labels, predicted_labels)
            balanced_accuracy = balanced_accuracy_score(self.labels, predicted_labels)
            specificity = tn / (tn + fp)
            npv = tn / (tn + fn) if (tn + fn) != 0 else 0
            precision = precision_score(self.labels, predicted_labels)
            recall = recall_score(self.labels, predicted_labels)
            f1 = f1_score(self.labels, predicted_labels)
            # in our case the best is to maximize the sensibility - recall?
            if recall > best_metrics['Recall']:
                best_threshold = threshold
                best_metrics = {
                    'Fold': self.fold, 'AUC': auc_score, 'Threshold':best_threshold, 'Balanced_accuracy': balanced_accuracy, 'Accuracy': accuracy, 'Specificity': specificity,
                    'NPV': npv, 'Precision': precision, 'Recall': recall, 'F1-score': f1
                }

        return best_threshold, best_metrics
    
    def evaluate_metrics_f1(self):
        thresholds = np.arange(0.1, 0.95, 0.05)
        auc_score = roc_auc_score(self.labels, self.probabilities)
        best_threshold = None
        best_metrics = {
            'Fold': self.fold, 'AUC': auc_score, 'Threshold': 0, 'Balanced_accuracy': 0, 'Accuracy': 0, 'Specificity': 0, 
            'NPV': 0, 'Precision': 0, 'Recall': 0, 'F1-score': 0
        }

        for threshold in thresholds:
            predicted_labels = self.probabilities >= threshold
            tn, fp, fn, tp = confusion_matrix(self.labels, predicted_labels).ravel()

            accuracy = accuracy_score(self.labels, predicted_labels)
            balanced_accuracy = balanced_accuracy_score(self.labels, predicted_labels)
            specificity = tn / (tn + fp)
            npv = tn / (tn + fn) if (tn + fn) != 0 else 0
            precision = precision_score(self.labels, predicted_labels)
            recall = recall_score(self.labels, predicted_labels)
            f1 = f1_score(self.labels, predicted_labels)
            # in our case the best is to maximize the sensibility - recall?
            if f1 > best_metrics['F1-score']:
                best_threshold = threshold
                best_metrics = {
                    'Fold': self.fold, 'AUC': auc_score, 'Threshold': best_threshold, 'Balanced_accuracy': balanced_accuracy, 'Accuracy': accuracy, 'Specificity': specificity,
                    'NPV': npv, 'Precision': precision, 'Recall': recall, 'F1-score': f1
                }

        return best_threshold, best_metrics