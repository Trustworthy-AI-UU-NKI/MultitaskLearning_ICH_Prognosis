import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) != 0 else 0

def calculate_npv(y_true, y_pred):
    tn, _, fn, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) != 0 else 0

def bootstrap_metric_ci(y_true, y_pred_threshold, n_bootstrap_samples=1000, sample_size=100, ci=95):
    metrics_functions = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score,
        # 'specificity' and 'NPV' will be calculated manually
    }
    metrics_results = {metric: np.zeros(n_bootstrap_samples) for metric in metrics_functions.keys()}
    metrics_results['specificity'] = np.zeros(n_bootstrap_samples)
    metrics_results['npv'] = np.zeros(n_bootstrap_samples)
    
    for i in range(n_bootstrap_samples):
        sample_indices = np.random.choice(np.arange(len(y_true)), size=sample_size, replace=True)
        y_true_sample = np.array(y_true)[sample_indices]
        y_pred_sample = np.array(y_pred_threshold)[sample_indices]
        
        for metric, func in metrics_functions.items():
            metrics_results[metric][i] = func(y_true_sample, y_pred_sample)
            
        metrics_results['specificity'][i] = calculate_specificity(y_true_sample, y_pred_sample)
        metrics_results['npv'][i] = calculate_npv(y_true_sample, y_pred_sample)
    
    for metric, values in metrics_results.items():
        lower_bound = np.percentile(values, (100 - ci) / 2)
        upper_bound = np.percentile(values, 100 - (100 - ci) / 2)
        mean_value = np.mean(values)
        print(f"{metric}: {mean_value:.3f}, 95% CI: [{lower_bound:.3f}, {upper_bound:.3f}]")