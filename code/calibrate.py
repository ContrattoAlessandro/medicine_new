import os
import sys

# Change working directory to the script's directory so relative paths resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# Set plot style for premium aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'legend.fontsize': 12,
    'figure.dpi': 300
})

def logit(p):
    """Convert probability to logit representation with clipping to avoid numerical issues."""
    p_clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p_clipped / (1.0 - p_clipped))

def sigmoid(x):
    """Compute the sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))

def compute_class_ece(y_true, y_pred, n_bins=10):
    """Compute Expected Calibration Error (ECE) for a single class."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for m in range(n_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def compute_class_log_loss(y_true, y_pred):
    """Compute binary cross-entropy (Log Loss) for a single class with clipping."""
    p_clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    loss = - (y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
    return np.mean(loss)

def get_reliability_coordinates(y_true, y_pred, n_bins=10):
    """Get bin confidence and accuracy coordinates for plotting reliability curve."""
    # Flatten the arrays to compute a global reliability diagram for multi-label tasks
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    
    for m in range(n_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        in_bin = (y_pred_flat >= bin_lower) & (y_pred_flat < bin_upper)
        if np.sum(in_bin) > 0:
            bin_centers.append(np.mean(y_pred_flat[in_bin]))
            bin_accuracies.append(np.mean(y_true_flat[in_bin]))
        else:
            bin_centers.append((bin_lower + bin_upper) / 2.0)
            bin_accuracies.append(0.0)
            
    return bin_centers, bin_accuracies

def main():
    exp_dir = '../output/exp0'
    plots_dir = '../output/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    print("==========================================================")
    print("      ECG MODEL CALIBRATION & PLATT SCALING ANALYSIS      ")
    print("==========================================================\n")

    # Paths to predictions
    y_val_path = os.path.join(exp_dir, 'data', 'y_val.npy')
    y_test_path = os.path.join(exp_dir, 'data', 'y_test.npy')
    
    model_dir = os.path.join(exp_dir, 'models', 'fastai_resnet1d18')
    y_val_pred_path = os.path.join(model_dir, 'y_val_pred.npy')
    y_test_pred_path = os.path.join(model_dir, 'y_test_pred.npy')
    
    # Check if predictions exist
    if not (os.path.exists(y_val_path) and os.path.exists(y_test_path) and 
            os.path.exists(y_val_pred_path) and os.path.exists(y_test_pred_path)):
        print(f"Error: Could not find model predictions at {model_dir}.")
        print("Please make sure you have run reproduce_results.py for exp0 first!")
        return

    # Load arrays
    y_val = np.load(y_val_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)
    y_val_pred = np.load(y_val_pred_path, allow_pickle=True)
    y_test_pred = np.load(y_test_pred_path, allow_pickle=True)
    
    n_classes = y_val.shape[1]
    print(f"Loaded validation and test sets (Number of classes: {n_classes})")
    
    # ----------------------------------------------------
    # Step 1: Compute baseline calibration metrics (Uncalibrated)
    # ----------------------------------------------------
    uncal_ece_list = []
    uncal_brier_list = []
    uncal_logloss_list = []
    
    for c in range(n_classes):
        y_t = y_test[:, c]
        y_p = y_test_pred[:, c]
        
        uncal_ece_list.append(compute_class_ece(y_t, y_p))
        uncal_brier_list.append(brier_score_loss(y_t, y_p))
        uncal_logloss_list.append(compute_class_log_loss(y_t, y_p))
        
    print(f"\n--- Baseline Test Metrics (Before Calibration) ---")
    print(f"Macro Expected Calibration Error (ECE): {np.mean(uncal_ece_list):.4f}")
    print(f"Macro Brier Score:                      {np.mean(uncal_brier_list):.4f}")
    print(f"Macro Log Loss (Binary Cross Entropy):   {np.mean(uncal_logloss_list):.4f}")
    
    # ----------------------------------------------------
    # Step 2: Fit Platt Scaling (Class-wise logistic regression) on Val predictions
    # ----------------------------------------------------
    y_test_pred_cal = np.zeros_like(y_test_pred)
    
    print("\nFitting Platt Scaling calibrator class-by-class on validation set...")
    skipped_classes = 0
    calibrators = {}
    
    for c in range(n_classes):
        y_t_val = y_val[:, c]
        y_p_val = y_val_pred[:, c]
        
        # Check if validation class label is constant (Platt scaling needs both classes)
        if len(np.unique(y_t_val)) < 2:
            y_test_pred_cal[:, c] = y_test_pred[:, c]
            skipped_classes += 1
            continue
            
        # Transform validation probabilities to logits (features for calibration regression)
        val_logits = logit(y_p_val).reshape(-1, 1)
        
        # Fit logistic regression calibrator
        lr = LogisticRegression(C=1e5, solver='lbfgs')
        lr.fit(val_logits, y_t_val)
        
        # Save calibrator for this class
        calibrators[c] = lr
        
        # Apply mapping on test set
        test_logits = logit(y_test_pred[:, c]).reshape(-1, 1)
        y_test_pred_cal[:, c] = lr.predict_proba(test_logits)[:, 1]
        
    if skipped_classes > 0:
        print(f"Warning: skipped Platt scaling for {skipped_classes} classes due to constant validation labels.")
        
    # Save the calibration dictionary
    import pickle
    calibrator_path = os.path.join(exp_dir, 'data', 'platt_calibrators.pkl')
    with open(calibrator_path, 'wb') as f:
        pickle.dump(calibrators, f)
    print(f"Saved fitted calibration models to: {calibrator_path}")
        
    # ----------------------------------------------------
    # Step 3: Compute post-calibration metrics (Calibrated)
    # ----------------------------------------------------
    cal_ece_list = []
    cal_brier_list = []
    cal_logloss_list = []
    
    for c in range(n_classes):
        y_t = y_test[:, c]
        y_p = y_test_pred_cal[:, c]
        
        cal_ece_list.append(compute_class_ece(y_t, y_p))
        cal_brier_list.append(brier_score_loss(y_t, y_p))
        cal_logloss_list.append(compute_class_log_loss(y_t, y_p))
        
    print(f"\n--- Post-Calibration Test Metrics (After Platt Scaling) ---")
    print(f"Macro Expected Calibration Error (ECE): {np.mean(cal_ece_list):.4f}  (Improvement: {np.mean(uncal_ece_list)-np.mean(cal_ece_list):+.4f})")
    print(f"Macro Brier Score:                      {np.mean(cal_brier_list):.4f}  (Improvement: {np.mean(uncal_brier_list)-np.mean(cal_brier_list):+.4f})")
    print(f"Macro Log Loss (Binary Cross Entropy):   {np.mean(cal_logloss_list):.4f}  (Improvement: {np.mean(uncal_logloss_list)-np.mean(cal_logloss_list):+.4f})")
    
    # ----------------------------------------------------
    # Step 4: Visual Analysis - Reliability Curves Plot
    # ----------------------------------------------------
    print("\nGenerating Reliability Diagram...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='#7f8c8d', alpha=0.8, linewidth=1.5, label='Perfect Calibration')
    
    # Before Calibration coordinates
    bin_centers_uncal, bin_accuracies_uncal = get_reliability_coordinates(y_test, y_test_pred)
    ax.plot(bin_centers_uncal, bin_accuracies_uncal, marker='s', color='#e74c3c', linewidth=2.0, 
            markersize=6, label=f'Uncalibrated Model (ECE = {np.mean(uncal_ece_list):.3f})')
            
    # After Calibration coordinates
    bin_centers_cal, bin_accuracies_cal = get_reliability_coordinates(y_test, y_test_pred_cal)
    ax.plot(bin_centers_cal, bin_accuracies_cal, marker='o', color='#2ecc71', linewidth=2.0, 
            markersize=6, label=f'Calibrated with Platt Scaling (ECE = {np.mean(cal_ece_list):.3f})')
            
    ax.set_title("Reliability Diagram Comparison (ResNet1d18)", pad=15, fontweight='bold', color='#2c3e50')
    ax.set_xlabel("Mean Predicted Probability (Confidence)", labelpad=10)
    ax.set_ylabel("Fraction of Positives (Empirical Probability)", labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'calibration_platt_scaling.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Step 5: Save comparative report to file
    # ----------------------------------------------------
    report_path = os.path.join(plots_dir, 'calibration_report.txt')
    with open(report_path, 'w') as f:
        f.write("==========================================================\n")
        f.write("      ECG MODEL CALIBRATION & PLATT SCALING REPORT        \n")
        f.write("==========================================================\n\n")
        f.write(f"Model evaluated: ResNet1d18\n")
        f.write(f"Task: all SCP statement prediction (71 classes)\n\n")
        f.write("Metric                  | Before Calibration | After Platt Scaling | Difference\n")
        f.write("-------------------------------------------------------------------------------\n")
        f.write(f"Macro ECE               | {np.mean(uncal_ece_list):.5f}            | {np.mean(cal_ece_list):.5f}             | {np.mean(cal_ece_list)-np.mean(uncal_ece_list):+.5f}\n")
        f.write(f"Macro Brier Score       | {np.mean(uncal_brier_list):.5f}            | {np.mean(cal_brier_list):.5f}             | {np.mean(cal_brier_list)-np.mean(uncal_brier_list):+.5f}\n")
        f.write(f"Macro Log Loss          | {np.mean(uncal_logloss_list):.5f}            | {np.mean(cal_logloss_list):.5f}             | {np.mean(cal_logloss_list)-np.mean(uncal_logloss_list):+.5f}\n")
        
    print(f"\nSuccessfully generated calibration comparison plot. Saved to: {plot_path}")
    print(f"Calibration metrics report saved to: {report_path}")
    print("\n==========================================================")

if __name__ == "__main__":
    main()
