import os
import ast
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from models.timeseries_utils import butter_filter, apply_butter_filter

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

# Color palette (harmonious HSL-derived colors)
COLORS = {
    'NORM': '#2ecc71',      # Emerald Green
    'MI': '#e74c3c',        # Alizarin Red
    'STTC': '#f1c40f',      # Sun Flower Yellow
    'CD': '#3498db',        # Peter River Blue
    'HYP': '#9b59b6',       # Amethyst Purple
    'resnet': '#2c3e50',    # Midnight Blue
    'wavelet': '#1abc9c',   # Turquoise
    'patchtst': '#e67e22',  # Orange
    'rawstats': '#7f8c8d'   # Asbestos Grey
}

def main():
    data_dir = '../data/ptbxl/'
    output_dir = '../output/'
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading PTB-XL database...")
    db = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    db.scp_codes = db.scp_codes.apply(lambda x: ast.literal_eval(x))

    statements = pd.read_csv(os.path.join(data_dir, 'scp_statements.csv'), index_col=0)
    diag_statements = statements[statements.diagnostic == 1.0]

    # Map labels
    def get_superclasses(scp_dict):
        classes = []
        for k in scp_dict.keys():
            if k in diag_statements.index:
                sc = diag_statements.loc[k].diagnostic_class
                if pd.notna(sc):
                    classes.append(sc)
        return list(set(classes))

    db['superclasses'] = db.scp_codes.apply(get_superclasses)
    db['all_classes'] = db.scp_codes.apply(lambda x: list(set(x.keys())))

    # =========================================================================
    # PLOT 1: Class Imbalance (Slide 6)
    # =========================================================================
    print("Generating Plot 1: Class Imbalance...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Superclasses
    super_series = pd.Series(np.concatenate(db['superclasses'].values)).value_counts()
    super_colors = [COLORS.get(idx, '#95a5a6') for idx in super_series.index]
    axes[0].bar(super_series.index, super_series.values, color=super_colors, edgecolor='none', alpha=0.9)
    axes[0].set_title("Distribution of Diagnostic Superclasses", pad=15)
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Superclass")
    for i, v in enumerate(super_series.values):
        axes[0].text(i, v + 200, f"{v}", ha='center', fontweight='bold')

    # All classes (Top 25)
    all_series = pd.Series(np.concatenate(db['all_classes'].values)).value_counts().head(25)
    axes[1].bar(all_series.index, all_series.values, color='#34495e', edgecolor='none', alpha=0.85)
    axes[1].set_title("Top 25 Most Frequent SCP Codes (out of 71)", pad=15)
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("SCP Code")
    axes[1].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_imbalance.png'), dpi=300)
    plt.close()

    # =========================================================================
    # PLOT 2: Raw vs Filtered ECG Signal (Slide 8)
    # =========================================================================
    print("Generating Plot 2: Raw vs Filtered ECG...")
    # Load raw signals (100Hz)
    raw_signals = np.load(os.path.join(data_dir, 'raw100.npy'), allow_pickle=True)
    
    # Pick a signal with some noise / baseline wander (e.g. index 2 or 10)
    sig_idx = 10
    raw_signal = raw_signals[sig_idx]
    
    # Apply filter
    fs = 100
    filter_sos = butter_filter(lowcut=0.5, highcut=40.0, fs=fs, order=5, btype='band')
    filtered_signal = apply_butter_filter(raw_signal, filter_sos)

    # Lead II is index 1
    lead_name = "Lead II"
    raw_lead = raw_signal[:, 1]
    filtered_lead = filtered_signal[:, 1]
    time = np.arange(len(raw_lead)) / fs

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    # Full 10 seconds raw
    axes[0, 0].plot(time, raw_lead, color='#7f8c8d', linewidth=1)
    axes[0, 0].set_title(f"Raw Signal (Full 10s) - {lead_name}", pad=10)
    axes[0, 0].set_ylabel("Amplitude (mV)")
    axes[0, 0].set_xlabel("Time (s)")

    # Full 10 seconds filtered
    axes[0, 1].plot(time, filtered_lead, color='#2c3e50', linewidth=1)
    axes[0, 1].set_title(f"Filtered Signal (Full 10s) - {lead_name}", pad=10)
    axes[0, 1].set_ylabel("Amplitude (mV)")
    axes[0, 1].set_xlabel("Time (s)")

    # Zoomed 3 seconds raw (from t=3s to t=6s)
    zoom_mask = (time >= 3.0) & (time <= 6.0)
    axes[1, 0].plot(time[zoom_mask], raw_lead[zoom_mask], color='#e74c3c', linewidth=1.5)
    axes[1, 0].set_title("Zoomed Raw Signal (3s)", pad=10)
    axes[1, 0].set_ylabel("Amplitude (mV)")
    axes[1, 0].set_xlabel("Time (s)")

    # Zoomed 3 seconds filtered
    axes[1, 1].plot(time[zoom_mask], filtered_lead[zoom_mask], color='#27ae60', linewidth=1.5)
    axes[1, 1].set_title("Zoomed Filtered Signal (3s)", pad=10)
    axes[1, 1].set_ylabel("Amplitude (mV)")
    axes[1, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'raw_vs_filtered.png'), dpi=300)
    plt.close()

    # =========================================================================
    # PLOT 2B: Wavelet Scalogram Before and After Filtering (Slide 8)
    # =========================================================================
    print("Generating Plot 2B: Wavelet Scalogram Before and After Filtering...")
    widths = np.arange(1, 64)
    # Compute Continuous Wavelet Transform (CWT) using Mexican Hat wavelet
    cwt_raw, _ = pywt.cwt(raw_lead[zoom_mask], widths, 'mexh')
    cwt_filtered, _ = pywt.cwt(filtered_lead[zoom_mask], widths, 'mexh')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex='col')

    # Top-Left: Raw 1D signal
    axes[0, 0].plot(time[zoom_mask], raw_lead[zoom_mask], color='#e74c3c', linewidth=1.5)
    axes[0, 0].set_title("Raw ECG Signal (Zoom 3s)", pad=10)
    axes[0, 0].set_ylabel("Amplitude (mV)")

    # Bottom-Left: Raw 2D Scalogram
    im_raw = axes[1, 0].imshow(np.abs(cwt_raw), extent=[time[zoom_mask][0], time[zoom_mask][-1], 1, 64],
                               cmap='viridis', aspect='auto', interpolation='bilinear', origin='lower')
    axes[1, 0].set_title("CWT Scalogram (Raw)", pad=10)
    axes[1, 0].set_ylabel("Scale (Width)")
    axes[1, 0].set_xlabel("Time (s)")
    fig.colorbar(im_raw, ax=axes[1, 0], label="Magnitude")

    # Top-Right: Filtered 1D signal
    axes[0, 1].plot(time[zoom_mask], filtered_lead[zoom_mask], color='#27ae60', linewidth=1.5)
    axes[0, 1].set_title("Filtered ECG Signal (Zoom 3s)", pad=10)
    axes[0, 1].set_ylabel("Amplitude (mV)")

    # Bottom-Right: Filtered 2D Scalogram
    im_filtered = axes[1, 1].imshow(np.abs(cwt_filtered), extent=[time[zoom_mask][0], time[zoom_mask][-1], 1, 64],
                                    cmap='viridis', aspect='auto', interpolation='bilinear', origin='lower')
    axes[1, 1].set_title("CWT Scalogram (Filtered)", pad=10)
    axes[1, 1].set_ylabel("Scale (Width)")
    axes[1, 1].set_xlabel("Time (s)")
    fig.colorbar(im_filtered, ax=axes[1, 1], label="Magnitude")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'wavelet_before_after.png'), dpi=300)
    plt.close()

    # =========================================================================
    # PLOT 3: ROC & PR Curves (Slide 9)
    # =========================================================================
    print("Generating Plot 3: ROC and Precision-Recall Curves...")
    
    # Load test labels
    exp_path = os.path.join(output_dir, 'exp0')
    y_test = np.load(os.path.join(exp_path, 'data', 'y_test.npy'), allow_pickle=True)
    n_classes = y_test.shape[1]

    # Models to compare
    models_info = {
        'resnet1d18': ('fastai_resnet1d18', 'ResNet1d18', COLORS['resnet']),
        'wavelet': ('Wavelet+NN', 'Wavelet+NN', COLORS['wavelet']),
        'patchtst': ('PatchTST_standard', 'PatchTST_standard', COLORS['patchtst']),
        'rawstats': ('RawStats+NN', 'RawStats+NN', COLORS['rawstats'])
    }

    preds = {}
    for key, (folder, name, color) in models_info.items():
        pred_path = os.path.join(exp_path, 'models', folder, 'y_test_pred.npy')
        if os.path.exists(pred_path):
            preds[key] = np.load(pred_path, allow_pickle=True)
        else:
            print(f"Warning: predictions for {name} not found at {pred_path}.")

    if len(preds) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for key, y_pred in preds.items():
            _, _, color = models_info[key]
            lbl = models_info[key][1]

            # 1. Macro ROC
            mean_fpr = np.linspace(0, 1, 1000)
            tprs = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
            mean_tpr = np.mean(tprs, axis=0)
            mean_auc = auc(mean_fpr, mean_tpr)

            axes[0].plot(mean_fpr, mean_tpr, color=color, linewidth=2,
                         label=f'{lbl} (macro-AUC = {mean_auc:.3f})')

            # 2. Macro PR
            mean_recall = np.linspace(0, 1, 1000)
            precisions = []
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
                precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
            mean_precision = np.mean(precisions, axis=0)
            mean_ap = average_precision_score(y_test, y_pred, average='macro')

            axes[1].plot(mean_recall, mean_precision, color=color, linewidth=2,
                         label=f'{lbl} (macro-AUPR = {mean_ap:.3f})')

        axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.5)
        axes[0].set_title("Macro-averaged ROC Curve Comparison", pad=15)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend(loc='lower right')

        axes[1].set_title("Macro-averaged Precision-Recall Curve Comparison", pad=15)
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(loc='lower left')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'roc_pr_comparison.png'), dpi=300)
        plt.close()

        # =========================================================================
        # PLOT 3B: Best Model Superclass ROC & PR Curves
        # =========================================================================
        best_key = 'resnet1d18'
        if best_key in preds:
            print("Generating Plot 3B: Best Model Superclass ROC/PR...")
            y_pred_best = preds[best_key]

            # Map to 5 superclasses
            superclasses = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
            y_test_super = np.zeros((len(y_test), len(superclasses)))
            y_pred_super = np.zeros((len(y_pred_best), len(superclasses)))

            with open(os.path.join(exp_path, 'data', 'mlb.pkl'), 'rb') as f:
                mlb = pickle.load(f)
            mlb_classes = list(mlb.classes_)

            for j, sc in enumerate(superclasses):
                subclasses = diag_statements[diag_statements.diagnostic_class == sc].index.values
                indices = [mlb_classes.index(c) for c in subclasses if c in mlb_classes]
                if len(indices) > 0:
                    y_test_super[:, j] = np.max(y_test[:, indices], axis=1)
                    y_pred_super[:, j] = np.max(y_pred_best[:, indices], axis=1)

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            for j, sc in enumerate(superclasses):
                color = COLORS[sc]
                
                # ROC
                fpr, tpr, _ = roc_curve(y_test_super[:, j], y_pred_super[:, j])
                sc_auc = auc(fpr, tpr)
                axes[0].plot(fpr, tpr, color=color, linewidth=2, label=f'{sc} (AUC = {sc_auc:.3f})')

                # PR
                precision, recall, _ = precision_recall_curve(y_test_super[:, j], y_pred_super[:, j])
                sc_ap = average_precision_score(y_test_super[:, j], y_pred_super[:, j])
                axes[1].plot(recall, precision, color=color, linewidth=2, label=f'{sc} (AUPR = {sc_ap:.3f})')

            axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.5)
            axes[0].set_title(f"ResNet1d18: ROC Curve per Superclass", pad=15)
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].legend(loc='lower right')

            axes[1].set_title(f"ResNet1d18: Precision-Recall Curve per Superclass", pad=15)
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].legend(loc='lower left')

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'best_model_superclass_curves.png'), dpi=300)
            plt.close()

            # =========================================================================
            # PLOT 4: Misclassification Examples (Slide 9)
            # =========================================================================
            print("Generating Plot 4: Misclassification Examples...")
            # We align test_labels to match y_test indices
            db['all_scp_len'] = db.all_classes.apply(lambda x: len(x))
            test_db = db[(db.all_scp_len > 0) & (db.strat_fold == 10)]

            # 1. Severe False Negative for MI:
            # Patients with True MI (index 1) but predicted probability is extremely low,
            # and predicted NORM (index 0) is high
            fn_score = y_test_super[:, 1] * (1.0 - y_pred_super[:, 1]) * y_pred_super[:, 0]
            fn_idx = np.argmax(fn_score)
            fn_ecg_id = test_db.index[fn_idx]

            # 2. Severe False Positive for MI:
            # Patients with True NORM (index 0) but predicted probability of MI (index 1) is extremely high
            fp_score = y_test_super[:, 0] * y_pred_super[:, 1] * (1.0 - y_pred_super[:, 0])
            fp_idx = np.argmax(fp_score)
            fp_ecg_id = test_db.index[fp_idx]

            # Load the raw waveforms for plotting
            fn_signal = apply_butter_filter(raw_signals[fn_ecg_id - 1], filter_sos)
            fp_signal = apply_butter_filter(raw_signals[fp_ecg_id - 1], filter_sos)

            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # False Negative plot
            axes[0].plot(np.arange(1000)/fs, fn_signal[:, 1], color='#c0392b', linewidth=1.5)
            axes[0].set_title(
                f"Severe False Negative Case (ECG ID: {fn_ecg_id})\n"
                f"True Label: MI | Predictions -> MI: {y_pred_super[fn_idx, 1]:.3f}, NORM: {y_pred_super[fn_idx, 0]:.3f}",
                color='#c0392b', pad=10, fontsize=14
            )
            axes[0].set_ylabel("Amplitude (mV)")
            axes[0].set_xlabel("Time (s)")

            # False Positive plot
            axes[1].plot(np.arange(1000)/fs, fp_signal[:, 1], color='#d35400', linewidth=1.5)
            axes[1].set_title(
                f"Severe False Positive Case (ECG ID: {fp_ecg_id})\n"
                f"True Label: NORM | Predictions -> MI: {y_pred_super[fp_idx, 1]:.3f}, NORM: {y_pred_super[fp_idx, 0]:.3f}",
                color='#d35400', pad=10, fontsize=14
            )
            axes[1].set_ylabel("Amplitude (mV)")
            axes[1].set_xlabel("Time (s)")

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'misclassification_examples.png'), dpi=300)
            plt.close()
            print(f"Misclassification examples generated successfully. FN ECG ID: {fn_ecg_id}, FP ECG ID: {fp_ecg_id}")

    print(f"\nAll plots generated successfully and saved in '{plots_dir}'!")

if __name__ == "__main__":
    main()
